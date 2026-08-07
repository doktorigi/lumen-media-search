from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .domain import AssetInput, AssetStatus, SegmentInput
from .embeddings import EmbeddingEngine
from .media import (
    MediaCandidate,
    discover_media,
    load_image,
    load_video_frames,
    thumbnail_name,
    write_thumbnail,
)
from .store import MediaStore


@dataclass(slots=True)
class IndexJob:
    id: str
    paths: tuple[str, ...]
    state: str = "queued"
    processed: int = 0
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    offline_skipped: int = 0
    current: str | None = None
    message: str | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "paths": list(self.paths),
            "state": self.state,
            "processed": self.processed,
            "total": self.total,
            "indexed": self.indexed,
            "skipped": self.skipped,
            "errors": self.errors,
            "offline_skipped": self.offline_skipped,
            "current": self.current,
            "message": self.message,
        }

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


@dataclass(slots=True)
class _PendingImage:
    root_id: int
    candidate: MediaCandidate
    image: Image.Image
    width: int
    height: int
    thumbnail_path: Path


class MediaIndexer:
    def __init__(
        self,
        settings: Settings,
        store: MediaStore,
        engine: EmbeddingEngine,
    ):
        self.settings = settings
        self.store = store
        self.engine = engine

    def run(self, paths: list[Path], job: IndexJob) -> None:
        try:
            job.state = "discovering"
            discovered: list[tuple[int, MediaCandidate]] = []
            seen_files: set[str] = set()
            for path in paths:
                root_path = path.expanduser().resolve(strict=True)
                if not root_path.is_dir():
                    raise ValueError(f"Not a directory: {root_path}")
                root = self.store.upsert_root(root_path)
                for candidate in discover_media((root_path,)):
                    key = str(candidate.path).casefold()
                    if key in seen_files:
                        continue
                    seen_files.add(key)
                    discovered.append((root.id, candidate))

            job.total = len(discovered)
            job.state = "indexing"
            pending_images: list[_PendingImage] = []

            for root_id, candidate in discovered:
                job.current = str(candidate.path)
                existing = self.store.get_asset_by_path(candidate.path)
                if self._is_unchanged(existing, candidate):
                    job.skipped += 1
                    job.processed += 1
                    continue

                if candidate.offline:
                    self.store.upsert_asset_and_replace_segments(
                        AssetInput(
                            path=candidate.path,
                            kind=candidate.kind,
                            size=candidate.size,
                            mtime_ns=candidate.mtime_ns,
                            root_id=root_id,
                            status="offline",
                            offline=True,
                        ),
                        (),
                    )
                    job.offline_skipped += 1
                    job.skipped += 1
                    job.processed += 1
                    continue

                if candidate.kind == "image":
                    try:
                        decoded = load_image(candidate.path)
                        thumb_path = self.settings.thumbnails_dir / thumbnail_name(
                            candidate.path, candidate.mtime_ns, 0
                        )
                        write_thumbnail(
                            decoded.image,
                            thumb_path,
                            edge=self.settings.thumbnail_edge,
                        )
                        pending_images.append(
                            _PendingImage(
                                root_id=root_id,
                                candidate=candidate,
                                image=decoded.image,
                                width=decoded.width,
                                height=decoded.height,
                                thumbnail_path=thumb_path,
                            )
                        )
                    except Exception as exc:
                        self._record_error(root_id, candidate, exc)
                        job.errors += 1
                        job.processed += 1
                    if len(pending_images) >= self.settings.image_batch_size:
                        self._flush_images(pending_images, job)
                        pending_images.clear()
                else:
                    if pending_images:
                        self._flush_images(pending_images, job)
                        pending_images.clear()
                    self._process_video(root_id, candidate, job)

            if pending_images:
                self._flush_images(pending_images, job)
            job.current = None
            job.state = "completed"
            job.message = (
                f"Indexed {job.indexed} assets; skipped {job.skipped}; "
                f"{job.errors} errors."
            )
        except Exception as exc:
            job.state = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
        finally:
            job.current = None
            job._done.set()

    @staticmethod
    def _is_unchanged(existing, candidate: MediaCandidate) -> bool:
        if existing is None:
            return False
        if existing.size != candidate.size or existing.mtime_ns != candidate.mtime_ns:
            return False
        if candidate.offline:
            return existing.offline
        return existing.status in {AssetStatus.INDEXED.value, AssetStatus.READY.value}

    def _flush_images(self, pending: list[_PendingImage], job: IndexJob) -> None:
        try:
            embeddings = self.engine.encode_images([item.image for item in pending])
            if embeddings.shape[0] != len(pending):
                raise RuntimeError("embedding engine returned the wrong batch length")
        except Exception as exc:
            for item in pending:
                self._record_error(item.root_id, item.candidate, exc)
                job.errors += 1
                job.processed += 1
            return

        for item, embedding in zip(pending, embeddings, strict=True):
            candidate = item.candidate
            try:
                self.store.upsert_asset_and_replace_segments(
                    AssetInput(
                        path=candidate.path,
                        kind=candidate.kind,
                        size=candidate.size,
                        mtime_ns=candidate.mtime_ns,
                        root_id=item.root_id,
                        width=item.width,
                        height=item.height,
                    ),
                    (
                        SegmentInput(
                            embedding=embedding,
                            timestamp_ms=0,
                            thumbnail_path=item.thumbnail_path,
                        ),
                    ),
                )
                job.indexed += 1
            except Exception as exc:
                self._record_error(item.root_id, candidate, exc)
                job.errors += 1
            finally:
                job.processed += 1

    def _process_video(
        self, root_id: int, candidate: MediaCandidate, job: IndexJob
    ) -> None:
        try:
            decoded = load_video_frames(
                candidate.path, max_frames=self.settings.video_max_frames
            )
            vectors: list[np.ndarray] = []
            frames = list(decoded.frames)
            for start in range(0, len(frames), self.settings.video_batch_size):
                batch = frames[start : start + self.settings.video_batch_size]
                encoded = self.engine.encode_images([frame for _, frame in batch])
                vectors.extend(encoded)
            if len(vectors) != len(frames):
                raise RuntimeError(
                    "embedding engine returned the wrong video batch length"
                )

            segments: list[SegmentInput] = []
            for (timestamp_ms, frame), vector in zip(frames, vectors, strict=True):
                thumb_path = self.settings.thumbnails_dir / thumbnail_name(
                    candidate.path, candidate.mtime_ns, timestamp_ms
                )
                write_thumbnail(frame, thumb_path, edge=self.settings.thumbnail_edge)
                segments.append(
                    SegmentInput(
                        embedding=vector,
                        timestamp_ms=timestamp_ms,
                        thumbnail_path=thumb_path,
                    )
                )
            self.store.upsert_asset_and_replace_segments(
                AssetInput(
                    path=candidate.path,
                    kind=candidate.kind,
                    size=candidate.size,
                    mtime_ns=candidate.mtime_ns,
                    root_id=root_id,
                    width=decoded.width,
                    height=decoded.height,
                    duration_ms=decoded.duration_ms,
                ),
                segments,
            )
            job.indexed += 1
        except Exception as exc:
            self._record_error(root_id, candidate, exc)
            job.errors += 1
        finally:
            job.processed += 1

    def _record_error(
        self, root_id: int, candidate: MediaCandidate, error: Exception
    ) -> None:
        message = f"{type(error).__name__}: {error}"[:1000]
        self.store.mark_asset_error(
            candidate.path,
            message,
            kind=candidate.kind,
            size=candidate.size,
            mtime_ns=candidate.mtime_ns,
            root_id=root_id,
            offline=candidate.offline,
        )


class IndexJobManager:
    def __init__(self, indexer: MediaIndexer):
        self.indexer = indexer
        self._lock = threading.Lock()
        self._jobs: dict[str, IndexJob] = {}
        self._active_id: str | None = None

    def start(self, paths: list[Path]) -> IndexJob:
        if not paths:
            raise ValueError("At least one folder is required")
        with self._lock:
            active = self._jobs.get(self._active_id or "")
            if active is not None and active.state not in {"completed", "failed"}:
                raise RuntimeError("An index job is already running")
            job = IndexJob(
                id=uuid.uuid4().hex,
                paths=tuple(str(path) for path in paths),
            )
            self._jobs[job.id] = job
            self._active_id = job.id
        thread = threading.Thread(
            target=self.indexer.run,
            args=(paths, job),
            name=f"lumen-index-{job.id[:8]}",
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> IndexJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> IndexJob | None:
        with self._lock:
            return self._jobs.get(self._active_id or "")
