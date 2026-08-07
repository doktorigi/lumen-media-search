from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

import lumen_media.indexer as indexer_module
from lumen_media.config import Settings
from lumen_media.embeddings import DeterministicTestEngine
from lumen_media.indexer import IndexJob, MediaIndexer
from lumen_media.media import MediaCandidate
from lumen_media.store import MediaStore


@pytest.fixture
def indexer_harness(
    tmp_path: Path,
) -> tuple[Settings, MediaStore, DeterministicTestEngine, MediaIndexer, Path]:
    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        image_batch_size=2,
        thumbnail_edge=24,
    )
    settings.ensure_directories()
    store = MediaStore(settings.database_path)
    store.initialize()
    engine = DeterministicTestEngine()
    indexer = MediaIndexer(settings, store, engine)
    media_root = tmp_path / "library"
    media_root.mkdir()
    return settings, store, engine, indexer, media_root


def _save_image(path: Path, size: tuple[int, int], color: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _run_index(indexer: MediaIndexer, root: Path, job_id: str) -> IndexJob:
    job = IndexJob(id=job_id, paths=(str(root),))
    indexer.run([root], job)
    assert job.wait(0)
    return job


def test_indexes_generated_images_end_to_end(indexer_harness) -> None:
    settings, store, _, indexer, media_root = indexer_harness
    expected = {
        media_root / "red.jpg": (13, 7),
        media_root / "green.png": (8, 11),
        media_root / "nested" / "blue.webp": (17, 9),
    }
    _save_image(media_root / "red.jpg", expected[media_root / "red.jpg"], "red")
    _save_image(media_root / "green.png", expected[media_root / "green.png"], "green")
    _save_image(
        media_root / "nested" / "blue.webp",
        expected[media_root / "nested" / "blue.webp"],
        "blue",
    )
    (media_root / "notes.txt").write_text("not media", encoding="utf-8")

    job = _run_index(indexer, media_root, "initial")

    assert job.state == "completed"
    assert (job.total, job.processed, job.indexed) == (3, 3, 3)
    assert (job.skipped, job.errors, job.offline_skipped) == (0, 0, 0)
    assert job.current is None
    assert job.message == "Indexed 3 assets; skipped 0; 0 errors."

    stats = store.library_stats()
    assert (stats.root_count, stats.asset_count, stats.image_count) == (1, 3, 3)
    assert (stats.indexed_count, stats.segment_count, stats.error_count) == (3, 3, 0)

    for path, dimensions in expected.items():
        asset = store.get_asset_by_path(path)
        assert asset is not None
        assert asset.status == "indexed"
        assert not asset.offline
        assert (asset.width, asset.height) == dimensions

        segments = store.list_segments(asset.id)
        assert len(segments) == 1
        assert segments[0].embedding_dim == DeterministicTestEngine.dimensions
        assert segments[0].timestamp_ms == 0
        thumbnail_path = Path(segments[0].thumbnail_path or "")
        assert thumbnail_path.parent == settings.thumbnails_dir
        assert thumbnail_path.is_file()
        with Image.open(thumbnail_path) as thumbnail:
            assert max(thumbnail.size) <= settings.thumbnail_edge

    items, matrix = store.load_search_matrix("image")
    assert len(items) == 3
    assert matrix.shape == (3, DeterministicTestEngine.dimensions)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_second_run_skips_unchanged_images(indexer_harness, monkeypatch) -> None:
    _, store, engine, indexer, media_root = indexer_harness
    paths = (media_root / "one.jpg", media_root / "two.png")
    _save_image(paths[0], (10, 6), "purple")
    _save_image(paths[1], (7, 12), "orange")

    first = _run_index(indexer, media_root, "first")
    assert (first.indexed, first.skipped, first.errors) == (2, 0, 0)
    segment_ids = {
        path: store.list_segments(store.get_asset_by_path(path).id)[0].id  # type: ignore[union-attr]
        for path in paths
    }

    encode_images = Mock(side_effect=AssertionError("unchanged images were encoded"))
    monkeypatch.setattr(engine, "encode_images", encode_images)
    second = _run_index(indexer, media_root, "second")

    assert second.state == "completed"
    assert (second.total, second.processed) == (2, 2)
    assert (second.indexed, second.skipped, second.errors) == (0, 2, 0)
    encode_images.assert_not_called()
    assert {
        path: store.list_segments(store.get_asset_by_path(path).id)[0].id  # type: ignore[union-attr]
        for path in paths
    } == segment_ids


def test_changed_image_is_reindexed(indexer_harness) -> None:
    _, store, _, indexer, media_root = indexer_harness
    image_path = media_root / "changing.png"
    _save_image(image_path, (9, 5), "red")

    first = _run_index(indexer, media_root, "before-change")
    assert first.indexed == 1
    original_asset = store.get_asset_by_path(image_path)
    assert original_asset is not None
    original_segment = store.list_segments(original_asset.id)[0]
    original_vector = store.load_search_matrix()[1][0].copy()

    _save_image(image_path, (9, 5), "blue")
    changed_mtime_ns = original_asset.mtime_ns + 1_000_000_000
    os.utime(image_path, ns=(changed_mtime_ns, changed_mtime_ns))

    second = _run_index(indexer, media_root, "after-change")

    assert second.state == "completed"
    assert (second.total, second.processed) == (1, 1)
    assert (second.indexed, second.skipped, second.errors) == (1, 0, 0)
    changed_asset = store.get_asset_by_path(image_path)
    assert changed_asset is not None
    assert changed_asset.id == original_asset.id
    assert changed_asset.mtime_ns == changed_mtime_ns
    changed_segment = store.list_segments(changed_asset.id)[0]
    assert changed_segment.thumbnail_path != original_segment.thumbnail_path
    assert not np.allclose(store.load_search_matrix()[1][0], original_vector)


def test_corrupt_image_records_error_and_job_continues(indexer_harness) -> None:
    _, store, _, indexer, media_root = indexer_harness
    valid_path = media_root / "valid.png"
    corrupt_path = media_root / "corrupt.jpg"
    _save_image(valid_path, (6, 4), "cyan")
    corrupt_path.write_bytes(b"this is not a JPEG")

    job = _run_index(indexer, media_root, "with-corrupt-image")

    assert job.state == "completed"
    assert (job.total, job.processed) == (2, 2)
    assert (job.indexed, job.skipped, job.errors) == (1, 0, 1)
    assert job.message == "Indexed 1 assets; skipped 0; 1 errors."

    valid = store.get_asset_by_path(valid_path)
    corrupt = store.get_asset_by_path(corrupt_path)
    assert valid is not None and valid.status == "indexed"
    assert len(store.list_segments(valid.id)) == 1
    assert corrupt is not None and corrupt.status == "error"
    assert corrupt.error is not None
    assert "UnidentifiedImageError" in corrupt.error
    assert store.list_segments(corrupt.id) == []
    assert store.library_stats().error_count == 1


def test_offline_candidate_is_recorded_without_opening_file(
    indexer_harness, monkeypatch
) -> None:
    _, store, _, indexer, media_root = indexer_harness
    candidate = MediaCandidate(
        path=media_root / "cloud-only.jpg",
        kind="image",
        size=123,
        mtime_ns=456,
        offline=True,
    )
    monkeypatch.setattr(
        indexer_module, "discover_media", lambda _roots: iter((candidate,))
    )
    load_image = Mock(side_effect=AssertionError("offline file content was opened"))
    monkeypatch.setattr(indexer_module, "load_image", load_image)

    job = _run_index(indexer, media_root, "offline")

    assert job.state == "completed"
    assert (job.total, job.processed) == (1, 1)
    assert (job.indexed, job.skipped, job.offline_skipped, job.errors) == (0, 1, 1, 0)
    load_image.assert_not_called()
    asset = store.get_asset_by_path(candidate.path)
    assert asset is not None
    assert asset.status == "offline"
    assert asset.offline
    assert store.list_segments(asset.id) == []


def test_index_job_to_dict_exposes_only_serializable_public_state() -> None:
    job = IndexJob(
        id="job-123",
        paths=(r"C:\photos", r"D:\videos"),
        state="indexing",
        processed=4,
        total=9,
        indexed=3,
        skipped=1,
        errors=2,
        offline_skipped=1,
        current=r"C:\photos\cat.jpg",
        message="Still working",
    )

    assert job.to_dict() == {
        "id": "job-123",
        "paths": [r"C:\photos", r"D:\videos"],
        "state": "indexing",
        "processed": 4,
        "total": 9,
        "indexed": 3,
        "skipped": 1,
        "errors": 2,
        "offline_skipped": 1,
        "current": r"C:\photos\cat.jpg",
        "message": "Still working",
    }
