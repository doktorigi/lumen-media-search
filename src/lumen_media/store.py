"""Thread-safe SQLite persistence for the local semantic media index."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .domain import (
    AssetInput,
    AssetRecord,
    AssetStatus,
    IndexJobSummary,
    LibraryStats,
    MediaKind,
    PathValue,
    RootRecord,
    SearchItem,
    SegmentInput,
    SegmentRecord,
    SettingRecord,
    as_float32_vector,
)

SCHEMA_VERSION = 1


class EmbeddingValidationError(ValueError):
    """Raised when stored vectors cannot form a safe dense search matrix."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS roots (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    enabled     INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    id           INTEGER PRIMARY KEY,
    root_id      INTEGER REFERENCES roots(id) ON DELETE SET NULL,
    path         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL CHECK (kind IN ('image', 'video')),
    size         INTEGER NOT NULL CHECK (size >= 0),
    mtime_ns     INTEGER NOT NULL CHECK (mtime_ns >= 0),
    width        INTEGER CHECK (width IS NULL OR width > 0),
    height       INTEGER CHECK (height IS NULL OR height > 0),
    duration_ms  INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    status       TEXT NOT NULL,
    error        TEXT,
    offline      INTEGER NOT NULL DEFAULT 0 CHECK (offline IN (0, 1)),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS assets_root_id_idx ON assets(root_id);
CREATE INDEX IF NOT EXISTS assets_kind_status_idx
    ON assets(kind, status, offline);

CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY,
    asset_id        INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    timestamp_ms    INTEGER NOT NULL DEFAULT 0 CHECK (timestamp_ms >= 0),
    thumbnail_path  TEXT,
    embedding       BLOB NOT NULL,
    embedding_dim   INTEGER NOT NULL CHECK (embedding_dim > 0),
    created_at      TEXT NOT NULL,
    CHECK (length(embedding) = embedding_dim * 4)
);

CREATE INDEX IF NOT EXISTS segments_asset_id_idx
    ON segments(asset_id, timestamp_ms, id);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_jobs (
    id                INTEGER PRIMARY KEY,
    root_id           INTEGER REFERENCES roots(id) ON DELETE SET NULL,
    status            TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    discovered_count  INTEGER NOT NULL DEFAULT 0 CHECK (discovered_count >= 0),
    indexed_count     INTEGER NOT NULL DEFAULT 0 CHECK (indexed_count >= 0),
    skipped_count     INTEGER NOT NULL DEFAULT 0 CHECK (skipped_count >= 0),
    error_count       INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0),
    message           TEXT
);

CREATE INDEX IF NOT EXISTS index_jobs_started_at_idx
    ON index_jobs(started_at DESC, id DESC);
"""


def canonicalize_path(path: PathValue) -> str:
    """Return the platform-canonical absolute representation of ``path``."""

    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must be a non-empty string or path-like value")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    resolved = str(Path(expanded).resolve(strict=False))
    # normcase provides the case-insensitive identity Windows filesystems need;
    # on case-sensitive platforms it leaves case intact.
    return os.path.normcase(os.path.normpath(resolved))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _as_nonnegative_int(
    name: str, value: object, *, optional: bool = False
) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _as_positive_int(name: str, value: object, *, optional: bool = False) -> int | None:
    result = _as_nonnegative_int(name, value, optional=optional)
    if result is not None and result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_asset(asset: AssetInput) -> AssetInput:
    try:
        kind = MediaKind(str(asset.kind))
    except ValueError as exc:
        raise ValueError("kind must be 'image' or 'video'") from exc

    status = str(asset.status).strip()
    if not status:
        raise ValueError("status must be non-empty")

    root_id = _as_positive_int("root_id", asset.root_id, optional=True)
    size = _as_nonnegative_int("size", asset.size)
    mtime_ns = _as_nonnegative_int("mtime_ns", asset.mtime_ns)
    width = _as_positive_int("width", asset.width, optional=True)
    height = _as_positive_int("height", asset.height, optional=True)
    duration_ms = _as_nonnegative_int("duration_ms", asset.duration_ms, optional=True)
    if not isinstance(asset.offline, (bool, np.bool_)):
        raise ValueError("offline must be a boolean")

    return replace(
        asset,
        path=canonicalize_path(asset.path),
        kind=kind,
        size=size,
        mtime_ns=mtime_ns,
        root_id=root_id,
        width=width,
        height=height,
        duration_ms=duration_ms,
        status=status,
        error=None if asset.error is None else str(asset.error),
        offline=bool(asset.offline),
    )


class MediaStore:
    """SQLite-backed media catalog.

    No connection is shared between calls.  This makes a single ``MediaStore``
    safe to use from the UI, scanner, and indexing worker threads concurrently;
    WAL mode and a busy timeout handle the resulting reader/writer overlap.
    """

    def __init__(self, db_path: PathValue, *, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.db_path = Path(
            os.path.expandvars(os.path.expanduser(os.fspath(db_path)))
        ).resolve(strict=False)
        self.timeout = float(timeout)

    def initialize(self) -> None:
        """Create or upgrade the database and enable WAL journaling."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._open_connection()
        try:
            mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
            if mode != "wal":
                raise RuntimeError(
                    f"SQLite could not enable WAL mode (reported {mode!r})"
                )
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(_SCHEMA)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        finally:
            connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={max(1, int(self.timeout * 1000))}")
        return connection

    @contextmanager
    def _operation(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_root(self, path: PathValue, *, enabled: bool = True) -> RootRecord:
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        canonical_path = canonicalize_path(path)
        now = _now()
        with self._operation(write=True) as connection:
            connection.execute(
                """
                INSERT INTO roots(path, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (canonical_path, int(enabled), now, now),
            )
            row = connection.execute(
                "SELECT * FROM roots WHERE path = ?", (canonical_path,)
            ).fetchone()
        assert row is not None
        return self._root_from_row(row)

    def list_roots(self) -> list[RootRecord]:
        with self._operation() as connection:
            rows = connection.execute("SELECT * FROM roots ORDER BY path").fetchall()
        return [self._root_from_row(row) for row in rows]

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        asset_id = _as_positive_int("asset_id", asset_id)  # type: ignore[assignment]
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return None if row is None else self._asset_from_row(row)

    def get_asset_by_path(self, path: PathValue) -> AssetRecord | None:
        canonical_path = canonicalize_path(path)
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE path = ?", (canonical_path,)
            ).fetchone()
        return None if row is None else self._asset_from_row(row)

    def upsert_asset_and_replace_segments(
        self,
        asset: AssetInput | PathValue | None = None,
        segments: Iterable[SegmentInput | Mapping[str, object]] = (),
        *,
        path: PathValue | None = None,
        kind: MediaKind | str | None = None,
        size: int | None = None,
        mtime_ns: int | None = None,
        root_id: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_ms: int | None = None,
        status: AssetStatus | str = AssetStatus.INDEXED,
        error: str | None = None,
        offline: bool = False,
    ) -> AssetRecord:
        """Upsert one asset and replace all of its segments atomically.

        Callers may pass an :class:`AssetInput`, or supply the fields as keyword
        arguments.  Empty ``segments`` is valid, which is important for
        placeholders and known-offline cloud files that must not be hydrated.
        """

        if isinstance(asset, AssetInput):
            if path is not None:
                raise TypeError("path cannot be supplied together with AssetInput")
            asset_input = asset
        else:
            if asset is not None:
                if path is not None:
                    raise TypeError("path was supplied twice")
                path = asset
            if path is None or kind is None or size is None or mtime_ns is None:
                raise TypeError("path, kind, size, and mtime_ns are required")
            asset_input = AssetInput(
                path=path,
                kind=kind,
                size=size,
                mtime_ns=mtime_ns,
                root_id=root_id,
                width=width,
                height=height,
                duration_ms=duration_ms,
                status=status,
                error=error,
                offline=offline,
            )

        validated_asset = _validate_asset(asset_input)
        prepared_segments = self._prepare_segments(segments)
        now = _now()

        with self._operation(write=True) as connection:
            asset_id = self._upsert_asset_row(connection, validated_asset, now)
            connection.execute("DELETE FROM segments WHERE asset_id = ?", (asset_id,))
            connection.executemany(
                """
                INSERT INTO segments(
                    asset_id, timestamp_ms, thumbnail_path,
                    embedding, embedding_dim, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        asset_id,
                        timestamp_ms,
                        thumbnail_path,
                        sqlite3.Binary(vector.tobytes(order="C")),
                        int(vector.size),
                        now,
                    )
                    for timestamp_ms, thumbnail_path, vector in prepared_segments
                ],
            )
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()

        assert row is not None
        return self._asset_from_row(row)

    def mark_asset_error(
        self,
        asset: int | PathValue | AssetInput,
        error: str,
        *,
        kind: MediaKind | str | None = None,
        size: int | None = None,
        mtime_ns: int | None = None,
        root_id: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_ms: int | None = None,
        offline: bool | None = None,
    ) -> AssetRecord:
        """Mark an existing asset failed, or create a failed row from metadata.

        Existing segments are retained transactionally but excluded from search
        while the asset is in the error state.  This permits a later successful
        retry to replace them without losing the last durable index on failure.
        """

        message = str(error).strip()
        if not message:
            raise ValueError("error must be a non-empty message")
        if offline is not None and not isinstance(offline, (bool, np.bool_)):
            raise ValueError("offline must be a boolean")
        now = _now()

        with self._operation(write=True) as connection:
            if isinstance(asset, AssetInput):
                failed_asset = _validate_asset(
                    replace(asset, status=AssetStatus.ERROR, error=message)
                )
                asset_id = self._upsert_asset_row(connection, failed_asset, now)
            elif isinstance(asset, int) and not isinstance(asset, bool):
                asset_id = _as_positive_int("asset_id", asset)  # type: ignore[assignment]
                row = connection.execute(
                    "SELECT * FROM assets WHERE id = ?", (asset_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"asset {asset_id} does not exist")
                failed_asset = self._failed_asset_from_existing(
                    row,
                    message=message,
                    kind=kind,
                    size=size,
                    mtime_ns=mtime_ns,
                    root_id=root_id,
                    width=width,
                    height=height,
                    duration_ms=duration_ms,
                    offline=offline,
                )
                asset_id = self._upsert_asset_row(connection, failed_asset, now)
            else:
                canonical_path = canonicalize_path(asset)
                row = connection.execute(
                    "SELECT * FROM assets WHERE path = ?", (canonical_path,)
                ).fetchone()
                if row is None:
                    if kind is None or size is None or mtime_ns is None:
                        raise KeyError(
                            "unknown asset path; kind, size, and mtime_ns are required to create it"
                        )
                    failed_asset = _validate_asset(
                        AssetInput(
                            path=canonical_path,
                            kind=kind,
                            size=size,
                            mtime_ns=mtime_ns,
                            root_id=root_id,
                            width=width,
                            height=height,
                            duration_ms=duration_ms,
                            status=AssetStatus.ERROR,
                            error=message,
                            offline=False if offline is None else offline,
                        )
                    )
                    asset_id = self._upsert_asset_row(connection, failed_asset, now)
                else:
                    failed_asset = self._failed_asset_from_existing(
                        row,
                        message=message,
                        kind=kind,
                        size=size,
                        mtime_ns=mtime_ns,
                        root_id=root_id,
                        width=width,
                        height=height,
                        duration_ms=duration_ms,
                        offline=offline,
                    )
                    asset_id = self._upsert_asset_row(connection, failed_asset, now)

            result = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()

        assert result is not None
        return self._asset_from_row(result)

    def list_segments(self, asset_id: int) -> list[SegmentRecord]:
        """Return segment metadata for diagnostics and focused indexer tests."""

        asset_id = _as_positive_int("asset_id", asset_id)  # type: ignore[assignment]
        with self._operation() as connection:
            rows = connection.execute(
                """
                SELECT id, asset_id, timestamp_ms, thumbnail_path,
                       embedding_dim, created_at
                FROM segments
                WHERE asset_id = ?
                ORDER BY timestamp_ms, id
                """,
                (asset_id,),
            ).fetchall()
        return [self._segment_from_row(row) for row in rows]

    def get_segment(self, segment_id: int) -> SegmentRecord | None:
        """Resolve one segment, primarily for safe thumbnail serving by ID."""

        segment_id = _as_positive_int("segment_id", segment_id)  # type: ignore[assignment]
        with self._operation() as connection:
            row = connection.execute(
                """
                SELECT id, asset_id, timestamp_ms, thumbnail_path,
                       embedding_dim, created_at
                FROM segments
                WHERE id = ?
                """,
                (segment_id,),
            ).fetchone()
        return None if row is None else self._segment_from_row(row)

    def library_stats(self) -> LibraryStats:
        with self._operation() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM roots) AS root_count,
                    count(*) AS asset_count,
                    coalesce(sum(kind = 'image'), 0) AS image_count,
                    coalesce(sum(kind = 'video'), 0) AS video_count,
                    (SELECT count(*) FROM segments) AS segment_count,
                    coalesce(sum(status IN ('indexed', 'ready') AND offline = 0), 0)
                        AS indexed_count,
                    coalesce(sum(status = 'error'), 0) AS error_count,
                    coalesce(sum(offline = 1), 0) AS offline_skipped,
                    coalesce(sum(size), 0) AS total_bytes
                FROM assets
                """
            ).fetchone()
        assert row is not None
        return LibraryStats(
            root_count=int(row["root_count"]),
            asset_count=int(row["asset_count"]),
            image_count=int(row["image_count"]),
            video_count=int(row["video_count"]),
            segment_count=int(row["segment_count"]),
            indexed_count=int(row["indexed_count"]),
            error_count=int(row["error_count"]),
            offline_skipped=int(row["offline_skipped"]),
            total_bytes=int(row["total_bytes"]),
        )

    def load_search_matrix(
        self, kind: MediaKind | str | None = None
    ) -> tuple[list[SearchItem], np.ndarray]:
        """Load searchable metadata and a normalized contiguous float32 matrix."""

        if kind is None:
            kind_value = None
        else:
            try:
                kind_value = MediaKind(str(kind)).value
            except ValueError as exc:
                raise ValueError("kind must be 'image', 'video', or None") from exc

        sql = """
            SELECT
                s.id AS segment_id,
                s.asset_id,
                s.timestamp_ms,
                s.thumbnail_path,
                s.embedding,
                s.embedding_dim,
                a.path,
                a.kind,
                a.size,
                a.mtime_ns,
                a.width,
                a.height,
                a.duration_ms
            FROM segments AS s
            JOIN assets AS a ON a.id = s.asset_id
            WHERE a.offline = 0
              AND a.status IN ('indexed', 'ready')
        """
        parameters: tuple[object, ...] = ()
        if kind_value is not None:
            sql += " AND a.kind = ?"
            parameters = (kind_value,)
        sql += " ORDER BY a.id, s.timestamp_ms, s.id"

        with self._operation() as connection:
            rows = connection.execute(sql, parameters).fetchall()

        if not rows:
            return [], np.empty((0, 0), dtype=np.float32, order="C")

        dimensions = {int(row["embedding_dim"]) for row in rows}
        if len(dimensions) != 1:
            raise EmbeddingValidationError(
                f"search index contains mixed embedding dimensions: {sorted(dimensions)}"
            )
        dimension = dimensions.pop()
        matrix = np.empty((len(rows), dimension), dtype=np.float32, order="C")
        items: list[SearchItem] = []

        for index, row in enumerate(rows):
            blob = bytes(row["embedding"])
            if len(blob) != dimension * np.dtype("<f4").itemsize:
                raise EmbeddingValidationError(
                    f"segment {row['segment_id']} has a corrupt embedding payload"
                )
            vector = np.frombuffer(blob, dtype="<f4", count=dimension)
            try:
                matrix[index] = as_float32_vector(vector)
            except ValueError as exc:
                raise EmbeddingValidationError(
                    f"segment {row['segment_id']} has an invalid embedding: {exc}"
                ) from exc
            items.append(self._search_item_from_row(row))

        return items, np.ascontiguousarray(matrix, dtype=np.float32)

    def set_setting(self, key: str, value: Any) -> SettingRecord:
        """Persist a JSON-compatible application setting."""

        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("setting key must be non-empty")
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("setting value must be JSON serializable") from exc
        now = _now()
        with self._operation(write=True) as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (normalized_key, encoded, now),
            )
        return SettingRecord(normalized_key, value, now)

    def get_setting_record(self, key: str) -> SettingRecord | None:
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("setting key must be non-empty")
        with self._operation() as connection:
            row = connection.execute(
                "SELECT key, value_json, updated_at FROM settings WHERE key = ?",
                (normalized_key,),
            ).fetchone()
        if row is None:
            return None
        return SettingRecord(
            key=str(row["key"]),
            value=json.loads(str(row["value_json"])),
            updated_at=str(row["updated_at"]),
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        record = self.get_setting_record(key)
        return default if record is None else record.value

    def record_index_job(
        self,
        *,
        status: str,
        root_id: int | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        discovered_count: int = 0,
        indexed_count: int = 0,
        skipped_count: int = 0,
        error_count: int = 0,
        message: str | None = None,
    ) -> IndexJobSummary:
        """Append a compact, durable summary of an indexing run."""

        normalized_status = str(status).strip()
        if not normalized_status:
            raise ValueError("status must be non-empty")
        root_id = _as_positive_int("root_id", root_id, optional=True)
        discovered_count = _as_nonnegative_int("discovered_count", discovered_count)  # type: ignore[assignment]
        indexed_count = _as_nonnegative_int("indexed_count", indexed_count)  # type: ignore[assignment]
        skipped_count = _as_nonnegative_int("skipped_count", skipped_count)  # type: ignore[assignment]
        error_count = _as_nonnegative_int("error_count", error_count)  # type: ignore[assignment]
        started_at = _now() if started_at is None else str(started_at)

        with self._operation(write=True) as connection:
            cursor = connection.execute(
                """
                INSERT INTO index_jobs(
                    root_id, status, started_at, finished_at,
                    discovered_count, indexed_count, skipped_count,
                    error_count, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    root_id,
                    normalized_status,
                    started_at,
                    finished_at,
                    discovered_count,
                    indexed_count,
                    skipped_count,
                    error_count,
                    message,
                ),
            )
            job_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        assert row is not None
        return self._job_from_row(row)

    def latest_index_job(self) -> IndexJobSummary | None:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM index_jobs ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    @staticmethod
    def _prepare_segments(
        segments: Iterable[SegmentInput | Mapping[str, object]],
    ) -> list[tuple[int, str | None, np.ndarray]]:
        prepared: list[tuple[int, str | None, np.ndarray]] = []
        expected_dimension: int | None = None
        for position, value in enumerate(segments):
            if isinstance(value, Mapping):
                try:
                    segment = SegmentInput(**value)  # type: ignore[arg-type]
                except TypeError as exc:
                    raise ValueError(
                        f"invalid segment at position {position}: {exc}"
                    ) from exc
            elif isinstance(value, SegmentInput):
                segment = value
            else:
                raise ValueError(
                    f"segment at position {position} must be SegmentInput or a mapping"
                )

            timestamp_ms = _as_nonnegative_int("timestamp_ms", segment.timestamp_ms)
            thumbnail_path = (
                None
                if segment.thumbnail_path is None
                else os.fspath(segment.thumbnail_path)
            )
            if thumbnail_path is not None and not isinstance(thumbnail_path, str):
                raise ValueError("thumbnail_path must be a string or path-like value")
            try:
                vector = as_float32_vector(segment.embedding)
            except ValueError as exc:
                raise ValueError(
                    f"invalid embedding at segment {position}: {exc}"
                ) from exc
            if expected_dimension is None:
                expected_dimension = int(vector.size)
            elif vector.size != expected_dimension:
                raise ValueError(
                    "all replacement segments for an asset must have the same embedding dimension"
                )
            prepared.append((timestamp_ms, thumbnail_path, vector))  # type: ignore[arg-type]
        return prepared

    @staticmethod
    def _upsert_asset_row(
        connection: sqlite3.Connection, asset: AssetInput, now: str
    ) -> int:
        connection.execute(
            """
            INSERT INTO assets(
                root_id, path, kind, size, mtime_ns, width, height,
                duration_ms, status, error, offline, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                root_id = COALESCE(excluded.root_id, assets.root_id),
                kind = excluded.kind,
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                width = excluded.width,
                height = excluded.height,
                duration_ms = excluded.duration_ms,
                status = excluded.status,
                error = excluded.error,
                offline = excluded.offline,
                updated_at = excluded.updated_at
            """,
            (
                asset.root_id,
                os.fspath(asset.path),
                str(asset.kind),
                asset.size,
                asset.mtime_ns,
                asset.width,
                asset.height,
                asset.duration_ms,
                str(asset.status),
                asset.error,
                int(asset.offline),
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT id FROM assets WHERE path = ?", (os.fspath(asset.path),)
        ).fetchone()
        assert row is not None
        return int(row["id"])

    @staticmethod
    def _failed_asset_from_existing(
        row: sqlite3.Row,
        *,
        message: str,
        kind: MediaKind | str | None,
        size: int | None,
        mtime_ns: int | None,
        root_id: int | None,
        width: int | None,
        height: int | None,
        duration_ms: int | None,
        offline: bool | None,
    ) -> AssetInput:
        """Merge optional scanner metadata into an existing failed asset."""

        return _validate_asset(
            AssetInput(
                path=str(row["path"]),
                kind=str(row["kind"]) if kind is None else kind,
                size=int(row["size"]) if size is None else size,
                mtime_ns=int(row["mtime_ns"]) if mtime_ns is None else mtime_ns,
                root_id=(
                    (None if row["root_id"] is None else int(row["root_id"]))
                    if root_id is None
                    else root_id
                ),
                width=(None if row["width"] is None else int(row["width"]))
                if width is None
                else width,
                height=(None if row["height"] is None else int(row["height"]))
                if height is None
                else height,
                duration_ms=(
                    None if row["duration_ms"] is None else int(row["duration_ms"])
                )
                if duration_ms is None
                else duration_ms,
                status=AssetStatus.ERROR,
                error=message,
                offline=bool(row["offline"]) if offline is None else offline,
            )
        )

    @staticmethod
    def _root_from_row(row: sqlite3.Row) -> RootRecord:
        return RootRecord(
            id=int(row["id"]),
            path=str(row["path"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _asset_from_row(row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            id=int(row["id"]),
            root_id=None if row["root_id"] is None else int(row["root_id"]),
            path=str(row["path"]),
            kind=MediaKind(str(row["kind"])),
            size=int(row["size"]),
            mtime_ns=int(row["mtime_ns"]),
            width=None if row["width"] is None else int(row["width"]),
            height=None if row["height"] is None else int(row["height"]),
            duration_ms=(
                None if row["duration_ms"] is None else int(row["duration_ms"])
            ),
            status=str(row["status"]),
            error=None if row["error"] is None else str(row["error"]),
            offline=bool(row["offline"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _segment_from_row(row: sqlite3.Row) -> SegmentRecord:
        return SegmentRecord(
            id=int(row["id"]),
            asset_id=int(row["asset_id"]),
            timestamp_ms=int(row["timestamp_ms"]),
            thumbnail_path=(
                None if row["thumbnail_path"] is None else str(row["thumbnail_path"])
            ),
            embedding_dim=int(row["embedding_dim"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _search_item_from_row(row: sqlite3.Row) -> SearchItem:
        return SearchItem(
            segment_id=int(row["segment_id"]),
            asset_id=int(row["asset_id"]),
            path=str(row["path"]),
            kind=MediaKind(str(row["kind"])),
            size=int(row["size"]),
            mtime_ns=int(row["mtime_ns"]),
            width=None if row["width"] is None else int(row["width"]),
            height=None if row["height"] is None else int(row["height"]),
            duration_ms=(
                None if row["duration_ms"] is None else int(row["duration_ms"])
            ),
            timestamp_ms=int(row["timestamp_ms"]),
            thumbnail_path=(
                None if row["thumbnail_path"] is None else str(row["thumbnail_path"])
            ),
            embedding_dim=int(row["embedding_dim"]),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> IndexJobSummary:
        return IndexJobSummary(
            id=int(row["id"]),
            root_id=None if row["root_id"] is None else int(row["root_id"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            finished_at=(
                None if row["finished_at"] is None else str(row["finished_at"])
            ),
            discovered_count=int(row["discovered_count"]),
            indexed_count=int(row["indexed_count"]),
            skipped_count=int(row["skipped_count"]),
            error_count=int(row["error_count"]),
            message=None if row["message"] is None else str(row["message"]),
        )


# A compact alias for callers that prefer the conventional repository name.
SQLiteStore = MediaStore
