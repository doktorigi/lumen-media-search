"""Domain objects shared by Lumen's indexer and SQLite store.

The records in this module deliberately contain only values that are easy to
pass between threads (strings, integers, booleans, and immutable dataclasses).
Embeddings enter the store through :class:`SegmentInput`; persisted embeddings
are exposed separately as a dense NumPy matrix by the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

PathValue: TypeAlias = str | PathLike[str]


class MediaKind(StrEnum):
    """Media types understood by the first version of the index."""

    IMAGE = "image"
    VIDEO = "video"


class AssetStatus(StrEnum):
    """Built-in asset states.

    The database accepts other non-empty states so future indexer versions can
    add detail without a schema migration.  ``READY`` is retained as a friendly
    alias for callers that use that term; its persisted value is ``indexed``.
    """

    PENDING = "pending"
    INDEXED = "indexed"
    READY = "indexed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RootRecord:
    id: int
    path: str
    enabled: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AssetInput:
    """Metadata supplied by the scanner when writing an asset."""

    path: PathValue
    kind: MediaKind | str
    size: int
    mtime_ns: int
    root_id: int | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    status: AssetStatus | str = AssetStatus.INDEXED
    error: str | None = None
    offline: bool = False


# ``AssetUpsert`` reads naturally in indexer code and keeps compatibility with
# either naming convention without introducing a second runtime type.
AssetUpsert = AssetInput


@dataclass(frozen=True, slots=True)
class AssetRecord:
    id: int
    root_id: int | None
    path: str
    kind: MediaKind
    size: int
    mtime_ns: int
    width: int | None
    height: int | None
    duration_ms: int | None
    status: str
    error: str | None
    offline: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SegmentInput:
    """A sampled frame (or the sole frame for an image) and its embedding."""

    embedding: ArrayLike
    timestamp_ms: int = 0
    thumbnail_path: PathValue | None = None


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    id: int
    asset_id: int
    timestamp_ms: int
    thumbnail_path: str | None
    embedding_dim: int
    created_at: str


@dataclass(frozen=True, slots=True)
class SearchItem:
    """Metadata for the row at the same position in a loaded search matrix."""

    segment_id: int
    asset_id: int
    path: str
    kind: MediaKind
    size: int
    mtime_ns: int
    width: int | None
    height: int | None
    duration_ms: int | None
    timestamp_ms: int
    thumbnail_path: str | None
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class LibraryStats:
    root_count: int
    asset_count: int
    image_count: int
    video_count: int
    segment_count: int
    indexed_count: int
    error_count: int
    offline_skipped: int
    total_bytes: int

    # Short plural aliases make the record pleasant to consume in templates and
    # accommodate both common naming styles without duplicating stored values.
    @property
    def roots(self) -> int:
        return self.root_count

    @property
    def assets(self) -> int:
        return self.asset_count

    @property
    def total_assets(self) -> int:
        return self.asset_count

    @property
    def images(self) -> int:
        return self.image_count

    @property
    def videos(self) -> int:
        return self.video_count

    @property
    def segments(self) -> int:
        return self.segment_count

    @property
    def errors(self) -> int:
        return self.error_count

    @property
    def offline_count(self) -> int:
        return self.offline_skipped

    @property
    def offline(self) -> int:
        return self.offline_skipped


@dataclass(frozen=True, slots=True)
class SettingRecord:
    key: str
    value: Any
    updated_at: str


@dataclass(frozen=True, slots=True)
class IndexJobSummary:
    id: int
    root_id: int | None
    status: str
    started_at: str
    finished_at: str | None
    discovered_count: int
    indexed_count: int
    skipped_count: int
    error_count: int
    message: str | None


def as_float32_vector(value: ArrayLike) -> np.ndarray:
    """Return a validated, L2-normalized, little-endian float32 vector.

    Normalizing at the persistence boundary makes cosine search a matrix-vector
    multiplication and prevents corrupt NaN/zero vectors from entering the
    index.  A fresh contiguous array is always returned.
    """

    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "embedding must be numeric and convertible to float32"
        ) from exc

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError("embedding must contain only finite values")

    norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding must have a non-zero finite norm")

    # Normalize in float64 first.  A valid collection of large float32 values
    # can have a norm above float32's maximum, so narrowing the divisor before
    # division could otherwise turn the result into an all-zero vector.
    normalized = (vector.astype(np.float64, copy=False) / norm).astype("<f4")
    if not np.isfinite(normalized).all():
        raise ValueError("embedding could not be normalized safely")
    if not np.any(normalized):
        raise ValueError("embedding underflowed while normalizing")
    return np.ascontiguousarray(normalized, dtype=np.dtype("<f4"))
