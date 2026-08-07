from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from lumen_media.domain import AssetInput, MediaKind, SegmentInput
from lumen_media.store import EmbeddingValidationError, MediaStore, canonicalize_path


@pytest.fixture
def store(tmp_path) -> MediaStore:
    result = MediaStore(tmp_path / "catalog.sqlite3")
    result.initialize()
    return result


def test_initialize_uses_wal_and_roots_are_canonical_and_unique(store, tmp_path):
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"roots", "assets", "segments", "settings", "index_jobs"} <= tables

    first = store.upsert_root(tmp_path / "photos" / ".." / "photos")
    second = store.upsert_root(tmp_path / "photos", enabled=False)

    assert first.id == second.id
    assert second.path == canonicalize_path(tmp_path / "photos")
    assert second.enabled is False
    assert store.list_roots() == [second]


def test_asset_and_segment_replacement_is_one_transaction(store, tmp_path):
    root = store.upsert_root(tmp_path)
    path = tmp_path / "photo.jpg"
    original = store.upsert_asset_and_replace_segments(
        AssetInput(path, "image", size=10, mtime_ns=100, root_id=root.id),
        [SegmentInput([3.0, 4.0], thumbnail_path="old.jpg")],
    )

    # Force a failure after the asset upsert and old-segment deletion. SQLite
    # must roll the entire replacement back, not leave a half-written asset.
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_test_segment
            BEFORE INSERT ON segments
            WHEN NEW.timestamp_ms = 999
            BEGIN
                SELECT RAISE(ABORT, 'test insert failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="test insert failure"):
        store.upsert_asset_and_replace_segments(
            AssetInput(path, "image", size=999, mtime_ns=999, root_id=root.id),
            [SegmentInput([1.0, 0.0], timestamp_ms=999, thumbnail_path="new.jpg")],
        )

    unchanged = store.get_asset(original.id)
    segments = store.list_segments(original.id)
    assert unchanged is not None
    assert (unchanged.size, unchanged.mtime_ns) == (10, 100)
    assert [(segment.timestamp_ms, segment.thumbnail_path) for segment in segments] == [
        (0, "old.jpg")
    ]
    assert store.get_segment(segments[0].id) == segments[0]
    assert store.get_segment(segments[0].id + 10_000) is None


def test_stats_count_kinds_errors_and_offline_placeholders(store, tmp_path):
    image = store.upsert_asset_and_replace_segments(
        path=tmp_path / "one.jpg",
        kind="image",
        size=10,
        mtime_ns=1,
        segments=[SegmentInput([1.0, 0.0])],
    )
    store.upsert_asset_and_replace_segments(
        path=tmp_path / "cloud.mp4",
        kind="video",
        size=25,
        mtime_ns=2,
        status="offline",
        offline=True,
        segments=[],
    )
    store.mark_asset_error(image.id, "decode failed")

    stats = store.library_stats()
    assert stats.asset_count == stats.total_assets == 2
    assert stats.image_count == stats.images == 1
    assert stats.video_count == stats.videos == 1
    assert stats.segment_count == 1
    assert stats.indexed_count == 0
    assert stats.error_count == stats.errors == 1
    assert stats.offline_skipped == stats.offline_count == 1
    assert stats.total_bytes == 35


def test_kind_filter_returns_matching_normalized_contiguous_matrix(store, tmp_path):
    store.upsert_asset_and_replace_segments(
        path=tmp_path / "one.jpg",
        kind=MediaKind.IMAGE,
        size=1,
        mtime_ns=1,
        segments=[SegmentInput([3.0, 4.0])],
    )
    video = store.upsert_asset_and_replace_segments(
        path=tmp_path / "clip.mp4",
        kind=MediaKind.VIDEO,
        size=2,
        mtime_ns=2,
        width=1920,
        height=1080,
        duration_ms=2_000,
        segments=[
            SegmentInput([0.0, 5.0], timestamp_ms=0),
            SegmentInput([8.0, 6.0], timestamp_ms=1_000),
        ],
    )

    items, matrix = store.load_search_matrix("video")
    assert [item.asset_id for item in items] == [video.id, video.id]
    assert [item.timestamp_ms for item in items] == [0, 1_000]
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
    np.testing.assert_allclose(matrix, [[0.0, 1.0], [0.8, 0.6]])


def test_embedding_and_kind_errors_are_clear_and_do_not_mutate(store, tmp_path):
    asset = store.upsert_asset_and_replace_segments(
        path=tmp_path / "one.jpg",
        kind="image",
        size=1,
        mtime_ns=1,
        segments=[SegmentInput([1.0, 0.0])],
    )

    with pytest.raises(ValueError, match="finite"):
        store.upsert_asset_and_replace_segments(
            path=tmp_path / "one.jpg",
            kind="image",
            size=2,
            mtime_ns=2,
            segments=[SegmentInput([np.nan, 1.0])],
        )
    assert store.get_asset(asset.id).size == 1  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="same embedding dimension"):
        store.upsert_asset_and_replace_segments(
            path=tmp_path / "two.jpg",
            kind="image",
            size=1,
            mtime_ns=1,
            segments=[SegmentInput([1.0, 0.0]), SegmentInput([1.0, 0.0, 0.0])],
        )
    with pytest.raises(ValueError, match="kind"):
        store.load_search_matrix("audio")


def test_mixed_stored_dimensions_raise_on_unfiltered_load(store, tmp_path):
    store.upsert_asset_and_replace_segments(
        path=tmp_path / "one.jpg",
        kind="image",
        size=1,
        mtime_ns=1,
        segments=[SegmentInput([1.0, 0.0])],
    )
    store.upsert_asset_and_replace_segments(
        path=tmp_path / "clip.mp4",
        kind="video",
        size=1,
        mtime_ns=1,
        segments=[SegmentInput([1.0, 0.0, 0.0])],
    )

    with pytest.raises(EmbeddingValidationError, match="mixed embedding dimensions"):
        store.load_search_matrix()
    assert store.load_search_matrix("image")[1].shape == (1, 2)
    assert store.load_search_matrix("video")[1].shape == (1, 3)


def test_mark_error_updates_existing_metadata_or_creates_failed_asset(store, tmp_path):
    existing = store.upsert_asset_and_replace_segments(
        path=tmp_path / "old.jpg",
        kind="image",
        size=1,
        mtime_ns=1,
        segments=[SegmentInput([1.0, 0.0])],
    )
    failed = store.mark_asset_error(
        existing.path,
        "GPU decode failed",
        size=20,
        mtime_ns=20,
    )
    assert failed.id == existing.id
    assert failed.status == "error"
    assert failed.error == "GPU decode failed"
    assert (failed.size, failed.mtime_ns) == (20, 20)
    assert len(store.list_segments(failed.id)) == 1
    assert store.load_search_matrix()[0] == []

    new_failure = store.mark_asset_error(
        tmp_path / "broken.mp4",
        "probe failed",
        kind="video",
        size=30,
        mtime_ns=30,
        duration_ms=0,
    )
    assert new_failure.kind is MediaKind.VIDEO
    assert new_failure.status == "error"
    assert new_failure.error == "probe failed"

    with pytest.raises(KeyError, match="kind, size, and mtime_ns"):
        store.mark_asset_error(tmp_path / "unknown.jpg", "failed")


def test_settings_and_index_job_summaries_round_trip(store):
    setting = store.set_setting("gpu", {"device": "AUTO", "enabled": True})
    assert setting.value == {"device": "AUTO", "enabled": True}
    assert store.get_setting("gpu") == setting.value
    assert store.get_setting("missing", "fallback") == "fallback"

    job = store.record_index_job(
        status="complete",
        discovered_count=4,
        indexed_count=3,
        skipped_count=1,
    )
    assert store.latest_index_job() == job
