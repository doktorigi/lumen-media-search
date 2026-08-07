from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

from lumen_media.media import (
    FILE_ATTRIBUTE_OFFLINE,
    FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS,
    FILE_ATTRIBUTE_RECALL_ON_OPEN,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_cloud_placeholder,
    load_image,
    media_kind,
    sample_timestamps,
    thumbnail_name,
)


class FakeStat:
    def __init__(self, attributes: int | None = None) -> None:
        if attributes is not None:
            self.st_file_attributes = attributes


@pytest.mark.parametrize("extension", sorted(IMAGE_EXTENSIONS))
def test_media_kind_recognizes_image_extensions_case_insensitively(
    extension: str,
) -> None:
    assert media_kind(Path(f"photo{extension.upper()}")) == "image"


@pytest.mark.parametrize("extension", sorted(VIDEO_EXTENSIONS))
def test_media_kind_recognizes_video_extensions_case_insensitively(
    extension: str,
) -> None:
    assert media_kind(Path(f"clip{extension.upper()}")) == "video"


@pytest.mark.parametrize("name", ["notes.txt", "archive.zip", "no-extension", ".jpg"])
def test_media_kind_rejects_unsupported_or_missing_extensions(name: str) -> None:
    assert media_kind(Path(name)) is None


@pytest.mark.parametrize(
    "attributes",
    [
        FILE_ATTRIBUTE_OFFLINE,
        FILE_ATTRIBUTE_RECALL_ON_OPEN,
        FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS,
        FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS,
    ],
)
def test_cloud_placeholder_detects_each_placeholder_attribute(attributes: int) -> None:
    assert is_cloud_placeholder(FakeStat(attributes))


def test_cloud_placeholder_defaults_missing_attributes_to_online() -> None:
    assert not is_cloud_placeholder(FakeStat())
    assert not is_cloud_placeholder(FakeStat(0))
    assert not is_cloud_placeholder(FakeStat(0x20))


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (-1, [0]),
        (0, [0]),
        (1, [0]),
        (3_999, [1_999]),
        (4_000, [2_000]),
    ],
)
def test_sample_timestamps_uses_midpoint_for_short_videos(
    duration_ms: int, expected: list[int]
) -> None:
    assert sample_timestamps(duration_ms) == expected


@pytest.mark.parametrize(
    ("duration_ms", "max_frames", "expected_count"),
    [
        (5_000, 60, 3),
        (60_000, 60, 30),
        (90_000, 7, 7),
        (3_600_000, 60, 60),
    ],
)
def test_sample_timestamps_are_ordered_bounded_and_capped(
    duration_ms: int, max_frames: int, expected_count: int
) -> None:
    timestamps = sample_timestamps(duration_ms, max_frames=max_frames)

    assert len(timestamps) == expected_count
    assert len(timestamps) <= max_frames
    assert timestamps == sorted(set(timestamps))
    assert all(0 <= timestamp < duration_ms for timestamp in timestamps)


def test_thumbnail_name_is_deterministic_and_content_addressed() -> None:
    path = Path("library") / "holiday photo.jpg"
    name = thumbnail_name(path, mtime_ns=123_456, timestamp_ms=789)

    assert name == thumbnail_name(path, mtime_ns=123_456, timestamp_ms=789)
    assert re.fullmatch(r"[0-9a-f]{32}\.jpg", name)
    assert name != thumbnail_name(path, mtime_ns=123_457, timestamp_ms=789)
    assert name != thumbnail_name(path, mtime_ns=123_456, timestamp_ms=790)
    assert name != thumbnail_name(path.with_name("other.jpg"), 123_456, 789)


def test_load_image_converts_to_detached_rgb(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (3, 2), (12, 34, 56, 78)).save(path)

    decoded = load_image(path)
    path.unlink()

    assert decoded.image.mode == "RGB"
    assert decoded.image.size == (3, 2)
    assert (decoded.width, decoded.height) == (3, 2)
    assert decoded.image.getpixel((0, 0)) == (12, 34, 56)


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise when displayed.
    Image.new("RGB", (2, 3), "red").save(path, exif=exif)

    decoded = load_image(path)

    assert decoded.image.mode == "RGB"
    assert decoded.image.size == (3, 2)
    assert (decoded.width, decoded.height) == (3, 2)
