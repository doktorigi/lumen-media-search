from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
PLACEHOLDER_ATTRIBUTE_MASK = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


@dataclass(frozen=True, slots=True)
class MediaCandidate:
    path: Path
    kind: str
    size: int
    mtime_ns: int
    offline: bool


@dataclass(frozen=True, slots=True)
class DecodedImage:
    image: Image.Image
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: tuple[tuple[int, Image.Image], ...]
    width: int
    height: int
    duration_ms: int


def media_kind(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def is_cloud_placeholder(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return bool(attributes & PLACEHOLDER_ATTRIBUTE_MASK)


def discover_media(roots: Iterable[Path]) -> Iterator[MediaCandidate]:
    """Yield supported media without opening file contents or following symlinks."""

    seen: set[str] = set()
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=False)
        root_key = os.path.normcase(str(resolved_root))
        if root_key in seen:
            continue
        seen.add(root_key)
        if not resolved_root.is_dir():
            continue

        for current, directories, filenames in os.walk(
            resolved_root, followlinks=False
        ):
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".") and not Path(current, name).is_symlink()
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = Path(current, filename)
                kind = media_kind(path)
                if kind is None:
                    continue
                try:
                    stat_result = path.stat(follow_symlinks=False)
                except (FileNotFoundError, OSError):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                yield MediaCandidate(
                    path=path.resolve(strict=False),
                    kind=kind,
                    size=stat_result.st_size,
                    mtime_ns=stat_result.st_mtime_ns,
                    offline=is_cloud_placeholder(stat_result),
                )


def load_image(path: Path) -> DecodedImage:
    with Image.open(path) as source:
        frame = ImageOps.exif_transpose(source)
        frame.seek(0)
        rgb = frame.convert("RGB")
        rgb.load()
    return DecodedImage(rgb, rgb.width, rgb.height)


def sample_timestamps(duration_ms: int, max_frames: int = 60) -> list[int]:
    """Choose a bounded, uniform temporal skeleton for semantic video search."""

    if duration_ms <= 0:
        return [0]
    duration_seconds = duration_ms / 1000.0
    if duration_seconds <= 4:
        return [duration_ms // 2]

    interval_seconds = min(30.0, max(2.0, duration_seconds / max_frames))
    count = min(max_frames, max(1, math.ceil(duration_seconds / interval_seconds)))
    step = duration_ms / count
    return [
        min(duration_ms - 1, max(0, round((index + 0.5) * step)))
        for index in range(count)
    ]


def load_video_frames(path: Path, max_frames: int = 60) -> DecodedVideo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("the video decoder could not open this file")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int(round(frame_count / fps * 1000)) if fps > 0 else 0

        frames: list[tuple[int, Image.Image]] = []
        for timestamp_ms in sample_timestamps(duration_ms, max_frames):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((timestamp_ms, Image.fromarray(rgb, mode="RGB")))

        if not frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((0, Image.fromarray(rgb, mode="RGB")))
        if not frames:
            raise ValueError("the video did not contain any decodable frames")
        if width <= 0 or height <= 0:
            width, height = frames[0][1].size
        return DecodedVideo(tuple(frames), width, height, duration_ms)
    finally:
        capture.release()


def thumbnail_name(path: Path, mtime_ns: int, timestamp_ms: int) -> str:
    payload = f"{os.path.normcase(str(path))}\0{mtime_ns}\0{timestamp_ms}".encode(
        "utf-8", errors="surrogatepass"
    )
    return hashlib.sha256(payload).hexdigest()[:32] + ".jpg"


def write_thumbnail(image: Image.Image, destination: Path, edge: int = 640) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = image.copy()
    thumbnail.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    if thumbnail.mode != "RGB":
        thumbnail = thumbnail.convert("RGB")
    temporary = destination.with_suffix(".tmp")
    thumbnail.save(temporary, format="JPEG", quality=86, optimize=True)
    temporary.replace(destination)
    return destination
