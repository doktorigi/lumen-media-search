from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings, with all mutable data kept outside the source tree."""

    data_dir: Path
    cache_dir: Path
    requested_device: str = "GPU"
    host: str = "127.0.0.1"
    port: int = 8765
    image_batch_size: int = 8
    video_batch_size: int = 8
    video_max_frames: int = 60
    thumbnail_edge: int = 640
    model_id: str = "scaleflex/clip-vit-base-patch32-openvino"
    model_revision: str = "26f5e9f1ec6185017f8a7422e63c8fc083403846"

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> Settings:
        resolved_data = data_dir or Path(
            os.environ.get("LUMEN_DATA_DIR", user_data_dir("Lumen", "Lumen"))
        )
        resolved_cache = cache_dir or Path(
            os.environ.get("LUMEN_CACHE_DIR", user_cache_dir("Lumen", "Lumen"))
        )
        return cls(
            data_dir=resolved_data.expanduser().resolve(),
            cache_dir=resolved_cache.expanduser().resolve(),
            requested_device=os.environ.get("LUMEN_DEVICE", "GPU").upper(),
            host=os.environ.get("LUMEN_HOST", "127.0.0.1"),
            port=_env_int("LUMEN_PORT", 8765),
            image_batch_size=max(1, _env_int("LUMEN_IMAGE_BATCH_SIZE", 8)),
            video_batch_size=max(1, _env_int("LUMEN_VIDEO_BATCH_SIZE", 8)),
            video_max_frames=max(1, _env_int("LUMEN_VIDEO_MAX_FRAMES", 60)),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "library.sqlite3"

    @property
    def thumbnails_dir(self) -> Path:
        return self.cache_dir / "thumbnails"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def compiled_models_dir(self) -> Path:
        return self.cache_dir / "compiled-models"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.cache_dir,
            self.thumbnails_dir,
            self.models_dir,
            self.compiled_models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
