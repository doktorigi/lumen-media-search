from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# Lumen's process never emits optional Hugging Face/Transformers telemetry.
# This must be set before importing either library.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from huggingface_hub import snapshot_download
from PIL import Image

from .config import Settings


@dataclass(frozen=True, slots=True)
class EngineStatus:
    requested_device: str
    active_device: str | None
    available_devices: tuple[str, ...]
    backend: str
    detail: str
    model_id: str
    loaded: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["available_devices"] = list(self.available_devices)
        return value


class EmbeddingEngine(Protocol):
    dimensions: int

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...

    def status(self) -> EngineStatus: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class OpenVinoClipEngine:
    """Joint CLIP image/text embeddings accelerated by Intel OpenVINO."""

    dimensions = 512
    _MODEL_SHA256 = {
        "clip-vit-base-patch32.bin": (
            "a3ff68d119cd70f91edb3016f53804d6bc101b38229a45d1df2e66353c2d58a9"
        ),
        "clip-vit-base-patch32.xml": (
            "0eaf9f5eb7fd76ec9788bcadc4f22a5bb06d40edd3bf31844618f6105ab77649"
        ),
        "preprocessor_config.json": (
            "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"
        ),
        "special_tokens_map.json": (
            "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf"
        ),
        "tokenizer.json": (
            "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"
        ),
        "tokenizer_config.json": (
            "cbcdcf14cede6b5c3a6841f37961364357d82bef3079beb115e95f27252526a0"
        ),
        "vocab.json": (
            "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"
        ),
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._core = None
        self._compiled = None
        self._processor = None
        self._image_output = None
        self._text_output = None
        self._active_device: str | None = None
        self._detail = "Model is ready to load on first use."
        self._available_devices = self._probe_devices()

    @staticmethod
    def _probe_devices() -> tuple[str, ...]:
        try:
            import openvino as ov

            return tuple(ov.Core().available_devices)
        except Exception:
            return tuple()

    def status(self) -> EngineStatus:
        return EngineStatus(
            requested_device=self.settings.requested_device,
            active_device=self._active_device,
            available_devices=self._available_devices,
            backend="OpenVINO CLIP",
            detail=self._detail,
            model_id=f"{self.settings.model_id}@{self.settings.model_revision[:8]}",
            loaded=self._compiled is not None,
        )

    def _model_path(self) -> Path:
        return self.settings.models_dir / "clip-vit-base-patch32-openvino"

    @classmethod
    def _model_is_valid(cls, target: Path) -> bool:
        for filename, expected in cls._MODEL_SHA256.items():
            path = target / filename
            if not path.is_file():
                return False
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                return False
        return True

    def _download_model(self) -> Path:
        target = self._model_path()
        if self._model_is_valid(target):
            return target
        self._detail = "Downloading the pinned CLIP model (about 307 MB)…"
        snapshot_download(
            repo_id=self.settings.model_id,
            revision=self.settings.model_revision,
            local_dir=target,
            allow_patterns=list(self._MODEL_SHA256),
            force_download=True,
            token=False,
        )
        if not self._model_is_valid(target):
            raise RuntimeError(
                "The downloaded CLIP model did not match Lumen's SHA-256 manifest"
            )
        return target

    @staticmethod
    def _find_output(compiled, wanted: str):
        for output in compiled.outputs:
            names = set(output.get_names())
            if wanted in names or any(wanted in name for name in names):
                return output
        available = sorted(
            name for output in compiled.outputs for name in output.get_names()
        )
        raise RuntimeError(f"Model output {wanted!r} not found; outputs: {available}")

    def _load(self) -> None:
        if self._compiled is not None:
            return
        with self._load_lock:
            if self._compiled is not None:
                return
            self.settings.ensure_directories()
            model_path = self._download_model()

            import openvino as ov

            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            from transformers import (
                CLIPImageProcessor,
                CLIPProcessor,
                CLIPTokenizerFast,
            )

            self._detail = "Compiling CLIP for the selected accelerator…"
            core = ov.Core()
            available = tuple(core.available_devices)
            self._available_devices = available
            requested = self.settings.requested_device
            device = requested
            if requested == "GPU" and not any(
                item.startswith("GPU") for item in available
            ):
                device = "CPU"
                self._detail = (
                    "Intel GPU was not visible to OpenVINO; using CPU fallback."
                )
            elif requested not in {"AUTO", "MULTI", "HETERO"} and not any(
                item == requested or item.startswith(requested + ".")
                for item in available
            ):
                device = "CPU"
                self._detail = (
                    f"{requested} was not visible to OpenVINO; using CPU fallback."
                )

            model = core.read_model(model_path / "clip-vit-base-patch32.xml")
            compile_config = {
                "CACHE_DIR": str(self.settings.compiled_models_dir),
                "PERFORMANCE_HINT": "THROUGHPUT",
            }
            try:
                compiled = core.compile_model(model, device, compile_config)
            except Exception as exc:
                if device == "CPU":
                    raise
                device = "CPU"
                compiled = core.compile_model(model, device, compile_config)
                self._detail = f"{requested} compilation failed ({type(exc).__name__}); using CPU fallback."
            else:
                self._detail = f"CLIP is running on {device}."

            self._core = core
            self._compiled = compiled
            self._processor = CLIPProcessor(
                image_processor=CLIPImageProcessor.from_pretrained(
                    model_path, local_files_only=True
                ),
                tokenizer=CLIPTokenizerFast.from_pretrained(
                    model_path, local_files_only=True
                ),
            )
            self._image_output = self._find_output(compiled, "image_embeds")
            self._text_output = self._find_output(compiled, "text_embeds")
            self._active_device = device

    def _infer(self, *, images: Sequence[Image.Image], texts: Sequence[str]):
        self._load()
        assert self._processor is not None
        assert self._compiled is not None
        inputs = self._processor(
            text=list(texts),
            images=list(images),
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        payload = {name: np.asarray(value) for name, value in inputs.items()}
        with self._inference_lock:
            return self._compiled(payload)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimensions), dtype=np.float32)
        result = self._infer(images=images, texts=("",))
        assert self._image_output is not None
        return _normalize(np.asarray(result[self._image_output]))

    def encode_text(self, text: str) -> np.ndarray:
        query = text.strip()
        if not query:
            raise ValueError("Search text cannot be empty")
        blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
        result = self._infer(images=(blank,), texts=(query,))
        assert self._text_output is not None
        return _normalize(np.asarray(result[self._text_output]))[0]


class DeterministicTestEngine:
    """Small deterministic engine used by tests; never selected in production."""

    dimensions = 16

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            pixels = np.asarray(image.resize((2, 2)).convert("RGB"), dtype=np.float32)
            values = np.resize(pixels.reshape(-1), self.dimensions)
            rows.append(values)
        return _normalize(np.asarray(rows, dtype=np.float32))

    def encode_text(self, text: str) -> np.ndarray:
        raw = text.encode("utf-8") or b"0"
        values = np.fromiter(
            (raw[i % len(raw)] for i in range(self.dimensions)), np.float32
        )
        return _normalize(values)[0]

    def status(self) -> EngineStatus:
        return EngineStatus(
            requested_device="TEST",
            active_device="TEST",
            available_devices=("TEST",),
            backend="deterministic-test",
            detail="Deterministic test engine",
            model_id="test",
            loaded=True,
        )
