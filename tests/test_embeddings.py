from __future__ import annotations

import hashlib
import os

from lumen_media.embeddings import OpenVinoClipEngine


def test_private_process_disables_hugging_face_telemetry() -> None:
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DO_NOT_TRACK"] == "1"


def test_model_manifest_accepts_only_expected_bytes(tmp_path, monkeypatch) -> None:
    payload = b"pinned model artifact"
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(payload)
    monkeypatch.setattr(
        OpenVinoClipEngine,
        "_MODEL_SHA256",
        {"model.bin": hashlib.sha256(payload).hexdigest()},
    )

    assert OpenVinoClipEngine._model_is_valid(tmp_path)

    artifact.write_bytes(b"tampered")

    assert not OpenVinoClipEngine._model_is_valid(tmp_path)
