#!/usr/bin/env bash
set -euo pipefail

# Mojo is not native on Windows. This helper validates Lumen's optional ranker
# inside WSL without modifying PATH or shell startup files.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validation_root="${LUMEN_MOJO_VALIDATION_ROOT:-$HOME/.cache/lumen-mojo-validation}"
venv="$validation_root/venv"

uv_bin="${LUMEN_UV_BIN:-}"
if [[ -z "$uv_bin" ]]; then
    uv_bin="$(command -v uv || true)"
fi
if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
    echo "uv is required to validate the Mojo kernel." >&2
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/" >&2
    echo "or set LUMEN_UV_BIN to a trusted uv executable." >&2
    exit 2
fi
mkdir -p "$validation_root"

if [[ ! -x "$venv/bin/python" ]]; then
    "$uv_bin" venv "$venv" --python 3.12
fi
"$uv_bin" pip install \
    --python "$venv/bin/python" \
    "mojo==1.0.0b2" \
    "numpy>=2.1,<3" \
    --prerelease allow

kernel_dir="$project_root/src/lumen_media/mojo_kernels"
export LUMEN_MOJO_KERNEL_PATH="$kernel_dir/mojo_ranker.mojo"
export LUMEN_MOJO_DOC_PATH="$validation_root/mojo-ranker-doc.json"
"$venv/bin/python" -c '
import os
from mojo.run import subprocess_run_mojo

subprocess_run_mojo(
    ["doc", os.environ["LUMEN_MOJO_KERNEL_PATH"], "-o", os.environ["LUMEN_MOJO_DOC_PATH"]],
    check=True,
)
'
echo "Mojo compiler validation passed."

if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    # A fresh WSL image may not have build-essential and may not permit
    # passwordless sudo. Zig is a pinned, checksummed, user-local C linker for
    # this validation environment only.
    zig_version="0.16.0"
    zig_archive="$validation_root/zig-$zig_version.tar.xz"
    zig_dir="$validation_root/zig-x86_64-linux-$zig_version"
    zig_checksum="70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00"
    if [[ ! -x "$zig_dir/zig" ]]; then
        curl -L --fail --silent --show-error \
            "https://ziglang.org/download/$zig_version/zig-x86_64-linux-$zig_version.tar.xz" \
            -o "$zig_archive"
        echo "$zig_checksum  $zig_archive" | sha256sum --check --status
        tar -xJf "$zig_archive" -C "$validation_root"
    fi
    export LUMEN_ZIG_BIN="$zig_dir/zig"
    export PATH="$project_root/scripts/portable-toolchain:$PATH"
fi

if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; then
    (
        cd "$kernel_dir"
        "$venv/bin/python" -c '
import numpy as np
import mojo.importer
import mojo_ranker

matrix = np.ascontiguousarray([[1, 0], [0, 1], [0.8, 0.2]], dtype=np.float32)
query = np.ascontiguousarray([1, 0], dtype=np.float32)
indices, scores = mojo_ranker.top_k(matrix, query, 2)
assert indices == [0, 2], (indices, scores)
print("Mojo runtime smoke test passed:", indices, scores)
'
    )
else
    echo "No C compiler is available; shared-library runtime smoke test skipped."
fi
