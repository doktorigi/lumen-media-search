# Contributing to Lumen

Thanks for helping make local media search more useful and more trustworthy.
Lumen is experimental and Windows-first, so focused bug reports and small,
well-tested pull requests are especially valuable.

## Set up a development environment

Install Python 3.11–3.14 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```powershell
git clone https://github.com/doktorigi/lumen-media-search.git
cd lumen-media-search
uv sync --frozen --all-groups
uv run pytest
```

The ordinary test suite uses a deterministic 16-dimensional embedding engine.
It does not download the CLIP model or require a GPU.

Before opening a pull request, run:

```powershell
uv run pytest
uvx ruff check .
uvx ruff format --check .
node --check src/lumen_media/static/app.js
uv build
```

## Project map

- `embeddings.py` owns the OpenVINO CLIP adapter, device fallback, model pin,
  and artifact verification.
- `media.py` discovers and decodes local media, recognizes OneDrive
  placeholders, samples videos, and writes thumbnails.
- `store.py` and `domain.py` own the SQLite schema and persistence boundary.
- `indexer.py` batches embeddings and performs change-aware indexing.
- `search.py` and `ranker.py` perform exact semantic retrieval and group video
  frames by asset.
- `mojo_kernels/mojo_ranker.mojo` is the optional Mojo implementation of exact
  top-k dot products.
- `api.py`, `cli.py`, and `static/` provide the authenticated loopback app.

## Design rules

Changes should preserve these boundaries:

1. Original media is never moved, renamed, modified, or uploaded.
2. Network access is limited to explicit dependency setup and the pinned
   first-run model download.
3. Private API, media, and thumbnail routes require the per-launch browser
   session; loopback binding alone is not treated as authentication.
4. OneDrive Files On-Demand placeholders must not be hydrated implicitly.
5. Embeddings are validated, finite, contiguous, normalized float32 vectors at
   the storage boundary.
6. The optional Mojo path must retain a tested NumPy fallback.
7. Search scores must be described as similarity, never confidence or
   probability.

If a change introduces a new external service, telemetry path, model, media
mutation, persistent user-data field, or executable download, call it out
prominently in the pull request.

## Testing hardware paths

Run a real OpenVINO smoke test manually when changing model or device code:

```powershell
uv run lumen doctor --load-model
```

The output must report the actual compiled device. A requested GPU is not proof
that inference ran on one.

The optional Mojo kernel can be checked in WSL/Linux with:

```powershell
wsl bash /mnt/c/path/to/lumen-media-search/scripts/validate_mojo.sh
```

That helper installs software governed by Modular's separate Community License;
it is not part of the normal Windows runtime.

## Pull requests

- Keep each pull request centered on one change.
- Add or update tests for behavior changes and failure paths.
- Update the README when user-visible commands, privacy behavior, model
  provenance, storage, or supported media change.
- Do not commit media libraries, model weights, SQLite databases, thumbnails,
  caches, credentials, or machine-specific paths.
- Include the checks you ran and any hardware-specific validation in the pull
  request description.

For suspected vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.
