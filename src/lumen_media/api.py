from __future__ import annotations

import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import Settings
from .embeddings import EmbeddingEngine, OpenVinoClipEngine
from .indexer import IndexJobManager, MediaIndexer
from .search import SemanticSearch
from .store import MediaStore


class IndexRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=32)


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    kind: Literal["all", "image", "video"] = "all"
    limit: int = Field(default=60, ge=1, le=200)


_SESSION_COOKIE = "lumen_session"
_BOOTSTRAP_HEADER = "X-Lumen-Bootstrap"
_PRIVATE_API_PREFIX = "/api/"
_SESSION_PATH = "/api/session"


@dataclass(slots=True)
class AppServices:
    settings: Settings
    store: MediaStore
    engine: EmbeddingEngine
    search: SemanticSearch
    jobs: IndexJobManager


def create_services(
    settings: Settings | None = None,
    *,
    store: MediaStore | None = None,
    engine: EmbeddingEngine | None = None,
) -> AppServices:
    resolved = settings or Settings.from_env()
    resolved.ensure_directories()
    media_store = store or MediaStore(resolved.database_path)
    media_store.initialize()
    embedding_engine = engine or OpenVinoClipEngine(resolved)
    search = SemanticSearch(media_store, embedding_engine)
    indexer = MediaIndexer(resolved, media_store, embedding_engine)
    return AppServices(
        settings=resolved,
        store=media_store,
        engine=embedding_engine,
        search=search,
        jobs=IndexJobManager(indexer),
    )


def create_app(
    settings: Settings | None = None,
    *,
    store: MediaStore | None = None,
    engine: EmbeddingEngine | None = None,
) -> FastAPI:
    services = create_services(settings, store=store, engine=engine)
    static_dir = Path(__file__).with_name("static")

    app = FastAPI(
        title="Lumen",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.services = services
    app.state.bootstrap_token = secrets.token_urlsafe(48)
    app.state.session_token = secrets.token_urlsafe(48)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def protect_local_session(request: Request, call_next):
        if request.headers.get("sec-fetch-site", "").casefold() == "cross-site":
            return _harden_response(
                JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-site requests cannot access Lumen."},
                )
            )

        origin = request.headers.get("origin")
        if origin:
            expected_port = services.settings.port
            try:
                parsed = urlsplit(origin)
                hostname = (parsed.hostname or "").casefold()
                actual_port = parsed.port or (
                    443 if parsed.scheme.casefold() == "https" else 80
                )
                valid_origin = (
                    parsed.scheme.casefold() == request.url.scheme.casefold()
                    and parsed.username is None
                    and parsed.password is None
                    and hostname in {"127.0.0.1", "localhost", "::1"}
                    and actual_port == expected_port
                )
            except ValueError:
                valid_origin = False
            if not valid_origin:
                return _harden_response(
                    JSONResponse(
                        status_code=403,
                        content={
                            "detail": "Foreign browser origins cannot control Lumen."
                        },
                    )
                )

        path = request.url.path
        if path.startswith(_PRIVATE_API_PREFIX) and path != _SESSION_PATH:
            supplied = request.cookies.get(_SESSION_COOKIE, "")
            if not _tokens_match(supplied, app.state.session_token):
                return _harden_response(
                    JSONResponse(
                        status_code=401,
                        content={"detail": "A Lumen session is required."},
                    )
                )

        response = await call_next(request)
        return _harden_response(response)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.post(_SESSION_PATH)
    async def create_session(request: Request) -> Response:
        supplied = request.headers.get(_BOOTSTRAP_HEADER, "")
        if not _tokens_match(supplied, app.state.bootstrap_token):
            raise HTTPException(status_code=403, detail="Invalid bootstrap token")
        response = JSONResponse({"ok": True})
        response.set_cookie(
            key=_SESSION_COOKIE,
            value=app.state.session_token,
            httponly=True,
            secure=request.url.scheme.casefold() == "https",
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        stats = services.store.library_stats()
        engine_status = services.engine.status().to_dict()
        engine_status["ranker"] = services.search.ranker.backend
        active = services.jobs.active()
        return {
            "engine": engine_status,
            "library": {
                "roots": stats.root_count,
                "assets": stats.asset_count,
                "segments": stats.segment_count,
                "photos": stats.image_count,
                "videos": stats.video_count,
                "indexed": stats.indexed_count,
                "errors": stats.error_count,
                "offline_skipped": stats.offline_skipped,
                "total_bytes": stats.total_bytes,
            },
            "job": None if active is None else active.to_dict(),
        }

    @app.get("/api/roots")
    async def roots() -> dict[str, object]:
        return {"roots": [asdict(root) for root in services.store.list_roots()]}

    @app.post("/api/index", status_code=202)
    async def start_index(payload: IndexRequest) -> dict[str, object]:
        paths: list[Path] = []
        for raw_path in payload.paths:
            try:
                path = Path(raw_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Folder does not exist: {raw_path}"
                ) from exc
            if not path.is_dir():
                raise HTTPException(status_code=400, detail=f"Not a folder: {raw_path}")
            paths.append(path)
        try:
            job = services.jobs.start(paths)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, object]:
        job = services.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Index job not found")
        return job.to_dict()

    @app.post("/api/search")
    async def search(payload: SearchRequest) -> dict[str, object]:
        try:
            result = await run_in_threadpool(
                services.search.search,
                payload.q,
                kind=payload.kind,
                limit=payload.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/api/media/{asset_id}")
    async def media(asset_id: int):
        asset = services.store.get_asset(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Media not found")
        if asset.offline:
            raise HTTPException(
                status_code=409,
                detail="This OneDrive item is cloud-only and was not downloaded.",
            )
        path = Path(asset.path)
        if not path.is_file():
            raise HTTPException(
                status_code=410, detail="Media file is no longer present"
            )
        content_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            media_type=content_type or "application/octet-stream",
            filename=None,
        )

    @app.get("/api/thumbnail/{segment_id}")
    async def thumbnail(segment_id: int):
        segment = services.store.get_segment(segment_id)
        if segment is None or segment.thumbnail_path is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        path = Path(segment.thumbnail_path).resolve(strict=False)
        root = services.settings.thumbnails_dir.resolve(strict=False)
        if not path.is_relative_to(root):
            raise HTTPException(status_code=403, detail="Invalid thumbnail location")
        if not path.is_file():
            raise HTTPException(
                status_code=410, detail="Thumbnail is no longer present"
            )
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/open/{asset_id}")
    async def open_asset(asset_id: int) -> dict[str, bool]:
        asset_path = _asset_path_or_404(services, asset_id)
        try:
            _open_path(asset_path)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not open file: {exc}"
            ) from exc
        return {"ok": True}

    @app.post("/api/reveal/{asset_id}")
    async def reveal_asset(asset_id: int) -> dict[str, bool]:
        asset_path = _asset_path_or_404(services, asset_id)
        try:
            _reveal_path(asset_path)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not reveal file: {exc}"
            ) from exc
        return {"ok": True}

    return app


def _harden_response(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def make_bootstrap_url(app: FastAPI, host: str, port: int) -> str:
    """Build a browser URL whose secret fragment is never sent in an HTTP request."""

    browser_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    token = quote(str(app.state.bootstrap_token), safe="")
    return f"http://{browser_host}:{port}/#token={token}"


def _tokens_match(supplied: str, expected: str) -> bool:
    return secrets.compare_digest(
        supplied.encode("utf-8", errors="surrogatepass"),
        expected.encode("utf-8", errors="surrogatepass"),
    )


def _asset_path_or_404(services: AppServices, asset_id: int) -> Path:
    asset = services.store.get_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Media not found")
    if asset.offline:
        raise HTTPException(status_code=409, detail="Media is currently cloud-only")
    path = Path(asset.path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="Media file is no longer present")
    return path


def _open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _reveal_path(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(["explorer.exe", f"/select,{path}"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    elif shutil.which("explorer.exe") and shutil.which("wslpath"):
        converted = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.Popen(["explorer.exe", f"/select,{converted}"])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])
