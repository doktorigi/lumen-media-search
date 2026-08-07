from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from lumen_media.api import create_app, make_bootstrap_url
from lumen_media.config import Settings
from lumen_media.domain import AssetInput, SegmentInput
from lumen_media.embeddings import DeterministicTestEngine
from lumen_media.store import MediaStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        requested_device="GPU",
    )


@pytest.fixture
def store(settings: Settings) -> MediaStore:
    result = MediaStore(settings.database_path)
    result.initialize()
    return result


@pytest.fixture
def engine() -> DeterministicTestEngine:
    return DeterministicTestEngine()


@pytest.fixture
def app(
    settings: Settings,
    store: MediaStore,
    engine: DeterministicTestEngine,
) -> FastAPI:
    return create_app(settings, store=store, engine=engine)


@pytest.fixture
def anonymous_client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as result:
        yield result


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as result:
        response = result.post(
            "/api/session",
            headers={"X-Lumen-Bootstrap": app.state.bootstrap_token},
        )
        assert response.status_code == 200
        yield result


def assert_private_response_headers(response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_root_page_and_static_asset_are_served(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert home.headers["content-type"].startswith("text/html")
    assert_private_response_headers(home)
    assert "<title>Lumen" in home.text
    assert 'href="/static/app.css"' in home.text

    stylesheet = client.get("/static/app.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert_private_response_headers(stylesheet)
    assert ":root" in stylesheet.text


def test_bootstrap_exchange_sets_a_strict_http_only_session_cookie(
    app: FastAPI,
    anonymous_client: TestClient,
) -> None:
    before = anonymous_client.get("/api/status")
    assert before.status_code == 401
    assert before.json() == {"detail": "A Lumen session is required."}
    assert_private_response_headers(before)

    response = anonymous_client.post(
        "/api/session",
        headers={"X-Lumen-Bootstrap": app.state.bootstrap_token},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("lumen_session=")
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/" in cookie
    assert "Max-Age" not in cookie
    assert_private_response_headers(response)
    assert anonymous_client.get("/api/status").status_code == 200


def test_invalid_bootstrap_token_does_not_create_a_session(
    app: FastAPI,
    anonymous_client: TestClient,
) -> None:
    response = anonymous_client.post(
        "/api/session",
        headers={"X-Lumen-Bootstrap": "x" * 64},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid bootstrap token"}
    assert "lumen_session=" not in response.headers.get("set-cookie", "")

    legacy_json = anonymous_client.post(
        "/api/session",
        json={"bootstrap_token": app.state.bootstrap_token},
    )
    assert legacy_json.status_code == 403
    assert "lumen_session=" not in legacy_json.headers.get("set-cookie", "")
    assert anonymous_client.get("/api/status").status_code == 401


def test_bootstrap_url_keeps_the_token_in_an_encoded_fragment(app: FastAPI) -> None:
    app.state.bootstrap_token = "A" * 32 + "/+"

    url = make_bootstrap_url(app, "::1", 9123)

    assert url == f"http://[::1]:9123/#token={'A' * 32}%2F%2B"
    assert "?" not in url


@pytest.mark.parametrize(
    ("method", "route", "payload"),
    [
        ("GET", "/api/status", None),
        ("GET", "/api/roots", None),
        ("POST", "/api/index", {"paths": ["C:\\"]}),
        ("GET", "/api/jobs/missing", None),
        ("POST", "/api/search", {"q": "private search"}),
        ("GET", "/api/media/1", None),
        ("GET", "/api/thumbnail/1", None),
        ("POST", "/api/open/1", None),
        ("POST", "/api/reveal/1", None),
    ],
)
def test_every_private_route_requires_a_session(
    anonymous_client: TestClient,
    method: str,
    route: str,
    payload: dict[str, object] | None,
) -> None:
    response = anonymous_client.request(method, route, json=payload)

    assert response.status_code == 401
    assert response.json() == {"detail": "A Lumen session is required."}
    assert_private_response_headers(response)


def test_cross_site_fetch_metadata_is_rejected_before_session_exchange(
    app: FastAPI,
    anonymous_client: TestClient,
) -> None:
    response = anonymous_client.post(
        "/api/session",
        headers={
            "X-Lumen-Bootstrap": app.state.bootstrap_token,
            "Sec-Fetch-Site": "cross-site",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Cross-site requests cannot access Lumen."}
    assert "lumen_session=" not in response.headers.get("set-cookie", "")
    assert_private_response_headers(response)


def test_authenticated_cross_site_request_is_still_rejected(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/status",
        headers={"Sec-Fetch-Site": "cross-site"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Cross-site requests cannot access Lumen."}


def test_status_reports_engine_library_and_job_shape(client: TestClient) -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    assert_private_response_headers(response)
    payload = response.json()
    assert set(payload) == {"engine", "library", "job"}
    assert payload["engine"] == {
        "requested_device": "TEST",
        "active_device": "TEST",
        "available_devices": ["TEST"],
        "backend": "deterministic-test",
        "detail": "Deterministic test engine",
        "model_id": "test",
        "loaded": True,
        "ranker": "NumPy BLAS",
    }
    assert payload["library"] == {
        "roots": 0,
        "assets": 0,
        "segments": 0,
        "photos": 0,
        "videos": 0,
        "indexed": 0,
        "errors": 0,
        "offline_skipped": 0,
        "total_bytes": 0,
    }
    assert payload["job"] is None


def test_foreign_browser_origin_is_rejected(client: TestClient) -> None:
    response = client.get(
        "/api/status",
        headers={"Origin": "https://photos.example"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Foreign browser origins cannot control Lumen."
    }


def test_index_rejects_a_path_that_does_not_exist(
    client: TestClient,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-library"

    response = client.post("/api/index", json={"paths": [str(missing)]})

    assert response.status_code == 400
    assert response.json() == {"detail": f"Folder does not exist: {missing}"}


def test_searching_an_empty_library_returns_an_empty_response(
    client: TestClient,
) -> None:
    response = client.post("/api/search", json={"q": "a dog on the beach"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "a dog on the beach"
    assert payload["ranker"] == "NumPy BLAS"
    assert isinstance(payload["elapsed_ms"], int)
    assert payload["elapsed_ms"] >= 0
    assert payload["results"] == []

    legacy_get = client.get(
        "/api/search",
        params={"q": "this must not appear in an access-log URL"},
    )
    assert legacy_get.status_code == 405


def test_indexed_image_is_searchable_and_its_files_are_served(
    client: TestClient,
    settings: Settings,
    store: MediaStore,
    engine: DeterministicTestEngine,
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    media_path = library / "blue-square.jpg"
    image = Image.new("RGB", (12, 8), color=(25, 80, 210))
    image.save(media_path, format="JPEG")

    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = settings.thumbnails_dir / "blue-square-thumb.jpg"
    image.resize((6, 4)).save(thumbnail_path, format="JPEG")

    root = store.upsert_root(library)
    embedding = engine.encode_images([image])[0]
    asset = store.upsert_asset_and_replace_segments(
        AssetInput(
            path=media_path,
            kind="image",
            size=media_path.stat().st_size,
            mtime_ns=media_path.stat().st_mtime_ns,
            root_id=root.id,
            width=12,
            height=8,
        ),
        [SegmentInput(embedding=embedding, thumbnail_path=thumbnail_path)],
    )
    segment = store.list_segments(asset.id)[0]

    search = client.post("/api/search", json={"q": "a blue square"})

    assert search.status_code == 200
    payload = search.json()
    assert payload["query"] == "a blue square"
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["asset_id"] == asset.id
    assert result["segment_id"] == segment.id
    assert result["kind"] == "image"
    assert result["name"] == media_path.name
    assert result["path"] == asset.path
    assert isinstance(result["score"], float)
    assert result["timestamp_ms"] == 0
    assert result["width"] == 12
    assert result["height"] == 8
    assert result["duration_ms"] is None
    assert result["thumbnail_url"] == f"/api/thumbnail/{segment.id}"
    assert result["media_url"] == f"/api/media/{asset.id}"

    media = client.get(result["media_url"])
    assert media.status_code == 200
    assert media.headers["content-type"] == "image/jpeg"
    assert_private_response_headers(media)
    assert media.content == media_path.read_bytes()

    thumbnail = client.get(result["thumbnail_url"])
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/jpeg"
    assert_private_response_headers(thumbnail)
    assert thumbnail.content == thumbnail_path.read_bytes()


@pytest.mark.parametrize(
    ("route", "detail"),
    [
        ("/api/jobs/missing-job", "Index job not found"),
        ("/api/media/999999", "Media not found"),
        ("/api/thumbnail/999999", "Thumbnail not found"),
    ],
)
def test_missing_ids_return_not_found(
    client: TestClient,
    route: str,
    detail: str,
) -> None:
    response = client.get(route)

    assert response.status_code == 404
    assert response.json() == {"detail": detail}
