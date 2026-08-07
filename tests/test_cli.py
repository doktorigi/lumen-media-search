from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import lumen_media.cli as cli
from lumen_media.config import Settings


def _lumen_environment(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "lumen-data"
    cache_dir = tmp_path / "lumen-cache"
    monkeypatch.setenv("LUMEN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LUMEN_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("LUMEN_PORT", raising=False)
    return data_dir, cache_dir


def test_version_is_available_without_a_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--version"])

    assert raised.value.code == 0
    assert capsys.readouterr().out == "lumen 0.1.0\n"


@pytest.mark.parametrize("port", ["0", "-1", "65536", "not-a-number"])
def test_serve_rejects_invalid_command_line_ports(port: str, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["serve", "--port", port])

    assert raised.value.code == 2
    assert "port must be" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["0", "65536", "not-a-number"])
def test_serve_rejects_invalid_environment_ports(
    port: str, monkeypatch, tmp_path: Path, capsys
) -> None:
    _lumen_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("LUMEN_PORT", port)
    create_app = Mock()
    monkeypatch.setattr(cli, "create_app", create_app)

    assert cli.main(["serve"]) == 1
    assert "LUMEN_PORT must be" in capsys.readouterr().err
    create_app.assert_not_called()


def test_serve_prints_and_opens_fragment_token_without_access_logs(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(
        cli.Settings,
        "from_env",
        classmethod(lambda cls: settings),
    )
    bootstrap_token = "A" * 32 + "/+"
    fake_app = SimpleNamespace(state=SimpleNamespace(bootstrap_token=bootstrap_token))
    monkeypatch.setattr(cli, "create_app", Mock(return_value=fake_app))
    run = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)
    opened_urls: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", opened_urls.append)

    class ImmediateTimer:
        daemon = False

        def __init__(self, interval, function, args=()):
            self.function = function
            self.args = args

        def start(self) -> None:
            self.function(*self.args)

    monkeypatch.setattr(cli.threading, "Timer", ImmediateTimer)

    result = cli.main(["serve", "--host", "::1", "--port", "9123", "--open"])

    expected_url = f"http://[::1]:9123/#token={'A' * 32}%2F%2B"
    assert result == 0
    assert opened_urls == [expected_url]
    assert expected_url in capsys.readouterr().out
    run.assert_called_once_with(
        fake_app,
        host="::1",
        port=9123,
        log_level="info",
        access_log=False,
    )


def test_missing_index_path_is_a_concise_error_without_creating_services(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _lumen_environment(monkeypatch, tmp_path)
    create_services = Mock()
    monkeypatch.setattr(cli, "create_services", create_services)

    result = cli.main(["index", str(tmp_path / "missing")])

    assert result == 1
    assert "lumen: error: path does not exist:" in capsys.readouterr().err
    create_services.assert_not_called()


def test_operational_failures_do_not_emit_a_traceback(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    _lumen_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "create_services",
        Mock(side_effect=RuntimeError("accelerator unavailable")),
    )

    assert cli.main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "lumen: error: accelerator unavailable\n"
    assert "Traceback" not in captured.err


def test_clear_with_yes_removes_only_lumen_state(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir, cache_dir = _lumen_environment(monkeypatch, tmp_path)
    data_dir.mkdir()
    cache_dir.mkdir()
    (data_dir / "settings.json").write_text("{}", encoding="utf-8")
    (cache_dir / "thumbnail.jpg").write_bytes(b"thumbnail")
    media_dir = tmp_path / "original-media"
    media_dir.mkdir()
    original = media_dir / "family-photo.jpg"
    original.write_bytes(b"original")

    assert cli.main(["clear", "--yes"]) == 0

    assert not data_dir.exists()
    assert not cache_dir.exists()
    assert original.read_bytes() == b"original"
    assert capsys.readouterr().out.count("Deleted ") == 2


def test_clear_requires_confirmation_and_decline_preserves_state(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir, cache_dir = _lumen_environment(monkeypatch, tmp_path)
    data_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setattr("builtins.input", Mock(return_value="no"))

    assert cli.main(["clear"]) == 0

    assert data_dir.is_dir()
    assert cache_dir.is_dir()
    assert "nothing was deleted" in capsys.readouterr().out


def test_clear_accepts_interactive_confirmation(monkeypatch, tmp_path: Path) -> None:
    data_dir, cache_dir = _lumen_environment(monkeypatch, tmp_path)
    data_dir.mkdir()
    cache_dir.mkdir()
    monkeypatch.setattr("builtins.input", Mock(return_value="yes"))

    assert cli.main(["clear"]) == 0
    assert not data_dir.exists()
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    "dangerous",
    [Path(Path.cwd().anchor), Path.home(), Path.cwd()],
)
def test_clear_validator_rejects_broad_or_user_paths(dangerous: Path) -> None:
    with pytest.raises(ValueError, match="refusing"):
        cli._validate_clear_target(dangerous)


def test_clear_refuses_a_configured_directory_containing_indexed_media(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    data_dir, cache_dir = _lumen_environment(monkeypatch, tmp_path)
    settings = Settings.from_env()
    settings.ensure_directories()
    media_inside_cache = cache_dir / "originals"
    media_inside_cache.mkdir()
    original = media_inside_cache / "photo.jpg"
    original.write_bytes(b"original")
    store = cli.MediaStore(settings.database_path)
    store.initialize()
    store.upsert_root(media_inside_cache)

    assert cli.main(["clear", "--yes"]) == 1

    assert original.read_bytes() == b"original"
    assert data_dir.is_dir()
    assert cache_dir.is_dir()
    assert "contains indexed media" in capsys.readouterr().err
