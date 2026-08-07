from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path

import uvicorn
from PIL import Image

from . import __version__
from .api import create_app, create_services, make_bootstrap_url
from .config import Settings
from .indexer import IndexJob
from .store import MediaStore

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _port(value: str | int, *, label: str = "port") -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65_535:
        raise ValueError(f"{label} must be between 1 and 65535")
    return port


def _port_argument(value: str) -> int:
    try:
        return _port(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumen", description="Local semantic photo and video search"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Run the local web app")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=_port_argument, default=None)
    serve.add_argument("--open", action="store_true", dest="open_browser")

    doctor = subcommands.add_parser("doctor", help="Inspect accelerator support")
    doctor.add_argument("--load-model", action="store_true")

    index = subcommands.add_parser("index", help="Index folders synchronously")
    index.add_argument("paths", nargs="+", type=Path)

    clear = subcommands.add_parser(
        "clear", help="Delete Lumen's index, thumbnails, and downloaded model"
    )
    clear.add_argument(
        "--yes",
        action="store_true",
        help="delete without an interactive confirmation prompt",
    )
    return parser


def _serve_port(argument: int | None, settings: Settings) -> int:
    if argument is not None:
        return argument
    raw_environment_port = os.environ.get("LUMEN_PORT")
    if raw_environment_port is not None:
        return _port(raw_environment_port, label="LUMEN_PORT")
    return _port(settings.port)


def _resolve_index_paths(raw_paths: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        try:
            path = raw_path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"path does not exist: {raw_path}") from exc
        if not path.is_dir():
            raise ValueError(f"not a directory: {raw_path}")
        paths.append(path)
    return paths


def _dangerous_clear_paths() -> set[Path]:
    values = [Path.home(), Path.home().parent, Path.cwd()]
    for name in (
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PUBLIC",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    ):
        raw_value = os.environ.get(name)
        if raw_value:
            values.append(Path(raw_value))
    return {value.expanduser().resolve(strict=False) for value in values}


def _validate_clear_target(path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    anchor = Path(target.anchor).resolve(strict=False)
    if target == anchor or target == target.parent:
        raise ValueError(f"refusing to clear filesystem root: {target}")
    for dangerous in _dangerous_clear_paths():
        if dangerous == target or dangerous.is_relative_to(target):
            raise ValueError(f"refusing broad or system path: {target}")
    if not any("lumen" in part.casefold() for part in target.parts):
        raise ValueError(f"refusing non-Lumen path: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"refusing non-directory path: {target}")
    return target


def _clear_targets(settings: Settings) -> tuple[Path, ...]:
    validated = {
        _validate_clear_target(settings.data_dir),
        _validate_clear_target(settings.cache_dir),
    }
    # If one configured directory contains the other, deleting the parent once
    # is sufficient and avoids traversing a stale child path afterward.
    targets: list[Path] = []
    for candidate in sorted(validated, key=lambda value: len(value.parts)):
        if not any(candidate.is_relative_to(parent) for parent in targets):
            targets.append(candidate)
    return tuple(targets)


def _assert_media_outside_clear_targets(
    settings: Settings, targets: tuple[Path, ...]
) -> None:
    if not settings.database_path.is_file():
        return
    try:
        roots = MediaStore(settings.database_path).list_roots()
    except Exception as exc:
        raise RuntimeError(
            "cannot verify indexed media roots; clear the corrupt catalog manually"
        ) from exc
    for root in roots:
        media_root = Path(root.path).resolve(strict=False)
        for target in targets:
            if media_root == target or media_root.is_relative_to(target):
                raise ValueError(
                    f"refusing to clear {target}; it contains indexed media {media_root}"
                )


def _confirm_clear(targets: tuple[Path, ...]) -> bool:
    listed = "\n".join(f"  - {target}" for target in targets)
    prompt = (
        "Delete Lumen's index, thumbnails, and downloaded model from:\n"
        f"{listed}\n"
        "Original photos and videos outside these directories are not deleted. "
        "Continue? [y/N] "
    )
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().casefold() in {"y", "yes"}


def _clear(settings: Settings, *, assume_yes: bool) -> int:
    targets = _clear_targets(settings)
    _assert_media_outside_clear_targets(settings, targets)
    if not assume_yes and not _confirm_clear(targets):
        print("Clear cancelled; nothing was deleted.")
        return 0
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Deleted {target}")
        else:
            print(f"Already absent: {target}")
    return 0


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "clear":
        return _clear(settings, assume_yes=args.yes)

    if args.command == "serve":
        host = args.host if args.host is not None else settings.host
        port = _serve_port(args.port, settings)
        if host not in _LOOPBACK_HOSTS:
            raise ValueError("Lumen only binds to a loopback address for privacy")
        if port != settings.port:
            settings = replace(settings, port=port)
        app = create_app(settings)
        bootstrap_url = make_bootstrap_url(app, host, port)
        print(f"Open Lumen in your browser: {bootstrap_url}", flush=True)
        if args.open_browser:
            timer = threading.Timer(1.0, webbrowser.open, args=(bootstrap_url,))
            timer.daemon = True
            timer.start()
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
        )
        return 0

    if args.command == "index":
        paths = _resolve_index_paths(args.paths)
        services = create_services(settings)
        job = IndexJob(id="cli", paths=tuple(str(path) for path in paths))
        services.jobs.indexer.run(paths, job)
        print(json.dumps(job.to_dict(), indent=2))
        return 0 if job.state == "completed" else 1

    services = create_services(settings)
    if args.command == "doctor":
        before = services.engine.status().to_dict()
        if args.load_model:
            image = Image.new("RGB", (224, 224), color=(65, 105, 225))
            image_vector = services.engine.encode_images((image,))[0]
            text_vector = services.engine.encode_text("a solid blue image")
            before["smoke_similarity"] = round(float(image_vector @ text_vector), 6)
            before = services.engine.status().to_dict() | {
                "smoke_similarity": before["smoke_similarity"]
            }
        print(json.dumps(before, indent=2))
        return 0

    raise RuntimeError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, Settings.from_env())
    except KeyboardInterrupt:
        print("lumen: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        print(f"lumen: error: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
