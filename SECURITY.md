# Security policy

## Supported versions

Lumen is pre-1.0. Security fixes are applied to the latest commit on `main`.
Older snapshots are not maintained as separate release lines.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability or privacy bypass. Use
GitHub's private vulnerability reporting for this repository:

<https://github.com/doktorigi/lumen-media-search/security/advisories/new>

Include the affected revision, operating system, reproduction steps, expected
impact, and any suggested mitigation. Avoid attaching personal media or a real
Lumen database; use synthetic files and redacted paths.

## Security model

Lumen is a single-user, local desktop-style application served through a web
browser. Its main controls are:

- The server accepts loopback addresses only and validates the HTTP `Host`.
- A high-entropy secret generated for each server process bootstraps an
  HttpOnly, same-site browser session.
- Private API, original-media, thumbnail, indexing, and operating-system action
  routes require that session.
- Foreign browser origins and cross-site fetch metadata are rejected.
- Private responses are marked `no-store` and same-origin.
- User-controlled paths are resolved by the server; media-serving routes accept
  database IDs rather than arbitrary filesystem paths.
- Original media is never modified or deleted by Lumen.

These controls reduce drive-by browser access and unauthenticated local HTTP
calls. They do not protect against an administrator, malware running as the same
user, a compromised browser or extension, process debugging, direct reads of the
SQLite database/cache, or operating-system account compromise.

## Sensitive local data

The database stores full filesystem paths, metadata, and visual embeddings. The
cache stores recognizable JPEG thumbnails and downloaded model files. Keep the
operating-system account and its application-data directories private.

Use `lumen clear` to remove Lumen's index and generated caches. This never
deletes original photos or videos. Normal filesystem deletion is not guaranteed
to be forensic secure erasure on every drive or cloud-synchronized filesystem.

## Network behavior

Lumen's only intentional runtime network request is the first download of the
pinned public model from Hugging Face. The revision and allowed files are fixed,
and each artifact is checked against a SHA-256 manifest before inference.
Hugging Face and Transformers telemetry are disabled in the process. Lumen does
not invoke OpenVINO model-conversion or telemetry tools.

Installing Python dependencies, running the optional Mojo validation helper,
and GitHub automation have their own explicit network behavior and third-party
terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
