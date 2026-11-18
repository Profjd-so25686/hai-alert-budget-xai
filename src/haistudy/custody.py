"""Chain of custody: SHA-256 manifests for dataset files.

Used twice in the study. At download (week 6) every HAI file is hashed and the
manifest is committed. Immediately before the held-out files are opened (week 20)
they are verified against that manifest, which shows they are the files that were
sealed and that they have not changed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of one file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(folder: Path, pattern: str = "*") -> dict:
    """Hash every file under folder matching pattern. Paths are stored relative to folder."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob(pattern) if p.is_file() and p.name != ".gitkeep")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folder": folder.name,
        "algorithm": "sha256",
        "files": {p.relative_to(folder).as_posix(): {"sha256": sha256_of(p), "bytes": p.stat().st_size} for p in files},
    }


def write_manifest(folder: Path, out: Path, pattern: str = "*") -> dict:
    manifest = build_manifest(folder, pattern)
    Path(out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(folder: Path, manifest_path: Path, only: list[str] | None = None) -> dict:
    """Compare files on disk with a manifest.

    only: optional list of substrings; when given, only manifest entries whose path
    contains one of them are checked (for example ["test3", "test4", "test5"]).
    Returns {"ok": [...], "changed": [...], "missing": [...]}.
    """
    folder = Path(folder)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = {"ok": [], "changed": [], "missing": []}
    for rel, meta in manifest["files"].items():
        if only and not any(s in rel for s in only):
            continue
        p = folder / rel
        if not p.is_file():
            result["missing"].append(rel)
        elif sha256_of(p) == meta["sha256"]:
            result["ok"].append(rel)
        else:
            result["changed"].append(rel)
    return result
