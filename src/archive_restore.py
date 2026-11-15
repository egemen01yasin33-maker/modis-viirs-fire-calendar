"""
Restore full FIRMS archive CSVs from Google Drive using data/archive_manifest.json.

Keeps every listed file — no Turkey-only trim. Existing local files that look
complete are skipped so you never re-download unnecessarily.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.archive_csv import ArchiveCsvError, data_dir

MANIFEST_NAME = "archive_manifest.json"
# Accept local files within 5% of declared size (or at least 50 MiB).
MIN_OK_BYTES = 50 * 1024 * 1024
SIZE_TOLERANCE = 0.05


def manifest_path(folder: Path | None = None) -> Path:
    root = folder or data_dir(Path.cwd())
    # Prefer data/archive_manifest.json; fall back to repo-root copy
    candidates = [
        root / MANIFEST_NAME,
        root.parent / MANIFEST_NAME,
        Path(__file__).resolve().parents[1] / "data" / MANIFEST_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return root / MANIFEST_NAME


def load_manifest(folder: Path | None = None) -> list[dict]:
    path = manifest_path(folder)
    if not path.is_file():
        raise ArchiveCsvError(f"Missing archive manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files") or []
    if not isinstance(files, list) or not files:
        raise ArchiveCsvError("Archive manifest has no files.")
    return files


def _looks_complete(path: Path, approx_bytes: int | None) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size < MIN_OK_BYTES:
        return False
    if approx_bytes and approx_bytes > 0:
        low = approx_bytes * (1.0 - SIZE_TOLERANCE)
        return size >= low
    return True


def missing_archives(folder: Path | None = None) -> list[dict]:
    """Return manifest entries that are absent or too small on disk."""
    root = folder or data_dir(Path.cwd())
    missing: list[dict] = []
    for entry in load_manifest(root):
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        dest = root / name
        approx = int(entry.get("approx_bytes") or 0)
        if not _looks_complete(dest, approx or None):
            missing.append(entry)
    return missing


def _gdown_bin() -> str:
    which = shutil.which("gdown")
    if which:
        return which
    # Prefer project venv
    venv = Path(sys.executable).resolve().parent / "gdown"
    if venv.is_file():
        return str(venv)
    return "gdown"


def restore_archives(
    folder: Path | None = None,
    *,
    only_missing: bool = True,
) -> list[str]:
    """
    Download archive CSVs listed in the manifest into data/.

    Returns list of filenames written (or confirmed present).
    """
    root = folder or data_dir(Path.cwd())
    root.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(root)
    gdown = _gdown_bin()
    done: list[str] = []

    for entry in entries:
        name = str(entry.get("name") or "").strip()
        drive_id = str(entry.get("drive_id") or "").strip()
        approx = int(entry.get("approx_bytes") or 0)
        if not name or not drive_id:
            continue
        dest = root / name
        if only_missing and _looks_complete(dest, approx or None):
            done.append(name)
            continue
        if dest.exists():
            dest.unlink()
        url = f"https://drive.google.com/uc?id={drive_id}"
        try:
            subprocess.run(
                [gdown, "-O", str(dest), url],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ArchiveCsvError(
                "gdown is not installed. Run: pip install gdown"
            ) from exc
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ArchiveCsvError(
                f"Drive restore failed for {name}: {err[:400]}"
            ) from exc
        if not _looks_complete(dest, approx or None):
            raise ArchiveCsvError(
                f"Downloaded {name} looks incomplete "
                f"({dest.stat().st_size if dest.exists() else 0} bytes)."
            )
        done.append(name)
    return done
