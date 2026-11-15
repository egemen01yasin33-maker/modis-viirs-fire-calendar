"""
Restore full FIRMS archive CSVs from GitHub Release (preferred) or Google Drive.

Uses data/archive_manifest.json. Existing local files that look complete are
skipped so you never re-download unnecessarily.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from src.archive_csv import ArchiveCsvError, data_dir

MANIFEST_NAME = "archive_manifest.json"
# Accept local files within 5% of declared size (or at least 50 MiB).
MIN_OK_BYTES = 50 * 1024 * 1024
SIZE_TOLERANCE = 0.05


def manifest_path(folder: Path | None = None) -> Path:
    root = folder or data_dir(Path.cwd())
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
    if size < MIN_OK_BYTES and (not approx_bytes or approx_bytes >= MIN_OK_BYTES):
        # Tiny NRT file (~67 MiB) is above MIN_OK; keep threshold for safety
        if approx_bytes and approx_bytes < MIN_OK_BYTES:
            low = approx_bytes * (1.0 - SIZE_TOLERANCE)
            return size >= low
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
    venv = Path(sys.executable).resolve().parent / "gdown"
    if venv.is_file():
        return str(venv)
    return "gdown"


def _download_http(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _download_drive(drive_id: str, dest: Path) -> None:
    gdown = _gdown_bin()
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
            f"Drive restore failed for {dest.name}: {err[:400]}"
        ) from exc


def restore_archives(
    folder: Path | None = None,
    *,
    only_missing: bool = True,
) -> list[str]:
    """
    Download archive CSVs listed in the manifest into data/.

    Prefers GitHub Release URLs; falls back to Google Drive IDs.
    Returns list of filenames written (or confirmed present).
    """
    root = folder or data_dir(Path.cwd())
    root.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(root)
    done: list[str] = []

    for entry in entries:
        name = str(entry.get("name") or "").strip()
        github_url = str(entry.get("github_url") or "").strip()
        drive_id = str(entry.get("drive_id") or "").strip()
        approx = int(entry.get("approx_bytes") or 0)
        if not name:
            continue
        dest = root / name
        if only_missing and _looks_complete(dest, approx or None):
            done.append(name)
            continue
        if dest.exists():
            dest.unlink()

        errors: list[str] = []
        if github_url:
            try:
                _download_http(github_url, dest)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"GitHub: {exc}")
                if dest.exists():
                    dest.unlink()
        if (not dest.exists() or not _looks_complete(dest, approx or None)) and drive_id:
            try:
                _download_drive(drive_id, dest)
            except ArchiveCsvError as exc:
                errors.append(str(exc))
        if not _looks_complete(dest, approx or None):
            detail = "; ".join(errors) if errors else "unknown error"
            raise ArchiveCsvError(
                f"Could not restore {name} ({detail}). "
                "See GitHub Release archive-data-v1."
            )
        done.append(name)
    return done
