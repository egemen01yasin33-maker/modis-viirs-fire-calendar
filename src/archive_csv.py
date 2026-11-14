"""
Load and merge FIRMS Archive CSV files from the local data/ folder.

Large multi-GB archives are streamed in chunks and filtered by bbox + dates
so Streamlit never materializes the full worldwide year in memory.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.harmonize import validate_firms_columns


class ArchiveCsvError(Exception):
    """User-facing error while reading archive CSVs."""


DATA_DIRNAME = "data"
CHUNK_SIZE = 250_000
# Soft cap after bbox+date filter — beyond this, ask the user to narrow scope
MAX_FILTERED_ROWS = 400_000
# Tiny fixtures kept in data/ for unit tests — skip in Archive mode inventory
SKIP_FILENAMES = frozenset({"sample_firms.csv"})
USECOLS = (
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
)


def data_dir(root: Path) -> Path:
    """Return (and create) the project data/ directory."""
    path = root / DATA_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_csv_files(folder: Path) -> list[Path]:
    """Sorted list of *.csv files directly under folder (non-recursive)."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.csv"))


def list_archive_csv_files(folder: Path) -> list[Path]:
    """Like list_csv_files but skips tiny test fixtures such as sample_firms.csv."""
    return [
        p
        for p in list_csv_files(folder)
        if p.name.lower() not in SKIP_FILENAMES
    ]


def describe_csv_files(folder: Path) -> list[dict]:
    """Lightweight inventory: name, size_bytes, size_label for each archive CSV."""
    rows: list[dict] = []
    for path in list_archive_csv_files(folder):
        size = path.stat().st_size
        rows.append(
            {
                "name": path.name,
                "size_bytes": size,
                "size_label": _format_bytes(size),
            }
        )
    return rows


def _format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def peek_archive_date_bounds(folder: Path) -> tuple[date | None, date | None]:
    """
    Infer min/max acq_date from the first and last data rows of each CSV.

    FIRMS archive dumps are typically sorted by acquisition time, so this is a
    cheap heuristic for UI defaults without scanning the whole file.
    """
    mins: list[date] = []
    maxs: list[date] = []
    files = list_archive_csv_files(folder) or list_csv_files(folder)
    for path in files:
        try:
            head = pd.read_csv(path, nrows=1)
        except Exception:  # noqa: BLE001
            continue
        cols = {str(c).strip().lower(): c for c in head.columns}
        if "acq_date" not in cols:
            continue
        first = _to_date(head.iloc[0][cols["acq_date"]])
        last = _peek_last_acq_date(path, cols["acq_date"])
        if first is not None:
            mins.append(first)
        if last is not None:
            maxs.append(last)
        elif first is not None:
            maxs.append(first)
    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)


def _to_date(value) -> date | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _peek_last_acq_date(path: Path, acq_col: str) -> date | None:
    """Read the last non-empty line and parse acq_date without full scan."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return None
            block = min(size, 256_000)
            fh.seek(size - block)
            tail = fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return None

    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    # Rebuild header map from a tiny head read
    try:
        header = pd.read_csv(path, nrows=0)
        col_names = list(header.columns)
        if acq_col not in col_names:
            # case-insensitive match
            lower = {c.lower(): c for c in col_names}
            acq_col = lower.get("acq_date", acq_col)
        idx = col_names.index(acq_col)
    except Exception:  # noqa: BLE001
        return None

    for line in reversed(lines):
        if line.lower().startswith("latitude"):
            continue
        parts = line.split(",")
        if idx >= len(parts):
            continue
        parsed = _to_date(parts[idx].strip())
        if parsed is not None:
            return parsed
    return None


def load_and_merge_csvs(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read every .csv in folder and concatenate (small files / tests).

    For multi-GB archives prefer ``load_filtered_csvs``.
    """
    files = list_csv_files(folder)
    if not files:
        raise ArchiveCsvError(
            f"No CSV files found in `{folder.as_posix()}/`. "
            "Download MODIS / VIIRS Archive CSVs from NASA FIRMS and place "
            "them in the data/ folder."
        )

    frames: list[pd.DataFrame] = []
    names: list[str] = []
    errors: list[str] = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: could not read ({exc})")
            continue

        missing = validate_firms_columns(df)
        if missing:
            errors.append(
                f"{path.name}: missing columns {', '.join(missing)}"
            )
            continue

        out = df.copy()
        out["archive_file"] = path.name
        frames.append(out)
        names.append(path.name)

    if not frames:
        detail = "; ".join(errors) if errors else "no usable rows"
        raise ArchiveCsvError(
            f"Could not load any valid FIRMS CSV from data/. {detail}"
        )

    merged = pd.concat(frames, ignore_index=True)
    return merged, names


def load_filtered_csvs(
    folder: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    chunksize: int = CHUNK_SIZE,
    max_rows: int = MAX_FILTERED_ROWS,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Stream every CSV under folder, keeping only rows inside bbox + date window.

    Returns (filtered_dataframe, filenames_used, stats).
    ``stats`` includes rows_scanned, rows_kept, truncated (bool).
    """
    files = list_archive_csv_files(folder) or list_csv_files(folder)
    if not files:
        raise ArchiveCsvError(
            f"No CSV files found in `{folder.as_posix()}/`. "
            "Download MODIS / VIIRS Archive CSVs from NASA FIRMS and place "
            "them in the data/ folder."
        )

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    names: list[str] = []
    errors: list[str] = []
    rows_scanned = 0
    rows_kept = 0
    truncated = False

    for path in files:
        try:
            kept, scanned, hit_cap = _filter_file_chunks(
                path,
                min_lon=min_lon,
                min_lat=min_lat,
                max_lon=max_lon,
                max_lat=max_lat,
                start_s=start_s,
                end_s=end_s,
                chunksize=chunksize,
                remaining=max_rows - rows_kept,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: could not read ({exc})")
            continue

        rows_scanned += scanned
        if kept is None:
            # header validation failed
            errors.append(f"{path.name}: missing required FIRMS columns")
            continue

        names.append(path.name)
        if not kept.empty:
            frames.append(kept)
            rows_kept += len(kept)
        if hit_cap:
            truncated = True
            break

    if not names and errors:
        raise ArchiveCsvError(
            "Could not load any valid FIRMS CSV from data/. "
            + "; ".join(errors)
        )

    if not frames:
        empty = pd.DataFrame()
        stats = {
            "rows_scanned": rows_scanned,
            "rows_kept": 0,
            "truncated": False,
            "errors": errors,
        }
        return empty, names, stats

    merged = pd.concat(frames, ignore_index=True)
    stats = {
        "rows_scanned": rows_scanned,
        "rows_kept": int(len(merged)),
        "truncated": truncated,
        "errors": errors,
    }
    return merged, names, stats


def _filter_file_chunks(
    path: Path,
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_s: str,
    end_s: str,
    chunksize: int,
    remaining: int,
) -> tuple[pd.DataFrame | None, int, bool]:
    """
    Filter one CSV by streaming chunks.

    Returns (kept_df_or_None_if_invalid, rows_scanned, hit_row_cap).
    """
    # Validate header first
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:  # noqa: BLE001
        raise ArchiveCsvError(f"{path.name}: {exc}") from exc

    missing = validate_firms_columns(header)
    if missing:
        return None, 0, False

    usecols = [c for c in header.columns if str(c).strip().lower() in {
        u.lower() for u in USECOLS
    } or str(c).strip().lower() in {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "confidence",
        "satellite",
        "instrument",
    }]

    parts: list[pd.DataFrame] = []
    scanned = 0
    kept = 0
    hit_cap = False

    reader: Iterable[pd.DataFrame] = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols if usecols else None,
        low_memory=False,
    )

    for chunk in reader:
        scanned += len(chunk)
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        # String compare on acq_date (YYYY-MM-DD) is fast and correct for ISO dates
        dates = chunk["acq_date"].astype(str).str.slice(0, 10)
        lat = pd.to_numeric(chunk["latitude"], errors="coerce")
        lon = pd.to_numeric(chunk["longitude"], errors="coerce")
        mask = (
            (lon >= min_lon)
            & (lon <= max_lon)
            & (lat >= min_lat)
            & (lat <= max_lat)
            & (dates >= start_s)
            & (dates <= end_s)
        )
        hit = chunk.loc[mask].copy()
        if hit.empty:
            continue
        hit["archive_file"] = path.name
        if kept + len(hit) > remaining:
            need = max(0, remaining - kept)
            hit = hit.iloc[:need]
            parts.append(hit)
            kept += len(hit)
            hit_cap = True
            break
        parts.append(hit)
        kept += len(hit)
        if kept >= remaining:
            hit_cap = True
            break

    if not parts:
        return pd.DataFrame(), scanned, hit_cap
    return pd.concat(parts, ignore_index=True), scanned, hit_cap


def save_uploaded_csvs(folder: Path, uploads) -> list[str]:
    """
    Save Streamlit UploadedFile objects into folder.

    Returns list of saved filenames. Overwrites same-named files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in uploads or []:
        name = Path(getattr(upload, "name", "") or "upload.csv").name
        if not name.lower().endswith(".csv"):
            name = f"{name}.csv"
        dest = folder / name
        dest.write_bytes(upload.getbuffer())
        saved.append(name)
    return saved


def filter_by_bbox(
    df: pd.DataFrame,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> pd.DataFrame:
    """Keep rows whose lat/lon fall inside the bbox."""
    if df.empty:
        return df
    work = df.copy()
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    mask = (
        (work["longitude"] >= min_lon)
        & (work["longitude"] <= max_lon)
        & (work["latitude"] >= min_lat)
        & (work["latitude"] <= max_lat)
    )
    return work.loc[mask].reset_index(drop=True)


def filter_by_dates(
    df: pd.DataFrame,
    start,
    end,
) -> pd.DataFrame:
    """
    Filter a harmonized frame by acq_datetime date window (inclusive).
    `start` / `end` may be date or Timestamp.
    """
    if df.empty or "acq_datetime" not in df.columns:
        return df
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df["acq_datetime"] >= start_ts) & (df["acq_datetime"] <= end_ts)
    return df.loc[mask].copy()
