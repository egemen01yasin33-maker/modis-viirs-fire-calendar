"""
Load and merge FIRMS Archive CSV files from the local data/ folder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.harmonize import validate_firms_columns


class ArchiveCsvError(Exception):
    """User-facing error while reading archive CSVs."""


DATA_DIRNAME = "data"


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


def load_and_merge_csvs(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read every .csv in folder and concatenate.

    Returns (merged_dataframe, list_of_filenames).
    Raises ArchiveCsvError when the folder is empty or files are unreadable /
    missing required FIRMS columns.
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
