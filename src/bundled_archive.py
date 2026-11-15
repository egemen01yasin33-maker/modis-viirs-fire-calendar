"""
Shippable 2023–2026 FIRMS extracts for hosted Archive mode.

Full multi-GB archives stay local (data/_cache/fires.duckdb). Cursor Apps and
GitHub get these compact regional Parquet files so users can pick dates and
compare periods without downloading the worldwide dump.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.archive_csv import ArchiveCsvError, MAX_FILTERED_ROWS, db_path
from src.harmonize import MATCHED, MODIS_ONLY, VIIRS_ONLY

ARCHIVE_DATE_MIN = date(2023, 1, 1)
ARCHIVE_DATE_MAX = date(2026, 11, 15)

# Region display name → bundled parquet basename
BUNDLED_REGION_FILES: dict[str, str] = {
    "Turkey": "turkey_2023_2026.parquet",
    "Aegean": "aegean_2023_2026.parquet",
    "Central Anatolia": "central_anatolia_2023_2026.parquet",
    "Marmara": "marmara_2023_2026.parquet",
    "California": "california_2023_2026.parquet",
    "Mediterranean": "mediterranean_2023_2026.parquet",
}


def bundled_dir(root: Path) -> Path:
    return root / "data" / "bundled"


def has_full_archive_db(root: Path) -> bool:
    path = db_path(root / "data")
    return path.is_file() and path.stat().st_size > 1_000_000


# Turkey-first order for the UI (Global last — slow / sparse on hosted demos)
UI_REGION_ORDER: tuple[str, ...] = (
    "Turkey",
    "Aegean",
    "Central Anatolia",
    "Marmara",
    "Mediterranean",
    "California",
    "Amazon",
    "Australia",
    "Indonesia",
    "Siberia",
    "Global",
)


def list_archive_regions(root: Path) -> list[str]:
    """Regions the UI can query in Archive mode (Turkey first)."""
    if has_full_archive_db(root):
        from src.firms_api import PRESET_REGIONS

        available = set(PRESET_REGIONS.keys())
    else:
        base = bundled_dir(root)
        available = {
            name
            for name, fname in BUNDLED_REGION_FILES.items()
            if (base / fname).is_file()
        }
    ordered = [name for name in UI_REGION_ORDER if name in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def pick_bundled_region_for_bbox(
    root: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
    """
    Choose the smallest shipped extract that covers the custom bbox center,
    falling back to Mediterranean / Turkey when possible.
    """
    from src.firms_api import PRESET_REGIONS

    available = set(list_archive_regions(root))
    cx = (min_lon + max_lon) / 2.0
    cy = (min_lat + max_lat) / 2.0

    # Prefer smallest containing preset among bundled files
    candidates: list[tuple[float, str]] = []
    for name, fname in BUNDLED_REGION_FILES.items():
        if name not in available or name not in PRESET_REGIONS:
            continue
        a, b, c, d = PRESET_REGIONS[name]
        if a <= cx <= c and b <= cy <= d:
            area = (c - a) * (d - b)
            candidates.append((area, name))
    if candidates:
        candidates.sort()
        return candidates[0][1]
    for fallback in ("Mediterranean", "Turkey", "California"):
        if fallback in available:
            return fallback
    if available:
        return sorted(available)[0]
    raise ArchiveCsvError("No bundled archive extracts are available on this host.")


def archive_date_bounds(root: Path) -> tuple[date, date]:
    if has_full_archive_db(root):
        try:
            import duckdb

            con = duckdb.connect(db_path(root / "data").as_posix(), read_only=True)
            try:
                lo, hi = con.execute(
                    "SELECT min(acq_date), max(acq_date) FROM hotspots"
                ).fetchone()
            finally:
                con.close()
            if lo and hi:
                return pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
        except Exception:  # noqa: BLE001
            pass
    return ARCHIVE_DATE_MIN, ARCHIVE_DATE_MAX


def _parse_bounds(start: date | str, end: date | str) -> tuple[str, str]:
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    if start_s > end_s:
        start_s, end_s = end_s, start_s
    return start_s, end_s


def _bundled_parquet_paths(root: Path, region: str | None = None) -> list[Path]:
    """Parquet files to query: one preset extract, or all bundled for custom bbox."""
    base = bundled_dir(root)
    if region and region in BUNDLED_REGION_FILES:
        path = base / BUNDLED_REGION_FILES[region]
        if path.is_file():
            return [path]
    paths = sorted(
        base / fname
        for fname in BUNDLED_REGION_FILES.values()
        if (base / fname).is_file()
    )
    if not paths:
        raise ArchiveCsvError(
            "No bundled archive extracts found under data/bundled/. "
            "Re-deploy with the 2023–2026 regional Parquet files."
        )
    return paths


def _query_parquet_window(
    paths: list[Path],
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_s: str,
    end_s: str,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    import duckdb

    src = paths[0].as_posix() if len(paths) == 1 else [p.as_posix() for p in paths]
    con = duckdb.connect()
    try:
        daily = con.execute(
            """
            SELECT
              CAST(acq_date AS DATE) AS period,
              count(*)::BIGINT AS total,
              coalesce(sum(TRY_CAST(frp AS DOUBLE)), 0)::DOUBLE AS frp_sum,
              sum(
                CASE
                  WHEN lower(coalesce(instrument, '')) LIKE '%modis%'
                    OR lower(coalesce(satellite, '')) IN ('terra', 'aqua')
                  THEN 1 ELSE 0
                END
              )::BIGINT AS modis_n,
              sum(
                CASE
                  WHEN lower(coalesce(instrument, '')) LIKE '%modis%'
                    OR lower(coalesce(satellite, '')) IN ('terra', 'aqua')
                  THEN 0 ELSE 1
                END
              )::BIGINT AS viirs_n
            FROM read_parquet(?, union_by_name=true)
            WHERE CAST(acq_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND CAST(longitude AS DOUBLE) BETWEEN ? AND ?
              AND CAST(latitude AS DOUBLE) BETWEEN ? AND ?
            GROUP BY 1
            ORDER BY 1
            """,
            [src, start_s, end_s, min_lon, max_lon, min_lat, max_lat],
        ).df()
        total_count = int(daily["total"].sum()) if not daily.empty else 0
        raw = con.execute(
            """
            SELECT
              CAST(latitude AS DOUBLE) AS latitude,
              CAST(longitude AS DOUBLE) AS longitude,
              CAST(acq_date AS VARCHAR) AS acq_date,
              CAST(acq_time AS VARCHAR) AS acq_time,
              CAST(satellite AS VARCHAR) AS satellite,
              CAST(instrument AS VARCHAR) AS instrument,
              CAST(confidence AS VARCHAR) AS confidence,
              TRY_CAST(frp AS DOUBLE) AS frp
            FROM read_parquet(?, union_by_name=true)
            WHERE CAST(acq_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND CAST(longitude AS DOUBLE) BETWEEN ? AND ?
              AND CAST(latitude AS DOUBLE) BETWEEN ? AND ?
            ORDER BY frp DESC NULLS LAST
            LIMIT ?
            """,
            [src, start_s, end_s, min_lon, max_lon, min_lat, max_lat, int(max_rows)],
        ).df()
    finally:
        con.close()
    return raw, daily, total_count


def load_archive_window(
    root: Path,
    *,
    region: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start: date | str,
    end: date | str,
    max_rows: int = MAX_FILTERED_ROWS,
) -> dict:
    """
    Return map sample + daily activity for one region/date window.

    Prefers the full DuckDB cache when present; otherwise uses bundled Parquet.
    Custom coordinate boxes query all bundled extracts (or the full DB).
    """
    start_s, end_s = _parse_bounds(start, end)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ArchiveCsvError(
            "Invalid coordinates: require west < east and south < north."
        )

    if has_full_archive_db(root):
        from src.archive_csv import load_daily_activity, load_filtered_csvs

        folder = root / "data"
        daily, daily_stats = load_daily_activity(
            folder,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            start_s,
            end_s,
        )
        raw, files, map_stats = load_filtered_csvs(
            folder,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            start_s,
            end_s,
            max_rows=max_rows,
            known_total=int(daily_stats.get("total_count") or 0),
        )
        return {
            "raw": raw,
            "daily": daily,
            "total_count": int(daily_stats.get("total_count") or 0),
            "files": files,
            "source": "full-archive",
            "map_stats": map_stats,
        }

    try:
        paths = _bundled_parquet_paths(
            root, None if region in {"Custom", "custom", ""} else region
        )
        raw, daily, total_count = _query_parquet_window(
            paths,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            start_s=start_s,
            end_s=end_s,
            max_rows=max_rows,
        )
    except ArchiveCsvError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArchiveCsvError(f"Could not read bundled archive: {exc}") from exc

    if not daily.empty:
        daily["period"] = pd.to_datetime(daily["period"])
        daily[MODIS_ONLY] = daily["modis_n"].astype(int)
        daily[VIIRS_ONLY] = daily["viirs_n"].astype(int)
        daily[MATCHED] = 0
        daily = daily.drop(columns=["modis_n", "viirs_n"], errors="ignore")

    return {
        "raw": raw,
        "daily": daily,
        "total_count": total_count,
        "files": [p.name for p in paths],
        "source": "bundled",
        "map_stats": {},
    }


def daily_to_series_json(daily: pd.DataFrame) -> dict:
    """Chart payload: daily sensor counts + FRP (Fire Radiative Power) sum."""
    from src.harmonize import CATEGORY_COLORS

    empty = {
        "labels": [],
        "matched": [],
        "modis": [],
        "viirs": [],
        "frp": [],
        "colors": {
            "matched": CATEGORY_COLORS[MATCHED],
            "modis": CATEGORY_COLORS[MODIS_ONLY],
            "viirs": CATEGORY_COLORS[VIIRS_ONLY],
            "frp": "#f0c674",
        },
    }
    if daily is None or daily.empty:
        return empty
    labels = [pd.Timestamp(p).strftime("%Y-%m-%d") for p in daily["period"].tolist()]
    matched = (
        [int(x) for x in daily[MATCHED].tolist()]
        if MATCHED in daily.columns
        else [0] * len(labels)
    )
    modis = (
        [int(x) for x in daily[MODIS_ONLY].tolist()]
        if MODIS_ONLY in daily.columns
        else [0] * len(labels)
    )
    viirs = (
        [int(x) for x in daily[VIIRS_ONLY].tolist()]
        if VIIRS_ONLY in daily.columns
        else [0] * len(labels)
    )
    if "frp_sum" in daily.columns:
        frp = [round(float(x or 0), 2) for x in daily["frp_sum"].tolist()]
    else:
        frp = [0.0] * len(labels)
    return {
        "labels": labels,
        "matched": matched,
        "modis": modis,
        "viirs": viirs,
        "frp": frp,
        "colors": {
            "matched": CATEGORY_COLORS[MATCHED],
            "modis": CATEGORY_COLORS[MODIS_ONLY],
            "viirs": CATEGORY_COLORS[VIIRS_ONLY],
            "frp": "#f0c674",
        },
    }
