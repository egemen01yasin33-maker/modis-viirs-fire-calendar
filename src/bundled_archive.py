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


def list_archive_regions(root: Path) -> list[str]:
    """Regions the UI can query in Archive mode."""
    if has_full_archive_db(root):
        from src.firms_api import PRESET_REGIONS

        return list(PRESET_REGIONS.keys())
    base = bundled_dir(root)
    return [
        name
        for name, fname in BUNDLED_REGION_FILES.items()
        if (base / fname).is_file()
    ]


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
    """
    start_s, end_s = _parse_bounds(start, end)

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

    fname = BUNDLED_REGION_FILES.get(region)
    if not fname:
        raise ArchiveCsvError(
            f"Archive extract for “{region}” is not bundled on this host. "
            "Pick Turkey, Aegean, Central Anatolia, Marmara, California, or Mediterranean."
        )
    path = bundled_dir(root) / fname
    if not path.is_file():
        raise ArchiveCsvError(
            f"Missing bundled archive file `{fname}`. Re-deploy with data/bundled/."
        )

    try:
        import duckdb

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
                FROM read_parquet(?)
                WHERE CAST(acq_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                  AND CAST(longitude AS DOUBLE) BETWEEN ? AND ?
                  AND CAST(latitude AS DOUBLE) BETWEEN ? AND ?
                GROUP BY 1
                ORDER BY 1
                """,
                [
                    path.as_posix(),
                    start_s,
                    end_s,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                ],
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
                FROM read_parquet(?)
                WHERE CAST(acq_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                  AND CAST(longitude AS DOUBLE) BETWEEN ? AND ?
                  AND CAST(latitude AS DOUBLE) BETWEEN ? AND ?
                ORDER BY frp DESC NULLS LAST
                LIMIT ?
                """,
                [
                    path.as_posix(),
                    start_s,
                    end_s,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                    int(max_rows),
                ],
            ).df()
        finally:
            con.close()
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
        "files": [fname],
        "source": "bundled",
        "map_stats": {},
    }


def daily_to_series_json(daily: pd.DataFrame) -> dict:
    """Chart payload from exact daily sensor counts (Matched often 0 in aggregate)."""
    from src.harmonize import CATEGORY_COLORS

    if daily is None or daily.empty:
        return {
            "labels": [],
            "matched": [],
            "modis": [],
            "viirs": [],
            "colors": {
                "matched": CATEGORY_COLORS[MATCHED],
                "modis": CATEGORY_COLORS[MODIS_ONLY],
                "viirs": CATEGORY_COLORS[VIIRS_ONLY],
            },
        }
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
    return {
        "labels": labels,
        "matched": matched,
        "modis": modis,
        "viirs": viirs,
        "colors": {
            "matched": CATEGORY_COLORS[MATCHED],
            "modis": CATEGORY_COLORS[MODIS_ONLY],
            "viirs": CATEGORY_COLORS[VIIRS_ONLY],
        },
    }
