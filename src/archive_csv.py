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
CACHE_DIRNAME = "_cache"
DB_FILENAME = "fires.duckdb"
CHUNK_SIZE = 250_000
# Map sample size — charts use full-range SQL aggregates instead.
# Keep this modest: Plotly/browser freezes above a few thousand markers.
# Spatial sampling (not random) preserves sparse regions like Turkey.
MAX_FILTERED_ROWS = 6_000
# Tiny fixtures kept in data/ for unit tests — skip in Archive mode inventory
SKIP_FILENAMES = frozenset({"sample_firms.csv"})
# Columns needed for harmonization + map (keep the query narrow/fast)
QUERY_COLUMNS = (
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "archive_file",
)
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


def cache_dir(folder: Path) -> Path:
    """Parquet cache directory beside the CSVs (gitignored)."""
    path = folder / CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parquet_cache(folder: Path) -> tuple[list[Path], int]:
    """
    Build/refresh Parquet siblings for each archive CSV.

    Returns (parquet_paths, number_rebuilt). One-time cost ~tens of seconds
    for multi-GB dumps; later region/date filters are sub-second.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ArchiveCsvError(
            "duckdb is required for fast archive queries. "
            "Install with: pip install duckdb pyarrow"
        ) from exc

    files = list_archive_csv_files(folder) or list_csv_files(folder)
    if not files:
        return [], 0

    out_dir = cache_dir(folder)
    con = duckdb.connect()
    rebuilt = 0
    paths: list[Path] = []

    for csv_path in files:
        pq_path = out_dir / f"{csv_path.stem}.parquet"
        needs = (
            not pq_path.exists()
            or pq_path.stat().st_mtime_ns < csv_path.stat().st_mtime_ns
        )
        if needs:
            # Validate header before spending time on conversion
            header = pd.read_csv(csv_path, nrows=0)
            missing = validate_firms_columns(header)
            if missing:
                continue
            src_name = csv_path.name.replace("'", "''")
            con.execute(
                f"""
                COPY (
                  SELECT
                    *,
                    '{src_name}' AS archive_file
                  FROM read_csv_auto(
                    '{csv_path.as_posix()}',
                    header=true,
                    sample_size=-1
                  )
                ) TO '{pq_path.as_posix()}'
                  (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            rebuilt += 1
        if pq_path.exists():
            paths.append(pq_path)

    return paths, rebuilt


def db_path(folder: Path) -> Path:
    """Persistent DuckDB file used for instant bbox/date filters."""
    return cache_dir(folder) / DB_FILENAME


def ensure_hotspots_db(folder: Path) -> tuple[Path, bool]:
    """
    Build a sorted DuckDB `hotspots` table from the Parquet cache.

    Returns (db_path, rebuilt). Rebuilt only when CSVs/parquets are newer.
    """
    import duckdb

    parquet_paths, pq_rebuilt = ensure_parquet_cache(folder)
    if not parquet_paths:
        raise ArchiveCsvError(
            "Could not prepare archive cache from the CSV files in data/."
        )

    # Older parquet without archive_file → rebuild
    probe_con = duckdb.connect()
    probe = probe_con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_paths[0].as_posix()}')"
    ).fetchdf()
    probe_con.close()
    cols = {str(c).lower() for c in probe["column_name"].tolist()}
    if "archive_file" not in cols:
        for p in parquet_paths:
            p.unlink(missing_ok=True)
        parquet_paths, pq_rebuilt = ensure_parquet_cache(folder)
        if not parquet_paths:
            raise ArchiveCsvError(
                "Could not prepare archive cache from the CSV files in data/."
            )

    path = db_path(folder)
    newest_src = max(p.stat().st_mtime_ns for p in parquet_paths)
    needs = (not path.exists()) or path.stat().st_mtime_ns < newest_src or pq_rebuilt
    if path.exists() and not needs:
        # Schema upgrades (e.g. adding frp) force a one-time rebuild
        probe = duckdb.connect(path.as_posix(), read_only=True)
        try:
            cols = {
                str(r[0]).lower()
                for r in probe.execute("DESCRIBE hotspots").fetchall()
            }
        finally:
            probe.close()
        if "frp" not in cols:
            needs = True
    if needs:
        if path.exists():
            path.unlink()
        con = duckdb.connect(path.as_posix())
        con.execute(
            """
            CREATE TABLE hotspots AS
            SELECT
              CAST(latitude AS DOUBLE) AS latitude,
              CAST(longitude AS DOUBLE) AS longitude,
              CAST(acq_date AS DATE) AS acq_date,
              CAST(acq_time AS VARCHAR) AS acq_time,
              CAST(satellite AS VARCHAR) AS satellite,
              CAST(instrument AS VARCHAR) AS instrument,
              CAST(confidence AS VARCHAR) AS confidence,
              TRY_CAST(frp AS DOUBLE) AS frp,
              CAST(archive_file AS VARCHAR) AS archive_file
            FROM read_parquet(?, union_by_name=true)
            ORDER BY acq_date, longitude, latitude
            """,
            [[p.as_posix() for p in parquet_paths]],
        )
        con.execute("CHECKPOINT")
        con.close()
        return path, True
    return path, False


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
    known_total: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Keep rows inside bbox + date window.

    Prefers a persistent DuckDB table (near-instant). Falls back to
    chunked pandas CSV scans if DuckDB is unavailable.
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

    try:
        return _load_filtered_duckdb(
            folder,
            files,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            start_s=start_s,
            end_s=end_s,
            max_rows=max_rows,
            known_total=known_total,
        )
    except ArchiveCsvError:
        raise
    except Exception:
        # Fall back to pandas chunk scan (slower but dependency-light)
        return _load_filtered_pandas(
            files,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            start_s=start_s,
            end_s=end_s,
            chunksize=chunksize,
            max_rows=max_rows,
        )


def _map_sample_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    total_n: int,
    max_rows: int,
) -> tuple[float, int]:
    """
    Return (cell_deg, per_cell) for spatial downsampling.

    Fine cells + several points per cell keep sparse countries visible on
    Global maps while still capping Plotly/payload size.
    """
    lon_span = max(float(max_lon) - float(min_lon), 1e-3)
    lat_span = max(float(max_lat) - float(min_lat), 1e-3)
    if lon_span >= 100 or lat_span >= 80:
        cell = 0.5
    elif lon_span >= 25 or lat_span >= 20:
        cell = 0.25
    else:
        cell = 0.1
    # Rough cell count for this bbox; use it to spend the row budget
    n_cells = max((lon_span / cell) * (lat_span / cell), 1.0)
    # Assume fires occupy a fraction of cells; spend budget across them
    occupied_guess = max(min(n_cells * 0.35, float(total_n)), 1.0)
    per_cell = int(max(1, min(12, round(max_rows / occupied_guess) + 1)))
    return float(cell), int(per_cell)


def _load_filtered_duckdb(
    folder: Path,
    files: list[Path],
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_s: str,
    end_s: str,
    max_rows: int,
    known_total: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    import duckdb

    path, rebuilt = ensure_hotspots_db(folder)
    con = duckdb.connect(path.as_posix(), read_only=True)
    select_sql = """
        latitude, longitude,
        CAST(acq_date AS VARCHAR) AS acq_date,
        acq_time, satellite, instrument, confidence, archive_file
    """
    where_sql = """
        acq_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        AND longitude BETWEEN ? AND ?
        AND latitude BETWEEN ? AND ?
    """
    where_params = [start_s, end_s, min_lon, max_lon, min_lat, max_lat]
    try:
        total_n = (
            int(known_total)
            if known_total is not None
            else int(
                con.execute(
                    f"SELECT count(*) FROM hotspots WHERE {where_sql}",
                    where_params,
                ).fetchone()[0]
            )
        )
        if total_n == 0:
            df = pd.DataFrame()
        elif total_n <= max_rows:
            df = con.execute(
                f"""
                SELECT {select_sql}
                FROM hotspots
                WHERE {where_sql}
                """,
                where_params,
            ).df()
        elif total_n <= max_rows * 4:
            # Moderate oversize (e.g. a few Global days): proportional hash
            # sample so sparse regions keep a fair share of the budget.
            modulus = 10_000
            threshold = max(1, int(modulus * (max_rows / total_n)) + 50)
            df = con.execute(
                f"""
                SELECT {select_sql}
                FROM hotspots
                WHERE {where_sql}
                  AND (abs(hash(latitude, longitude, acq_time)) % ?) < ?
                ORDER BY hash(latitude * 1000, longitude * 1000)
                LIMIT ?
                """,
                [*where_params, modulus, threshold, max_rows],
            ).df()
            if len(df) < min(max_rows // 5, 2_000):
                df = con.execute(
                    f"""
                    SELECT {select_sql}
                    FROM hotspots
                    WHERE {where_sql}
                    LIMIT ?
                    """,
                    [*where_params, max_rows],
                ).df()
        else:
            # Huge windows: fine spatial grid + several points per cell,
            # shuffled before LIMIT so west-sorted storage does not bias.
            cell, per_cell = _map_sample_grid(
                min_lon, min_lat, max_lon, max_lat, total_n, max_rows
            )
            df = con.execute(
                f"""
                SELECT {select_sql}
                FROM (
                  SELECT
                    latitude, longitude, acq_date, acq_time, satellite,
                    instrument, confidence, archive_file,
                    ROW_NUMBER() OVER (
                      PARTITION BY
                        CAST(floor(latitude / ?) AS INTEGER),
                        CAST(floor(longitude / ?) AS INTEGER)
                      ORDER BY hash(acq_time, satellite, latitude)
                    ) AS rn
                  FROM hotspots
                  WHERE {where_sql}
                ) AS gridded
                WHERE rn <= ?
                ORDER BY hash(latitude * 1000, longitude * 1000)
                LIMIT ?
                """,
                [cell, cell, *where_params, per_cell, max_rows],
            ).df()
            if len(df) < min(max_rows // 5, 2_000):
                df = con.execute(
                    f"""
                    SELECT {select_sql}
                    FROM hotspots
                    WHERE {where_sql}
                    LIMIT ?
                    """,
                    [*where_params, max_rows],
                ).df()
    finally:
        con.close()

    truncated = total_n > max_rows
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    names = [p.name for p in files]
    stats = {
        "rows_scanned": total_n,
        "rows_kept": int(len(df)),
        "truncated": truncated,
        "errors": [],
        "engine": "duckdb-table",
        "cache_rebuilt": int(rebuilt),
    }
    return df, names, stats


def load_daily_activity(
    folder: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start: date | datetime | str,
    end: date | datetime | str,
) -> tuple[pd.DataFrame, dict]:
    """
    Exact daily totals for any bbox + date span via DuckDB GROUP BY.

    Does not load individual points — safe for Global + multi-year charts.
    Returns (daily_df, stats) with columns:
      period, total, frp_sum, MODIS-only, VIIRS-only, Matched=0,
      is_anomaly, mean_total, threshold
    """
    from src.harmonize import MATCHED, MODIS_ONLY, VIIRS_ONLY

    ensure_hotspots_db(folder)
    import duckdb

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    path = db_path(folder)
    con = duckdb.connect(path.as_posix(), read_only=True)
    try:
        daily = con.execute(
            """
            SELECT
              acq_date AS period,
              count(*)::BIGINT AS total,
              coalesce(sum(frp), 0)::DOUBLE AS frp_sum,
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
            FROM hotspots
            WHERE acq_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND longitude BETWEEN ? AND ?
              AND latitude BETWEEN ? AND ?
            GROUP BY acq_date
            ORDER BY acq_date
            """,
            [start_s, end_s, min_lon, max_lon, min_lat, max_lat],
        ).df()
        frp_row = con.execute(
            """
            SELECT
              count(*)::BIGINT AS n,
              coalesce(sum(frp), 0)::DOUBLE AS frp_sum
            FROM hotspots
            WHERE acq_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND longitude BETWEEN ? AND ?
              AND latitude BETWEEN ? AND ?
            """,
            [start_s, end_s, min_lon, max_lon, min_lat, max_lat],
        ).fetchone()
        total_count = int(frp_row[0] or 0)
        frp_total = float(frp_row[1] or 0.0)
    finally:
        con.close()

    if daily.empty:
        empty = pd.DataFrame(
            columns=[
                "period",
                "total",
                "frp_sum",
                MATCHED,
                MODIS_ONLY,
                VIIRS_ONLY,
                "is_anomaly",
                "mean_total",
                "threshold",
            ]
        )
        return empty, {
            "total_count": 0,
            "days": 0,
            "frp_sum": 0.0,
            "frp_avg": 0.0,
        }

    daily["period"] = pd.to_datetime(daily["period"])
    daily[MODIS_ONLY] = daily["modis_n"].astype(int)
    daily[VIIRS_ONLY] = daily["viirs_n"].astype(int)
    daily[MATCHED] = 0
    daily["total"] = daily["total"].astype(int)
    daily["frp_sum"] = daily["frp_sum"].astype(float)
    positive = daily.loc[daily["total"] > 0, "total"]
    mean_total = float(positive.mean()) if len(positive) else 0.0
    threshold = mean_total * 2.0
    daily["mean_total"] = mean_total
    daily["threshold"] = threshold
    daily["is_anomaly"] = (daily["total"] > threshold) & (mean_total > 0)
    daily = daily.drop(columns=["modis_n", "viirs_n"])
    stats = {
        "total_count": total_count,
        "days": int(len(daily)),
        "frp_sum": frp_total,
        "frp_avg": (frp_total / total_count) if total_count else 0.0,
    }
    return daily, stats


def load_monthly_activity(
    folder: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start: date | datetime | str,
    end: date | datetime | str,
) -> pd.DataFrame:
    """
    Monthly hotspot counts + FRP totals for the selected bbox/dates.

    Columns: period (month-start timestamp), total, frp_sum,
    pct_change_total, pct_change_frp (vs previous month in the result).
    """
    ensure_hotspots_db(folder)
    import duckdb

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    path = db_path(folder)
    con = duckdb.connect(path.as_posix(), read_only=True)
    try:
        monthly = con.execute(
            """
            SELECT
              date_trunc('month', acq_date) AS period,
              count(*)::BIGINT AS total,
              coalesce(sum(frp), 0)::DOUBLE AS frp_sum
            FROM hotspots
            WHERE acq_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
              AND longitude BETWEEN ? AND ?
              AND latitude BETWEEN ? AND ?
            GROUP BY 1
            ORDER BY 1
            """,
            [start_s, end_s, min_lon, max_lon, min_lat, max_lat],
        ).df()
    finally:
        con.close()

    if monthly.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "total",
                "frp_sum",
                "pct_change_total",
                "pct_change_frp",
            ]
        )

    monthly["period"] = pd.to_datetime(monthly["period"])
    monthly["total"] = monthly["total"].astype(int)
    monthly["frp_sum"] = monthly["frp_sum"].astype(float)
    prev_total = monthly["total"].shift(1)
    prev_frp = monthly["frp_sum"].shift(1)
    monthly["pct_change_total"] = (
        (monthly["total"] - prev_total) / prev_total.replace(0, pd.NA) * 100.0
    )
    monthly["pct_change_frp"] = (
        (monthly["frp_sum"] - prev_frp) / prev_frp.replace(0, pd.NA) * 100.0
    )
    return monthly


def _load_filtered_pandas(
    files: list[Path],
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_s: str,
    end_s: str,
    chunksize: int,
    max_rows: int,
) -> tuple[pd.DataFrame, list[str], dict]:
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
            "engine": "pandas",
        }
        return empty, names, stats

    merged = pd.concat(frames, ignore_index=True)
    stats = {
        "rows_scanned": rows_scanned,
        "rows_kept": int(len(merged)),
        "truncated": truncated,
        "errors": errors,
        "engine": "pandas",
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
