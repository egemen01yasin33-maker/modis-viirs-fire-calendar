"""
MODIS ↔ VIIRS sıcak nokta (hot spot) harmonizasyon yardımcıları.

NASA FIRMS CSV satırlarını sensöre ayırır; 1 km ve ±6 saat eşiği içinde
eşleşen MODIS–VIIRS çiftlerini "Matched (High Confidence)" olarak işaretler.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Zorunlu FIRMS sütunları
REQUIRED_COLUMNS = (
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "confidence",
    "satellite",
)

# Eşleşme eşikleri
DISTANCE_KM = 1.0
TIME_WINDOW_HOURS = 6.0

MATCHED = "Matched (High Confidence)"
MODIS_ONLY = "MODIS-only"
VIIRS_ONLY = "VIIRS-only"

# Map / chart colors (turquoise scale on black UI)
CATEGORY_COLORS = {
    MATCHED: "#00e5d4",  # bright turquoise
    MODIS_ONLY: "#14919b",  # deep teal
    VIIRS_ONLY: "#7df9ff",  # light cyan
}

# MODIS / VIIRS uydu adları (FIRMS'te sık görülen değerler)
_MODIS_TOKENS = ("terra", "aqua", "modis")
_VIIRS_TOKENS = (
    "viirs",
    "npp",
    "suomi",
    "noaa-20",
    "noaa-21",
    "noaa20",
    "noaa21",
    "n20",
    "n21",
    "j1",
    "j2",
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını küçük harfe indirger ve boşlukları temizler."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def validate_firms_columns(df: pd.DataFrame) -> list[str]:
    """Eksik zorunlu sütunları döndürür (yoksa boş liste)."""
    cols = {str(c).strip().lower() for c in df.columns}
    return [c for c in REQUIRED_COLUMNS if c not in cols]


def classify_sensor(satellite: str, instrument: str | None = None) -> str | None:
    """
    Uydu / enstrüman alanından sensör sınıfını çıkarır.

    Dönüş: "MODIS", "VIIRS" veya tanınamazsa None.
    """
    text = f"{satellite or ''} {instrument or ''}".strip().lower()
    if not text:
        return None
    if any(tok in text for tok in _MODIS_TOKENS):
        return "MODIS"
    # Tek harf "N" / "N*" VIIRS (Suomi NPP) kodları
    sat = str(satellite or "").strip().lower()
    if sat in {"n", "n1", "n2"} or sat.startswith("n") and sat[1:].isdigit():
        return "VIIRS"
    if any(tok in text for tok in _VIIRS_TOKENS):
        return "VIIRS"
    return None


def parse_acquisition_datetime(df: pd.DataFrame) -> pd.Series:
    """
    acq_date + acq_time → timezone-naive UTC datetime.

    FIRMS acq_time genelde HHMM veya HMM (ör. 1345, 45).
    """
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    # Sayısal veya string HHMM → sıfırlarla 4 hane
    times = (
        df["acq_time"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(4)
    )
    # Geçersiz zamanları NaT'e düşürmek için pd.to_datetime kullan
    combined = dates.dt.strftime("%Y-%m-%d") + " " + times
    return pd.to_datetime(combined, format="%Y-%m-%d %H%M", errors="coerce")


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """İki nokta (veya nokta dizileri) arasındaki büyük daire mesafesi (km)."""
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(a))


def prepare_firms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham FIRMS tablosunu temizler, sensör ve zaman damgası ekler.

    Tanınmayan uydu satırları atılır.
    """
    missing = validate_firms_columns(df)
    if missing:
        raise ValueError(
            "Eksik FIRMS sütunları: " + ", ".join(missing)
        )

    out = _normalize_columns(df).copy()
    sat = out["satellite"].astype(str)
    inst = (
        out["instrument"].astype(str)
        if "instrument" in out.columns
        else pd.Series("", index=out.index)
    )
    text = (sat + " " + inst).str.lower()
    sensor = pd.Series(pd.NA, index=out.index, dtype=object)
    modis_mask = text.str.contains("|".join(_MODIS_TOKENS), regex=True, na=False)
    sensor = sensor.mask(modis_mask, "MODIS")
    sat_l = sat.str.strip().str.lower()
    n_code = sat_l.str.match(r"^n\d*$", na=False)
    viirs_mask = (
        (~modis_mask)
        & (
            text.str.contains("|".join(_VIIRS_TOKENS), regex=True, na=False)
            | n_code
        )
    )
    sensor = sensor.mask(viirs_mask, "VIIRS")
    out["sensor"] = sensor
    out["acq_datetime"] = parse_acquisition_datetime(out)

    # Geçerli koordinat / zaman / sensör
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude", "acq_datetime", "sensor"])
    out = out[out["sensor"].isin(["MODIS", "VIIRS"])].reset_index(drop=True)
    out["row_id"] = np.arange(len(out))
    return out


def harmonize_hotspots(
    df: pd.DataFrame,
    distance_km: float = DISTANCE_KM,
    time_window_hours: float = TIME_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    MODIS ve VIIRS noktalarını mesafe + zaman penceresine göre eşleştirir.

    Kurallar:
    - Mesafe ≤ distance_km (varsayılan 1 km)
    - |Δt| ≤ time_window_hours (varsayılan ±6 saat)
    - Eşleşen her iki nokta da MATCHED olur
    - Eşleşmeyenler sensöre göre MODIS-only / VIIRS-only kalır

    Not: Açgözlü bipartite eşleme — her MODIS için en yakın uygun VIIRS
    (henüz eşleşmemiş) seçilir; böylece bir VIIRS birden fazla MODIS'e
    bağlanmaz.
    """
    prepared = prepare_firms(df)
    if prepared.empty:
        return prepared.assign(category=pd.Series(dtype=str))

    categories = np.array(
        [
            MODIS_ONLY if s == "MODIS" else VIIRS_ONLY
            for s in prepared["sensor"]
        ],
        dtype=object,
    )

    modis = prepared[prepared["sensor"] == "MODIS"]
    viirs = prepared[prepared["sensor"] == "VIIRS"]

    if modis.empty or viirs.empty:
        result = prepared.copy()
        result["category"] = categories
        return result

    modis_lat = modis["latitude"].to_numpy(dtype=float)
    modis_lon = modis["longitude"].to_numpy(dtype=float)
    modis_t = modis["acq_datetime"].to_numpy(dtype="datetime64[ns]")
    modis_idx = modis["row_id"].to_numpy(dtype=int)

    viirs_lat = viirs["latitude"].to_numpy(dtype=float)
    viirs_lon = viirs["longitude"].to_numpy(dtype=float)
    viirs_t = viirs["acq_datetime"].to_numpy(dtype="datetime64[ns]")
    viirs_idx = viirs["row_id"].to_numpy(dtype=int)

    used_viirs: set[int] = set()
    max_delta = np.timedelta64(int(time_window_hours * 3600), "s")
    # ~1 km cells + 6h time bins → only compare nearby candidates (near O(n))
    cell_deg = max(distance_km / 111.0, 0.005)
    time_bin_ns = int(time_window_hours * 3600 * 1e9)

    from collections import defaultdict

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    viirs_t_int = viirs_t.astype("datetime64[ns]").astype(np.int64)
    for j in range(len(viirs_idx)):
        gx = int(np.floor(viirs_lon[j] / cell_deg))
        gy = int(np.floor(viirs_lat[j] / cell_deg))
        gt = int(viirs_t_int[j] // time_bin_ns)
        buckets[(gx, gy, gt)].append(j)

    modis_t_int = modis_t.astype("datetime64[ns]").astype(np.int64)
    for i in range(len(modis_idx)):
        gx = int(np.floor(modis_lon[i] / cell_deg))
        gy = int(np.floor(modis_lat[i] / cell_deg))
        gt = int(modis_t_int[i] // time_bin_ns)
        cand: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dt_bin in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy, gt + dt_bin), ()))
        if not cand:
            continue
        cand_local = np.fromiter(
            (j for j in cand if int(viirs_idx[j]) not in used_viirs),
            dtype=int,
        )
        if cand_local.size == 0:
            continue
        dt = np.abs(viirs_t[cand_local] - modis_t[i])
        time_ok = dt <= max_delta
        if not np.any(time_ok):
            continue
        cand_local = cand_local[time_ok]
        dists = haversine_km(
            modis_lat[i],
            modis_lon[i],
            viirs_lat[cand_local],
            viirs_lon[cand_local],
        )
        within = dists <= distance_km
        if not np.any(within):
            continue
        within_local = cand_local[within]
        within_dists = dists[within]
        best = int(within_local[np.argmin(within_dists)])
        m_row = int(modis_idx[i])
        v_row = int(viirs_idx[best])
        categories[m_row] = MATCHED
        categories[v_row] = MATCHED
        used_viirs.add(v_row)

    result = prepared.copy()
    result["category"] = categories
    return result


def summarize_categories(df: pd.DataFrame) -> dict[str, int]:
    """Toplam / matched / MODIS-only / VIIRS-only sayıları."""
    if df.empty or "category" not in df.columns:
        return {
            "total": 0,
            "matched": 0,
            "modis_only": 0,
            "viirs_only": 0,
        }
    cat = df["category"]
    return {
        "total": int(len(df)),
        "matched": int((cat == MATCHED).sum()),
        "modis_only": int((cat == MODIS_ONLY).sum()),
        "viirs_only": int((cat == VIIRS_ONLY).sum()),
    }


def activity_timeseries(
    df: pd.DataFrame,
    freq: str = "D",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Günlük (D) veya haftalık (W-MON) yangın aktivite serisi.

    Sütunlar: period, Matched, MODIS-only, VIIRS-only
    (Matched satırlar matched sayısına; only satırlar kendi kategorisine gider.)
    """
    empty = pd.DataFrame(
        columns=["period", MATCHED, MODIS_ONLY, VIIRS_ONLY]
    )
    if df.empty:
        return empty

    work = df.copy()
    if start is not None:
        work = work[work["acq_datetime"] >= pd.Timestamp(start)]
    if end is not None:
        # Günün sonuna kadar dahil et
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        work = work[work["acq_datetime"] <= end_ts]
    if work.empty:
        return empty

    work = work.set_index("acq_datetime").sort_index()
    # period etiketi
    grouped = work.groupby([pd.Grouper(freq=freq), "category"]).size()
    pivot = grouped.unstack(fill_value=0)

    for col in (MATCHED, MODIS_ONLY, VIIRS_ONLY):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[[MATCHED, MODIS_ONLY, VIIRS_ONLY]].reset_index()
    pivot = pivot.rename(columns={"acq_datetime": "period"})
    return pivot


def iter_category_frames(df: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    """Harita katmanları için kategori → alt tablo."""
    for cat in (MATCHED, MODIS_ONLY, VIIRS_ONLY):
        yield cat, df[df["category"] == cat]
