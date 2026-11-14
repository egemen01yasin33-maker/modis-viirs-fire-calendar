"""
NASA FIRMS Area API client.

MAP_KEY is never logged or written to disk; error messages never include
the key or a URL that contains it.
"""

from __future__ import annotations

import io
import re
import unicodedata
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import requests

FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Preset regions: (min_lon, min_lat, max_lon, max_lat)
PRESET_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Turkey": (25.5, 35.8, 45.0, 42.3),
    "Mediterranean": (27.5, 36.0, 37.5, 38.5),
    "Aegean": (26.0, 36.5, 30.0, 40.0),
    "Central Anatolia": (31.0, 37.5, 36.0, 41.0),
    "Marmara": (26.0, 39.5, 30.5, 41.8),
}

# Fixed city / place bboxes for custom-region text lookup
CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "ankara": (32.4, 39.6, 33.2, 40.2),
    "istanbul": (28.5, 40.8, 29.5, 41.3),
    "izmir": (26.9, 38.2, 27.4, 38.6),
    "antalya": (30.4, 36.7, 31.0, 37.1),
    "bursa": (28.9, 40.1, 29.3, 40.3),
    "adana": (35.1, 36.9, 35.5, 37.2),
    "gaziantep": (37.2, 36.9, 37.5, 37.2),
    "konya": (32.3, 37.7, 32.7, 38.1),
    "mersin": (34.4, 36.7, 34.8, 37.0),
    "trabzon": (39.5, 40.9, 39.9, 41.1),
    "mugla": (28.0, 36.8, 29.0, 37.5),
    "canakkale": (26.2, 39.9, 26.6, 40.3),
    "eskisehir": (30.3, 39.6, 30.7, 39.9),
    "diyarbakir": (40.0, 37.8, 40.4, 38.1),
    "van": (43.2, 38.4, 43.5, 38.7),
    "samsun": (36.1, 41.2, 36.5, 41.4),
    "antalya center": (30.5, 36.8, 30.9, 37.0),
}

AVAILABLE_SENSORS = (
    "MODIS_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
)

# NRT Area API allows at most 10 days ending on a given date.
# Exported for app.py date-range UI captions / validation.
MAX_DAY_RANGE: int = 10

__all__ = (
    "AVAILABLE_SENSORS",
    "CITY_BBOXES",
    "FIRMS_AREA_BASE",
    "MAX_DAY_RANGE",
    "PRESET_REGIONS",
    "FirmsApiError",
    "fetch_firms_area",
    "fetch_sensor_area",
    "resolve_place_bbox",
)

_INVALID_KEY_MARKERS = (
    "invalid map_key",
    "invalid map key",
    "map_key is invalid",
    "unauthorized",
    "not authorized",
    "incorrect map key",
)


class FirmsApiError(Exception):
    """User-facing FIRMS API error (never contains the key)."""


def _redact_key(text: str, map_key: str) -> str:
    """Strip MAP_KEY leakage from text (defense in depth)."""
    if not map_key:
        return text
    return text.replace(map_key, "[REDACTED]")


def _normalize_place(name: str) -> str:
    """Lowercase ASCII-ish key for city lookup (İstanbul → istanbul)."""
    text = (name or "").strip().lower()
    # Turkish İ/I handling before NFKD
    text = (
        text.replace("ı", "i")
        .replace("İ", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_place_bbox(place_name: str) -> tuple[float, float, float, float]:
    """
    Resolve a city / place name to a fixed bbox.

    Raises FirmsApiError if the place is unknown.
    """
    key = _normalize_place(place_name)
    if not key:
        raise FirmsApiError("Enter a city or region name (for example Ankara).")

    if key in CITY_BBOXES:
        return CITY_BBOXES[key]

    # Partial match: "ankara merkez" → ankara
    for city, bbox in CITY_BBOXES.items():
        if city in key or key in city:
            return bbox

    known = ", ".join(sorted({c.title() for c in CITY_BBOXES}))
    raise FirmsApiError(
        f"Unknown place “{place_name.strip()}”. "
        f"Try a preset region or one of: {known}."
    )


def _build_url(
    map_key: str,
    sensor: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    days: int,
    end_date: date,
) -> str:
    """
    Area CSV URL ending on end_date for `days` days.

    https://firms.modaps.eosdis.nasa.gov/api/area/csv/
      {MAP_KEY}/{SENSOR}/{west},{south},{east},{north}/{day_range}/{YYYY-MM-DD}
    """
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    return (
        f"{FIRMS_AREA_BASE}/{map_key}/{sensor}/{bbox}/"
        f"{days}/{end_date.isoformat()}"
    )


def _looks_like_invalid_key(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _INVALID_KEY_MARKERS)


def _parse_csv_response(text: str) -> pd.DataFrame:
    """Parse FIRMS CSV text into a DataFrame; empty body → empty table."""
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame()

    if stripped.lower().startswith("<!doctype") or stripped.lower().startswith(
        "<html"
    ):
        raise FirmsApiError(
            "FIRMS returned unexpected HTML. "
            "Check your MAP_KEY and request parameters."
        )

    if _looks_like_invalid_key(stripped):
        raise FirmsApiError(
            "Invalid MAP_KEY. Get a valid key from your NASA FIRMS account."
        )

    first_line = stripped.splitlines()[0]
    if "latitude" not in first_line.lower() and "," not in first_line:
        message = first_line.strip()[:200] or "Unknown FIRMS error."
        if _looks_like_invalid_key(message):
            raise FirmsApiError(
                "Invalid MAP_KEY. Get a valid key from your NASA FIRMS account."
            )
        raise FirmsApiError(f"FIRMS API error: {message}")

    try:
        df = pd.read_csv(io.StringIO(stripped))
    except Exception as exc:  # noqa: BLE001
        raise FirmsApiError(
            "Could not parse the FIRMS response as CSV. "
            "Check your MAP_KEY and region parameters."
        ) from exc

    return df


def fetch_sensor_area(
    map_key: str,
    sensor: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    days: int,
    end_date: date,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Download Area API CSV for a single sensor ending on end_date."""
    if sensor not in AVAILABLE_SENSORS:
        raise FirmsApiError(f"Unsupported sensor: {sensor}")

    url = _build_url(
        map_key, sensor, min_lon, min_lat, max_lon, max_lat, days, end_date
    )

    try:
        response = requests.get(url, timeout=timeout)
    except requests.Timeout as exc:
        raise FirmsApiError(
            "FIRMS API timed out. Check your connection or try a smaller "
            "region / shorter date range."
        ) from exc
    except requests.ConnectionError as exc:
        raise FirmsApiError(
            "Could not connect to the FIRMS API. Check your internet connection."
        ) from exc
    except requests.RequestException as exc:
        safe = _redact_key(str(exc), map_key)
        raise FirmsApiError(f"FIRMS connection error: {safe}") from exc

    if response.status_code in (401, 403):
        raise FirmsApiError(
            "Invalid or unauthorized MAP_KEY. "
            "Get a new key at https://firms.modaps.eosdis.nasa.gov/api/."
        )

    if response.status_code == 404:
        raise FirmsApiError(
            "FIRMS endpoint not found (404). Check the sensor name and bbox."
        )

    if response.status_code >= 400:
        raise FirmsApiError(
            f"FIRMS API returned HTTP {response.status_code}. "
            "Check your request parameters and try again."
        )

    return _parse_csv_response(response.text)


def fetch_firms_area(
    map_key: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: date,
    end_date: date,
    sensors: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch and concatenate Area API results for the date range.

    NRT sources support at most 10 days ending on `end_date`. Sensors default
    to MODIS_NRT + VIIRS_NOAA20_NRT + VIIRS_NOAA21_NRT.
    """
    key = (map_key or "").strip()
    if not key:
        raise FirmsApiError(
            "MAP_KEY is required. Add FIRMS_MAP_KEY to your .env file."
        )

    sensor_list = [s for s in (sensors or AVAILABLE_SENSORS) if s]
    if not sensor_list:
        raise FirmsApiError("No sensors configured.")

    if not (min_lon < max_lon and min_lat < max_lat):
        raise FirmsApiError(
            "Invalid bbox: require min_lon < max_lon and min_lat < max_lat."
        )

    if end_date < start_date:
        raise FirmsApiError("End date must be on or after the start date.")

    days = (end_date - start_date).days + 1
    if days < 1 or days > MAX_DAY_RANGE:
        raise FirmsApiError(
            f"Date range must be between 1 and {MAX_DAY_RANGE} days "
            f"(FIRMS NRT limit). You selected {days} days."
        )

    # Guard against requesting far-future / very old NRT windows
    today = date.today()
    if end_date > today:
        raise FirmsApiError("End date cannot be in the future.")
    if start_date < today - timedelta(days=365):
        raise FirmsApiError(
            "Start date is too far in the past for NRT sources. "
            "Use a more recent window."
        )

    frames: list[pd.DataFrame] = []
    for sensor in sensor_list:
        df = fetch_sensor_area(
            key, sensor, min_lon, min_lat, max_lon, max_lat, days, end_date
        )
        if df is not None and not df.empty:
            df = df.copy()
            df["firms_source"] = sensor
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Keep only rows inside the requested calendar window when acq_date exists
    if "acq_date" in combined.columns:
        acq = pd.to_datetime(combined["acq_date"], errors="coerce").dt.date
        mask = (acq >= start_date) & (acq <= end_date)
        combined = combined.loc[mask | acq.isna()].reset_index(drop=True)

    return combined
