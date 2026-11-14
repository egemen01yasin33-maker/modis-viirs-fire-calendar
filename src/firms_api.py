"""
NASA FIRMS Area API client.

MAP_KEY is never logged or written to disk; error messages never include
the key or a URL that contains it.
"""

from __future__ import annotations

import io
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
}

AVAILABLE_SENSORS = (
    "MODIS_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
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


def _build_url(
    map_key: str,
    sensor: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    days: int,
) -> str:
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    return f"{FIRMS_AREA_BASE}/{map_key}/{sensor}/{bbox}/{days}"


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
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Download Area API CSV for a single sensor."""
    if sensor not in AVAILABLE_SENSORS:
        raise FirmsApiError(f"Unsupported sensor: {sensor}")

    url = _build_url(map_key, sensor, min_lon, min_lat, max_lon, max_lat, days)

    try:
        response = requests.get(url, timeout=timeout)
    except requests.Timeout as exc:
        raise FirmsApiError(
            "FIRMS API timed out. Check your connection or try a smaller "
            "region / fewer days."
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
    sensors: Iterable[str],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    days: int,
) -> pd.DataFrame:
    """Fetch and concatenate Area API results for the selected sensors."""
    key = (map_key or "").strip()
    if not key:
        raise FirmsApiError(
            "MAP_KEY is required. Add FIRMS_MAP_KEY to your .env file."
        )

    sensor_list = [s for s in sensors if s]
    if not sensor_list:
        raise FirmsApiError("Select at least one sensor.")

    if not (min_lon < max_lon and min_lat < max_lat):
        raise FirmsApiError(
            "Invalid bbox: require min_lon < max_lon and min_lat < max_lat."
        )

    if days < 1 or days > 10:
        raise FirmsApiError("Day count must be between 1 and 10.")

    frames: list[pd.DataFrame] = []
    for sensor in sensor_list:
        df = fetch_sensor_area(
            key, sensor, min_lon, min_lat, max_lon, max_lat, days
        )
        if df is not None and not df.empty:
            df = df.copy()
            df["firms_source"] = sensor
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
