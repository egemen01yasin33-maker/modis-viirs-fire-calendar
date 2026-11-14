"""
NASA FIRMS Area API istemcisi.

MAP_KEY hiçbir zaman loglanmaz veya diske yazılmaz; hata mesajlarında
anahtar veya anahtar içeren URL gösterilmez.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import requests

FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Hazır bölgeler: (min_lon, min_lat, max_lon, max_lat)
PRESET_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Türkiye": (25.5, 35.8, 45.0, 42.3),
    "Akdeniz Bölgesi": (27.5, 36.0, 37.5, 38.5),
    "Ege": (26.0, 36.5, 30.0, 40.0),
    "İç Anadolu": (31.0, 37.5, 36.0, 41.0),
}

AVAILABLE_SENSORS = (
    "MODIS_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
)

# FIRMS sık dönen hata ifadeleri (yanıt gövdesinde aranır)
_INVALID_KEY_MARKERS = (
    "invalid map_key",
    "invalid map key",
    "map_key is invalid",
    "unauthorized",
    "not authorized",
    "incorrect map key",
)


class FirmsApiError(Exception):
    """Kullanıcıya gösterilebilir FIRMS API hatası (anahtar içermez)."""


def _redact_key(text: str, map_key: str) -> str:
    """Metinden MAP_KEY sızıntısını temizler (savunma amaçlı)."""
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
    """FIRMS CSV metnini DataFrame'e çevirir; veri yoksa boş tablo."""
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame()

    # HTML / düz metin hata yanıtları
    if stripped.lower().startswith("<!doctype") or stripped.lower().startswith(
        "<html"
    ):
        raise FirmsApiError(
            "FIRMS API beklenmeyen bir HTML yanıtı döndü. "
            "MAP_KEY veya istek parametrelerini kontrol edin."
        )

    if _looks_like_invalid_key(stripped):
        raise FirmsApiError(
            "Geçersiz MAP_KEY. NASA FIRMS hesabınızdan geçerli bir anahtar alın."
        )

    # Tek satırlık hata mesajı (CSV başlığı değil)
    first_line = stripped.splitlines()[0]
    if "latitude" not in first_line.lower() and "," not in first_line:
        # Anahtar içermeyen kısa hata metni
        message = first_line.strip()[:200] or "Bilinmeyen FIRMS hatası."
        if _looks_like_invalid_key(message):
            raise FirmsApiError(
                "Geçersiz MAP_KEY. NASA FIRMS hesabınızdan geçerli bir anahtar alın."
            )
        raise FirmsApiError(f"FIRMS API hatası: {message}")

    try:
        df = pd.read_csv(io.StringIO(stripped))
    except Exception as exc:  # noqa: BLE001
        raise FirmsApiError(
            "FIRMS yanıtı CSV olarak okunamadı. "
            "MAP_KEY ve bölge parametrelerini kontrol edin."
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
    """
    Tek bir sensör için Area API CSV'sini indirir.

    Raises:
        FirmsApiError: geçersiz key, bağlantı veya parse hataları.
    """
    if sensor not in AVAILABLE_SENSORS:
        raise FirmsApiError(f"Desteklenmeyen sensör: {sensor}")

    url = _build_url(map_key, sensor, min_lon, min_lat, max_lon, max_lat, days)

    try:
        response = requests.get(url, timeout=timeout)
    except requests.Timeout as exc:
        raise FirmsApiError(
            "FIRMS API zaman aşımına uğradı. İnternet bağlantınızı deneyin "
            "veya daha küçük bir bölge / daha az gün seçin."
        ) from exc
    except requests.ConnectionError as exc:
        raise FirmsApiError(
            "FIRMS API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin."
        ) from exc
    except requests.RequestException as exc:
        # Anahtar URL'de olabilir; mesajı süz
        safe = _redact_key(str(exc), map_key)
        raise FirmsApiError(f"FIRMS bağlantı hatası: {safe}") from exc

    if response.status_code in (401, 403):
        raise FirmsApiError(
            "Geçersiz veya yetkisiz MAP_KEY. "
            "https://firms.modaps.eosdis.nasa.gov/api/ adresinden yeni anahtar alın."
        )

    if response.status_code == 404:
        raise FirmsApiError(
            "FIRMS uç noktası bulunamadı (404). Sensör adını ve bbox değerlerini kontrol edin."
        )

    if response.status_code >= 400:
        raise FirmsApiError(
            f"FIRMS API HTTP {response.status_code} döndü. "
            "İstek parametrelerini kontrol edip yeniden deneyin."
        )

    # Yanıt gövdesinde anahtar aranmaz; yalnızca içerik parse edilir
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
    """
    Seçilen sensörler için Area API çağrılarını birleştirir.

    Boş sonuçlarda boş DataFrame döner (exception değil); tamamen başarısız
    istekler FirmsApiError fırlatır.
    """
    key = (map_key or "").strip()
    if not key:
        raise FirmsApiError("MAP_KEY gerekli. Kenar çubuğuna FIRMS anahtarınızı girin.")

    sensor_list = [s for s in sensors if s]
    if not sensor_list:
        raise FirmsApiError("En az bir sensör seçin.")

    if not (min_lon < max_lon and min_lat < max_lat):
        raise FirmsApiError(
            "Geçersiz bbox: min_lon < max_lon ve min_lat < max_lat olmalı."
        )

    if days < 1 or days > 10:
        raise FirmsApiError("Gün sayısı 1 ile 10 arasında olmalıdır.")

    frames: list[pd.DataFrame] = []
    for sensor in sensor_list:
        df = fetch_sensor_area(
            key, sensor, min_lon, min_lat, max_lon, max_lat, days
        )
        if df is not None and not df.empty:
            # Kaynak sensör etiketini koru (harmonizasyon instrument/satellite kullanır)
            df = df.copy()
            df["firms_source"] = sensor
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
