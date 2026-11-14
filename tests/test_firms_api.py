"""FIRMS API yardımcıları için birim testleri (ağ çağrısı yok)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.firms_api import (
    AVAILABLE_SENSORS,
    PRESET_REGIONS,
    FirmsApiError,
    _parse_csv_response,
    fetch_firms_area,
    fetch_sensor_area,
)


def test_preset_regions_valid_bbox():
    for name, (min_lon, min_lat, max_lon, max_lat) in PRESET_REGIONS.items():
        assert min_lon < max_lon, name
        assert min_lat < max_lat, name


def test_parse_csv_valid():
    csv_text = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        "38.4,27.1,2024-08-10,1345,85,Terra\n"
    )
    df = _parse_csv_response(csv_text)
    assert len(df) == 1
    assert float(df.iloc[0]["latitude"]) == 38.4


def test_parse_invalid_key_message():
    try:
        _parse_csv_response("Invalid MAP_KEY or something")
        assert False, "expected FirmsApiError"
    except FirmsApiError as exc:
        assert "Geçersiz MAP_KEY" in str(exc)
        assert "something" not in str(exc) or "MAP_KEY" in str(exc)


def test_parse_empty():
    df = _parse_csv_response("")
    assert df.empty


def test_fetch_requires_key():
    try:
        fetch_firms_area(
            "",
            AVAILABLE_SENSORS[:1],
            26.0,
            36.0,
            30.0,
            40.0,
            1,
        )
        assert False, "expected FirmsApiError"
    except FirmsApiError as exc:
        assert "MAP_KEY" in str(exc)


def test_fetch_sensor_mocked_success():
    csv_body = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        "38.4,27.1,2024-08-10,1345,85,Terra\n"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = csv_body

    with patch("src.firms_api.requests.get", return_value=mock_resp) as get_mock:
        df = fetch_sensor_area(
            "dummy-key-not-logged",
            "MODIS_NRT",
            26.0,
            36.0,
            30.0,
            40.0,
            2,
        )
        assert len(df) == 1
        # URL'de key vardır ama test çıktısına yazmıyoruz
        called_url = get_mock.call_args[0][0]
        assert "MODIS_NRT" in called_url
        assert "26.0,36.0,30.0,40.0" in called_url


def test_fetch_multi_sensor_concat():
    csv_modis = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        "38.4,27.1,2024-08-10,1345,85,Terra\n"
    )
    csv_viirs = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        "38.41,27.11,2024-08-10,1400,high,N20\n"
    )

    def fake_get(url, timeout=60.0):
        resp = MagicMock()
        resp.status_code = 200
        if "MODIS_NRT" in url:
            resp.text = csv_modis
        else:
            resp.text = csv_viirs
        return resp

    with patch("src.firms_api.requests.get", side_effect=fake_get):
        df = fetch_firms_area(
            "dummy-key",
            ["MODIS_NRT", "VIIRS_NOAA20_NRT"],
            26.0,
            36.0,
            30.0,
            40.0,
            1,
        )
        assert len(df) == 2
        assert set(df["firms_source"]) == {"MODIS_NRT", "VIIRS_NOAA20_NRT"}


def test_key_not_in_http_error_message():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    secret = "super-secret-map-key-xyz"
    with patch("src.firms_api.requests.get", return_value=mock_resp):
        try:
            fetch_sensor_area(secret, "MODIS_NRT", 26.0, 36.0, 30.0, 40.0, 1)
            assert False, "expected FirmsApiError"
        except FirmsApiError as exc:
            assert secret not in str(exc)


if __name__ == "__main__":
    test_preset_regions_valid_bbox()
    test_parse_csv_valid()
    test_parse_invalid_key_message()
    test_parse_empty()
    test_fetch_requires_key()
    test_fetch_sensor_mocked_success()
    test_fetch_multi_sensor_concat()
    test_key_not_in_http_error_message()
    print("all firms_api tests passed")
