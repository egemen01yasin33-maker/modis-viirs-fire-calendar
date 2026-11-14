"""Unit tests for FIRMS API helpers (no network)."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    resolve_place_bbox,
)


def test_preset_regions_valid_bbox():
    assert "Marmara" in PRESET_REGIONS
    for name, (min_lon, min_lat, max_lon, max_lat) in PRESET_REGIONS.items():
        assert min_lon < max_lon, name
        assert min_lat < max_lat, name


def test_resolve_place_bbox_ankara():
    bbox = resolve_place_bbox("Ankara")
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_resolve_place_bbox_turkish_chars():
    bbox = resolve_place_bbox("İstanbul")
    assert bbox == resolve_place_bbox("istanbul")


def test_resolve_place_unknown():
    try:
        resolve_place_bbox("Atlantis")
        assert False, "expected FirmsApiError"
    except FirmsApiError as exc:
        assert "Unknown place" in str(exc)


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
        assert "Invalid MAP_KEY" in str(exc)


def test_parse_empty():
    df = _parse_csv_response("")
    assert df.empty


def test_fetch_requires_key():
    today = date.today()
    try:
        fetch_firms_area(
            "",
            26.0,
            36.0,
            30.0,
            40.0,
            today - timedelta(days=1),
            today,
        )
        assert False, "expected FirmsApiError"
    except FirmsApiError as exc:
        assert "MAP_KEY" in str(exc)


def test_fetch_rejects_long_range():
    today = date.today()
    try:
        fetch_firms_area(
            "dummy-key",
            26.0,
            36.0,
            30.0,
            40.0,
            today - timedelta(days=15),
            today,
        )
        assert False, "expected FirmsApiError"
    except FirmsApiError as exc:
        assert "10" in str(exc)


def test_fetch_sensor_mocked_success():
    csv_body = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        "38.4,27.1,2024-08-10,1345,85,Terra\n"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = csv_body
    end = date(2024, 8, 10)

    with patch("src.firms_api.requests.get", return_value=mock_resp) as get_mock:
        df = fetch_sensor_area(
            "dummy-key-not-logged",
            "MODIS_NRT",
            26.0,
            36.0,
            30.0,
            40.0,
            2,
            end,
        )
        assert len(df) == 1
        called_url = get_mock.call_args[0][0]
        assert "MODIS_NRT" in called_url
        assert "26.0,36.0,30.0,40.0" in called_url
        assert "2024-08-10" in called_url


def test_fetch_multi_sensor_concat_defaults():
    end = date.today()
    start = end
    day = end.isoformat()
    csv_modis = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        f"38.4,27.1,{day},1345,85,Terra\n"
    )
    csv_viirs = (
        "latitude,longitude,acq_date,acq_time,confidence,satellite\n"
        f"38.41,27.11,{day},1400,high,N20\n"
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
            26.0,
            36.0,
            30.0,
            40.0,
            start,
            end,
        )
        assert set(df["firms_source"]) == set(AVAILABLE_SENSORS)
        assert len(df) == 3


def test_key_not_in_http_error_message():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    secret = "super-secret-map-key-xyz"

    with patch("src.firms_api.requests.get", return_value=mock_resp):
        try:
            fetch_sensor_area(
                secret,
                "MODIS_NRT",
                26.0,
                36.0,
                30.0,
                40.0,
                1,
                date.today(),
            )
            assert False, "expected FirmsApiError"
        except FirmsApiError as exc:
            assert secret not in str(exc)


if __name__ == "__main__":
    test_preset_regions_valid_bbox()
    test_resolve_place_bbox_ankara()
    test_resolve_place_bbox_turkish_chars()
    test_resolve_place_unknown()
    test_parse_csv_valid()
    test_parse_invalid_key_message()
    test_parse_empty()
    test_fetch_requires_key()
    test_fetch_rejects_long_range()
    test_fetch_sensor_mocked_success()
    test_fetch_multi_sensor_concat_defaults()
    test_key_not_in_http_error_message()
    print("all firms_api tests passed")
