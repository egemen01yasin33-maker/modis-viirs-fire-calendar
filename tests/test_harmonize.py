"""Basit regresyon testleri — örnek FIRMS verisi üzerinde harmonizasyon."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harmonize import (
    MATCHED,
    MODIS_ONLY,
    VIIRS_ONLY,
    activity_timeseries,
    harmonize_hotspots,
    summarize_categories,
)

SAMPLE = ROOT / "data" / "sample_firms.csv"


def test_sample_harmonization_counts():
    df = pd.read_csv(SAMPLE)
    out = harmonize_hotspots(df)
    stats = summarize_categories(out)
    assert stats["total"] == 26
    assert stats["matched"] == 14
    assert stats["modis_only"] == 6
    assert stats["viirs_only"] == 6


def test_matched_pairs_are_cross_sensor():
    out = harmonize_hotspots(pd.read_csv(SAMPLE))
    matched = out[out["category"] == MATCHED]
    assert set(matched["sensor"]) == {"MODIS", "VIIRS"}
    assert (matched["sensor"] == "MODIS").sum() == (matched["sensor"] == "VIIRS").sum()


def test_activity_timeseries_daily():
    out = harmonize_hotspots(pd.read_csv(SAMPLE))
    ts = activity_timeseries(out, freq="D")
    assert not ts.empty
    assert set(ts.columns) >= {MATCHED, MODIS_ONLY, VIIRS_ONLY, "period"}
    assert int(ts[MATCHED].sum()) == 14


if __name__ == "__main__":
    test_sample_harmonization_counts()
    test_matched_pairs_are_cross_sensor()
    test_activity_timeseries_daily()
    print("all tests passed")
