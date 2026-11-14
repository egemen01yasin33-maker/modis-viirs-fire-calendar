"""Tests for archive CSV loading."""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.archive_csv import (
    ArchiveCsvError,
    filter_by_bbox,
    filter_by_dates,
    load_and_merge_csvs,
    load_filtered_csvs,
    peek_archive_date_bounds,
)
from src.harmonize import MATCHED, harmonize_hotspots, summarize_categories


def test_load_sample_firms_and_harmonize():
    # Isolate sample so multi-GB archive dumps in data/ are not fully loaded
    sample = ROOT / "data" / "sample_firms.csv"
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        shutil.copy(sample, folder / "sample_firms.csv")
        merged, names = load_and_merge_csvs(folder)
        assert "sample_firms.csv" in names
        assert len(merged) >= 1
        out = harmonize_hotspots(merged)
        stats = summarize_categories(out)
        assert stats["total"] == (
            stats["matched"] + stats["modis_only"] + stats["viirs_only"]
        )
        assert stats["matched"] > 0


def test_empty_folder_raises():
    with tempfile.TemporaryDirectory() as td:
        try:
            load_and_merge_csvs(Path(td))
            assert False, "expected ArchiveCsvError"
        except ArchiveCsvError as exc:
            assert "No CSV" in str(exc)


def test_filter_helpers():
    df = pd.DataFrame(
        {
            "latitude": [39.9, 41.0],
            "longitude": [32.8, 29.0],
            "acq_datetime": pd.to_datetime(
                ["2024-08-01 12:00", "2024-08-05 12:00"]
            ),
            "category": [MATCHED, MATCHED],
        }
    )
    boxed = filter_by_bbox(df, 32.0, 39.0, 33.5, 40.5)
    assert len(boxed) == 1
    dated = filter_by_dates(df, "2024-08-01", "2024-08-02")
    assert len(dated) == 1


def test_load_filtered_csvs_on_sample():
    sample = ROOT / "data" / "sample_firms.csv"
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        shutil.copy(sample, folder / "sample_firms.csv")
        # Wide bbox should keep sample rows
        df, names, stats = load_filtered_csvs(
            folder,
            min_lon=-180,
            min_lat=-90,
            max_lon=180,
            max_lat=90,
            start=date(2000, 1, 1),
            end=date(2100, 1, 1),
        )
        assert "sample_firms.csv" in names
        assert len(df) >= 1
        assert stats["rows_kept"] == len(df)

        empty, _, stats2 = load_filtered_csvs(
            folder,
            min_lon=0,
            min_lat=0,
            max_lon=1,
            max_lat=1,
            start=date(2000, 1, 1),
            end=date(2000, 1, 2),
        )
        assert empty.empty
        assert stats2["rows_kept"] == 0


def test_peek_dates_on_sample():
    sample = ROOT / "data" / "sample_firms.csv"
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        shutil.copy(sample, folder / "sample_firms.csv")
        lo, hi = peek_archive_date_bounds(folder)
        assert lo is not None and hi is not None
        assert lo <= hi


if __name__ == "__main__":
    test_load_sample_firms_and_harmonize()
    test_empty_folder_raises()
    test_filter_helpers()
    test_load_filtered_csvs_on_sample()
    test_peek_dates_on_sample()
    print("all archive_csv tests passed")
