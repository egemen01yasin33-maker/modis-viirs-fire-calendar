"""Tests for archive CSV loading."""

from __future__ import annotations

import sys
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
)
from src.harmonize import MATCHED, harmonize_hotspots, summarize_categories


def test_load_sample_firms_and_harmonize():
    folder = ROOT / "data"
    merged, names = load_and_merge_csvs(folder)
    assert "sample_firms.csv" in names
    assert len(merged) >= 1
    out = harmonize_hotspots(merged)
    stats = summarize_categories(out)
    assert stats["total"] == stats["matched"] + stats["modis_only"] + stats["viirs_only"]
    assert stats["matched"] > 0


def test_empty_folder_raises(tmp_path: Path | None = None):
    empty = ROOT / "data" / "_empty_test_dir_should_not_exist"
    # use a fresh temp dir via pathlib next to repo
    import tempfile

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


if __name__ == "__main__":
    test_load_sample_firms_and_harmonize()
    test_empty_folder_raises()
    test_filter_helpers()
    print("all archive_csv tests passed")
