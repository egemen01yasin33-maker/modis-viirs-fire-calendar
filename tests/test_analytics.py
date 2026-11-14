"""Unit tests for anomaly detection and comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import (
    compare_summaries,
    count_anomalies,
    daily_hotspot_counts,
    summary_sentence,
)
from src.harmonize import MATCHED, MODIS_ONLY, VIIRS_ONLY


def _frame(days_counts: list[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for day, n in days_counts:
        for i in range(n):
            rows.append(
                {
                    "acq_datetime": pd.Timestamp(f"{day} 12:00:00")
                    + pd.Timedelta(minutes=i),
                    "category": MATCHED if i % 2 == 0 else MODIS_ONLY,
                }
            )
    return pd.DataFrame(rows)


def test_anomaly_flags_days_over_2x_mean():
    # Means: (2+2+2+10)/4 = 4 → threshold 8 → only day4 is anomaly
    df = _frame(
        [
            ("2024-08-01", 2),
            ("2024-08-02", 2),
            ("2024-08-03", 2),
            ("2024-08-04", 10),
        ]
    )
    daily = daily_hotspot_counts(
        df,
        start=pd.Timestamp("2024-08-01"),
        end=pd.Timestamp("2024-08-04"),
    )
    assert count_anomalies(daily) == 1
    flagged = daily.loc[daily["is_anomaly"], "period"].dt.strftime("%Y-%m-%d")
    assert list(flagged) == ["2024-08-04"]


def test_summary_sentence():
    text = summary_sentence(100, 40, 2)
    assert text == (
        "Selected period: 100 total hotspots, "
        "40 matched (high confidence), "
        "2 anomalies detected."
    )


def test_compare_summaries_pct():
    primary = pd.DataFrame(
        {
            "category": [MATCHED, MATCHED, MODIS_ONLY, VIIRS_ONLY],
        }
    )
    comparison = pd.DataFrame(
        {
            "category": [MATCHED, MODIS_ONLY],
        }
    )
    out = compare_summaries(primary, comparison)
    assert out["total"]["primary"] == 4
    assert out["total"]["comparison"] == 2
    assert out["total"]["pct"] == 100.0


if __name__ == "__main__":
    test_anomaly_flags_days_over_2x_mean()
    test_summary_sentence()
    test_compare_summaries_pct()
    print("all analytics tests passed")
