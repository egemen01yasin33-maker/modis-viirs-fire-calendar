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
    elevated_activity_last_days,
    harmonization_quality,
    latest_month_changes,
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


def test_harmonization_quality_and_alert():
    df = pd.DataFrame(
        {
            "category": [MATCHED, MATCHED, MATCHED, MODIS_ONLY],
            "confidence": [90, "h", 40, 99],
        }
    )
    # 2 of 3 matched exceed 80 (90 and h); 40 does not → ~66.7%
    q = harmonization_quality(df)
    assert q is not None and 66.0 <= q <= 67.0

    daily = pd.DataFrame(
        {
            "period": pd.date_range("2025-01-01", periods=10),
            "total": [2] * 9 + [40],
            "is_anomaly": [False] * 9 + [True],
        }
    )
    assert elevated_activity_last_days(daily, days=7) is True
    assert elevated_activity_last_days(daily.iloc[:9], days=7) is False

    monthly = pd.DataFrame(
        {
            "period": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "total": [100, 150],
            "frp_sum": [1000.0, 800.0],
            "pct_change_total": [None, 50.0],
            "pct_change_frp": [None, -20.0],
        }
    )
    ch = latest_month_changes(monthly)
    assert ch["pct_change_total"] == 50.0
    assert ch["pct_change_frp"] == -20.0


if __name__ == "__main__":
    test_anomaly_flags_days_over_2x_mean()
    test_summary_sentence()
    test_compare_summaries_pct()
    test_harmonization_quality_and_alert()
    print("all analytics tests passed")
