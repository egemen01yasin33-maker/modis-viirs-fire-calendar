"""
Activity analytics: daily totals, anomaly flags, period comparison.
"""

from __future__ import annotations

import pandas as pd

from src.harmonize import MATCHED, summarize_categories

ANOMALY_COLOR = "#ff4d6d"


def daily_hotspot_counts(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Daily total hotspot counts.

    Returns columns: period (date), total, is_anomaly
    """
    empty = pd.DataFrame(columns=["period", "total", "is_anomaly"])
    if df.empty or "acq_datetime" not in df.columns:
        return empty

    work = df.copy()
    if start is not None:
        work = work[work["acq_datetime"] >= pd.Timestamp(start)]
    if end is not None:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        work = work[work["acq_datetime"] <= end_ts]
    if work.empty:
        return empty

    daily = (
        work.set_index("acq_datetime")
        .groupby(pd.Grouper(freq="D"))
        .size()
        .rename("total")
        .reset_index()
        .rename(columns={"acq_datetime": "period"})
    )

    # Fill small gaps only — never invent multi-year zero series (freezes UI)
    if start is not None and end is not None and not daily.empty:
        req_start = pd.Timestamp(start).normalize()
        req_end = pd.Timestamp(end).normalize()
        obs_start = pd.Timestamp(daily["period"].min()).normalize()
        obs_end = pd.Timestamp(daily["period"].max()).normalize()
        fill_start = max(req_start, obs_start)
        fill_end = min(req_end, obs_end)
        span_days = (fill_end - fill_start).days + 1
        if 1 < span_days <= 120:
            full_idx = pd.date_range(start=fill_start, end=fill_end, freq="D")
            daily = (
                daily.set_index("period")
                .reindex(full_idx, fill_value=0)
                .rename_axis("period")
                .reset_index()
            )

    # Mean from days that actually have fires (zeros from gaps don't dilute)
    positive = daily.loc[daily["total"] > 0, "total"]
    mean_total = float(positive.mean()) if len(positive) else 0.0
    threshold = mean_total * 2.0
    daily["is_anomaly"] = (daily["total"] > threshold) & (mean_total > 0)
    daily["mean_total"] = mean_total
    daily["threshold"] = threshold
    return daily


def count_anomalies(daily: pd.DataFrame) -> int:
    """Number of anomaly days in a daily totals frame."""
    if daily.empty or "is_anomaly" not in daily.columns:
        return 0
    return int(daily["is_anomaly"].sum())


def pct_change(current: float, baseline: float) -> float | None:
    """Percent change from baseline → current; None if baseline is 0."""
    if baseline == 0:
        return None if current == 0 else float("inf")
    return ((current - baseline) / baseline) * 100.0


def compare_summaries(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
) -> dict[str, dict[str, float | int | None]]:
    """
    Side-by-side Total / Matched / MODIS-only / VIIRS-only with % diff
    (primary relative to secondary / comparison period).
    """
    a = summarize_categories(primary)
    b = summarize_categories(secondary)
    keys = ("total", "matched", "modis_only", "viirs_only")
    out: dict[str, dict[str, float | int | None]] = {}
    for key in keys:
        av = int(a[key])
        bv = int(b[key])
        out[key] = {
            "primary": av,
            "comparison": bv,
            "delta": av - bv,
            "pct": pct_change(av, bv),
        }
    return out


def summary_sentence(
    total: int,
    matched: int,
    anomalies: int,
) -> str:
    """One-line period summary for the main panel."""
    return (
        f"Selected period: {total} total hotspots, "
        f"{matched} matched (high confidence), "
        f"{anomalies} anomalies detected."
    )
