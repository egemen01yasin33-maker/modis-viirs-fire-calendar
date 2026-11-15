"""
Activity analytics: daily totals, anomaly flags, FRP helpers, period comparison.
"""

from __future__ import annotations

import pandas as pd

from src.harmonize import MATCHED, summarize_categories

ANOMALY_COLOR = "#ff4d6d"

# VIIRS letter codes → numeric stand-ins for threshold checks
_CONFIDENCE_LETTER = {
    "h": 100.0,
    "high": 100.0,
    "n": 50.0,
    "nominal": 50.0,
    "l": 20.0,
    "low": 20.0,
}


def confidence_to_score(value) -> float | None:
    """Parse FIRMS confidence (0–100 or VIIRS letter) to a 0–100 score."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    if text in _CONFIDENCE_LETTER:
        return _CONFIDENCE_LETTER[text]
    try:
        return float(text)
    except ValueError:
        return None


def harmonization_quality(
    df: pd.DataFrame,
    *,
    threshold: float = 80.0,
) -> float | None:
    """
    Share of Matched points with confidence > threshold, as 0–100 percent.

    Returns None when there are no Matched rows to score.
    """
    if df is None or df.empty or "category" not in df.columns:
        return None
    matched = df[df["category"] == MATCHED]
    if matched.empty or "confidence" not in matched.columns:
        return None
    scores = matched["confidence"].map(confidence_to_score)
    valid = scores.dropna()
    if valid.empty:
        return None
    return float((valid > threshold).mean() * 100.0)


def elevated_activity_last_days(
    daily: pd.DataFrame,
    *,
    days: int = 7,
) -> bool:
    """True when any anomaly day falls in the last `days` of the series."""
    if daily is None or daily.empty or "is_anomaly" not in daily.columns:
        return False
    work = daily.copy()
    work["period"] = pd.to_datetime(work["period"])
    end = work["period"].max()
    window = work[work["period"] >= end - pd.Timedelta(days=days - 1)]
    return bool(window["is_anomaly"].any())


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


def latest_month_changes(monthly: pd.DataFrame) -> dict[str, float | None]:
    """Percent change for the newest month vs the previous one."""
    if monthly is None or len(monthly) < 2:
        return {"pct_change_total": None, "pct_change_frp": None}
    last = monthly.iloc[-1]
    return {
        "pct_change_total": (
            float(last["pct_change_total"])
            if pd.notna(last.get("pct_change_total"))
            else None
        ),
        "pct_change_frp": (
            float(last["pct_change_frp"])
            if pd.notna(last.get("pct_change_frp"))
            else None
        ),
    }
