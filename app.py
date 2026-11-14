"""
Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.
Supports FIRMS Archive CSVs in data/ and live NRT Area API fetches.

FIRMS_MAP_KEY is loaded from .env for NRT mode only.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.analytics import (
    ANOMALY_COLOR,
    compare_summaries,
    count_anomalies,
    daily_hotspot_counts,
)
from src.archive_csv import (
    ArchiveCsvError,
    data_dir,
    describe_csv_files,
    ensure_hotspots_db,
    list_archive_csv_files,
    load_daily_activity,
    load_filtered_csvs,
    peek_archive_date_bounds,
    save_uploaded_csvs,
)
from src.firms_api import (
    AVAILABLE_SENSORS,
    PRESET_REGIONS,
    FirmsApiError,
    fetch_firms_area,
    resolve_place_bbox,
)

try:
    from src.firms_api import MAX_DAY_RANGE
except ImportError:  # pragma: no cover
    MAX_DAY_RANGE = 10

from src.harmonize import (
    CATEGORY_COLORS,
    MATCHED,
    MODIS_ONLY,
    VIIRS_ONLY,
    activity_timeseries,
    harmonize_hotspots,
    summarize_categories,
    validate_firms_columns,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_PATH = data_dir(ROOT)

st.set_page_config(
    page_title="Unified Fire Activity Calendar",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="stSidebarCollapsedControl"] { display: none !important; }
      section.main .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_firms_map_key() -> str:
    """Read FIRMS_MAP_KEY from the environment (value is never logged)."""
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def _map_zoom_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> int:
    span = max(abs(max_lon - min_lon), abs(max_lat - min_lat))
    if span >= 100:
        return 2
    if span >= 40:
        return 3
    if span >= 15:
        return 5
    if span >= 5:
        return 6
    return 7


def _is_wide_bbox(
    bbox: tuple[float, float, float, float] | None,
) -> bool:
    """True for near-global boxes where tile maps wrap / duplicate continents."""
    if bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    return (max_lon - min_lon) >= 120 or (max_lat - min_lat) >= 100


MAX_MAP_POINTS = 450
MAX_CHART_ANOMALIES = 15
MAX_CHART_POINTS = 180
# Skip expensive MODIS↔VIIRS matching above this map-sample size
MATCH_ROW_LIMIT = 8_000


def _sample_for_map(df: pd.DataFrame, limit: int = MAX_MAP_POINTS) -> pd.DataFrame:
    """Keep the map responsive: prefer Matched points, then sample the rest."""
    if len(df) <= limit:
        return df
    matched = df[df["category"] == MATCHED]
    others = df[df["category"] != MATCHED]
    keep_matched = min(len(matched), max(limit // 3, 250))
    keep_others = limit - keep_matched
    parts = []
    if keep_matched and not matched.empty:
        parts.append(matched.sample(n=keep_matched, random_state=42))
    if keep_others and not others.empty:
        n = min(keep_others, len(others))
        parts.append(others.sample(n=n, random_state=42))
    if not parts:
        return df.head(limit)
    return pd.concat(parts, ignore_index=True)


def build_map_figure(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
) -> go.Figure:
    """
    Map figure without wrapped/duplicated continents.

    Wide / Global views use scatter_geo (single world). Regional views use
    OpenStreetMap tiles with explicit bounds.
    """
    plot_df = _sample_for_map(df)
    color_map = {k: v for k, v in CATEGORY_COLORS.items()}
    cat_order = list(CATEGORY_COLORS.keys())

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            geo=dict(
                projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#1a1a1a",
                showocean=True,
                oceancolor="#0b1220",
                showcountries=True,
                countrycolor="#333333",
            ),
            height=520,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Normalize longitudes into [-180, 180] to avoid antimeridian artifacts
    plot_df = plot_df.copy()
    plot_df["longitude"] = ((plot_df["longitude"] + 180.0) % 360.0) - 180.0
    plot_df["latitude"] = plot_df["latitude"].clip(-90, 90)

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon = float(plot_df["longitude"].min())
        max_lon = float(plot_df["longitude"].max())
        min_lat = float(plot_df["latitude"].min())
        max_lat = float(plot_df["latitude"].max())

    # —— Global / very wide: one world, no OSM tile wrap ——
    if _is_wide_bbox((min_lon, min_lat, max_lon, max_lat)):
        fig = px.scatter_geo(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="category",
            color_discrete_map=color_map,
            category_orders={"category": cat_order},
            height=520,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        fig.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="#1c1c1c",
            showocean=True,
            oceancolor="#0a1520",
            showlakes=False,
            showcountries=True,
            countrycolor="#3a3a3a",
            showcoastlines=True,
            coastlinecolor="#555555",
            bgcolor="rgba(0,0,0,0)",
            # Fit to data / bbox without repeating the globe
            lataxis_range=[max(-85.0, min_lat - 2), min(85.0, max_lat + 2)],
            lonaxis_range=[max(-180.0, min_lon), min(180.0, max_lon)],
            visible=True,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.01),
            paper_bgcolor="rgba(0,0,0,0)",
            geo_bgcolor="rgba(0,0,0,0)",
            uirevision="fire-map-geo",
        )
        return fig

    # —— Regional: OSM tiles with locked bounds (no horizontal wrap) ——
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    zoom = float(_map_zoom_for_bbox(min_lon, min_lat, max_lon, max_lat))
    # Pad bounds slightly so points near edges stay visible
    pad_lon = max((max_lon - min_lon) * 0.05, 0.3)
    pad_lat = max((max_lat - min_lat) * 0.05, 0.3)
    west = max(-180.0, min_lon - pad_lon)
    east = min(180.0, max_lon + pad_lon)
    south = max(-85.0, min_lat - pad_lat)
    north = min(85.0, max_lat + pad_lat)

    fig = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="category",
        color_discrete_map=color_map,
        category_orders={"category": cat_order},
        zoom=zoom,
        height=520,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
            # Explicit bounds stop the repeating-world tile pan
            bounds=dict(west=west, east=east, south=south, north=north),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
        uirevision="fire-map-region",
    )
    return fig


def build_activity_chart(
    ts: pd.DataFrame,
    freq_label: str,
    daily: pd.DataFrame | None = None,
    *,
    focus_start: date | None = None,
    focus_end: date | None = None,
) -> go.Figure:
    """Lightweight stacked activity chart (no heavy rangeslider / marker spam)."""
    fig = go.Figure()
    work = ts.copy() if not ts.empty else pd.DataFrame(columns=["period"])

    if daily is not None and not daily.empty:
        base = daily[
            ["period", "total", "is_anomaly", "mean_total", "threshold"]
        ].copy()
        if not work.empty:
            for col in CATEGORY_COLORS:
                if col in work.columns:
                    base = base.merge(
                        work[["period", col]], on="period", how="left"
                    )
                    base[col] = base[col].fillna(0)
        work = base

    if work.empty or "period" not in work.columns:
        fig.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    work = work.sort_values("period").reset_index(drop=True)
    work["period"] = pd.to_datetime(work["period"])
    # Never plot multi-year empty calendars — keep chart to data extent
    if focus_start and focus_end:
        work = work[
            (work["period"] >= pd.Timestamp(focus_start))
            & (work["period"] < pd.Timestamp(focus_end) + pd.Timedelta(days=1))
        ]
    if work.empty:
        fig.update_layout(template="plotly_dark", height=420)
        return fig

    if "total" not in work.columns:
        cat_cols = [c for c in CATEGORY_COLORS if c in work.columns]
        work["total"] = work[cat_cols].sum(axis=1) if cat_cols else 0

    # Prefer lines for long series (bars freeze the browser); bars only for short windows
    use_bars = len(work) <= 60
    if use_bars:
        for col, color in CATEGORY_COLORS.items():
            if col not in work.columns:
                continue
            fig.add_trace(
                go.Bar(
                    x=work["period"],
                    y=work[col],
                    name=col,
                    marker_color=color,
                    opacity=0.9,
                    hovertemplate="%{x|%Y-%m-%d}<br>"
                    + col
                    + ": %{y:,}<extra></extra>",
                )
            )
    else:
        for col, color in CATEGORY_COLORS.items():
            if col not in work.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=work["period"],
                    y=work[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1.6),
                    hovertemplate="%{x|%Y-%m-%d}<br>"
                    + col
                    + ": %{y:,}<extra></extra>",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=work["period"],
            y=work["total"],
            mode="lines",
            name="Total",
            line=dict(color="#ffffff", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>Total: %{y:,}<extra></extra>",
        )
    )

    window = 7 if freq_label == "Daily" else 4
    if len(work) >= max(3, window // 2):
        roll = (
            work.set_index("period")["total"]
            .rolling(window=min(window, len(work)), min_periods=1)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=roll.index,
                y=roll.values,
                mode="lines",
                name=f"{window}-day average",
                line=dict(color="#7df9ff", width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Avg: %{y:.0f}<extra></extra>",
            )
        )

    if "mean_total" in work.columns and len(work):
        mean_v = float(work["mean_total"].iloc[0])
        if mean_v > 0:
            fig.add_hline(
                y=mean_v,
                line_dash="dash",
                line_color="rgba(125, 249, 255, 0.5)",
                annotation_text=f"Mean {mean_v:.0f}",
                annotation_position="top left",
            )

    if "is_anomaly" in work.columns:
        anomalies = work[work["is_anomaly"]].nlargest(
            MAX_CHART_ANOMALIES, "total"
        )
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["period"],
                    y=anomalies["total"],
                    mode="markers",
                    name="Busy day",
                    marker=dict(
                        color=ANOMALY_COLOR,
                        size=11,
                        symbol="diamond",
                        line=dict(width=1, color="#ffffff"),
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>Busy: %{y:,}<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Fire activity · {freq_label}",
            x=0,
            xanchor="left",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            type="date",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Hot spots",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=70, b=40),
        height=440,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        bargap=0.15,
        uirevision="activity-chart-lite",
    )
    return fig


def _parse_date_range(value, today: date) -> tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, date):
        return value, value
    return today, today


def _archive_mtimes_key(folder: Path) -> tuple:
    files = list_archive_csv_files(folder)
    return tuple((p.name, p.stat().st_mtime_ns) for p in files)


def _classify_map_points(raw: pd.DataFrame) -> pd.DataFrame:
    """Fast map categories: full matching only for small samples."""
    if raw.empty:
        return raw
    if len(raw) <= MATCH_ROW_LIMIT:
        return harmonize_hotspots(raw)
    # Large Global samples: sensor label only (keeps UI responsive)
    from src.harmonize import prepare_firms, MODIS_ONLY, VIIRS_ONLY

    prepared = prepare_firms(raw)
    if prepared.empty:
        return prepared
    prepared = prepared.copy()
    prepared["category"] = prepared["sensor"].map(
        lambda s: MODIS_ONLY if s == "MODIS" else VIIRS_ONLY
    )
    return prepared


def _downsample_chart_daily(daily: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """Aggregate long series so Plotly never renders thousands of bars."""
    if daily.empty or len(daily) <= max_points:
        return daily
    work = daily.sort_values("period").set_index("period")
    span_days = max((work.index.max() - work.index.min()).days, 1)
    if span_days > 700:
        rule = "MS"  # monthly
    elif span_days > 180:
        rule = "W-MON"
    else:
        # Even short-but-dense windows: bucket into ~max_points bins
        rule = f"{max(1, span_days // max_points)}D"
    agg = {
        "total": "sum",
        MATCHED: "sum",
        MODIS_ONLY: "sum",
        VIIRS_ONLY: "sum",
    }
    agg = {k: v for k, v in agg.items() if k in work.columns}
    out = work.resample(rule).agg(agg).dropna(how="all").reset_index()
    positive = out.loc[out["total"] > 0, "total"] if "total" in out.columns else []
    mean_total = float(positive.mean()) if len(positive) else 0.0
    out["mean_total"] = mean_total
    out["threshold"] = mean_total * 2.0
    out["is_anomaly"] = (out["total"] > out["threshold"]) & (mean_total > 0)
    return out


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def load_archive_slice(
    folder_str: str,
    mtimes_key: tuple,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_iso: str,
    end_iso: str,
) -> dict:
    """
    Load archive for any region/date span:
    - exact daily activity series (SQL aggregate, full range)
    - point sample for the map (fast classify / optional match)
    """
    folder = Path(folder_str)
    daily, daily_stats = load_daily_activity(
        folder,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start_iso,
        end=end_iso,
    )
    raw, names, stats = load_filtered_csvs(
        folder,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start_iso,
        end=end_iso,
        known_total=int(daily_stats.get("total_count") or 0),
    )
    harmonized = _classify_map_points(raw)
    # Keep only columns the UI needs (smaller session / cache payload)
    keep = [
        c
        for c in (
            "latitude",
            "longitude",
            "acq_datetime",
            "sensor",
            "category",
            "confidence",
            "satellite",
        )
        if c in harmonized.columns
    ]
    if keep and not harmonized.empty:
        harmonized = harmonized[keep].copy()
    stats = {**stats, **daily_stats}
    return {
        "raw_rows": int(daily_stats.get("total_count") or len(raw)),
        "files": names,
        "harmonized": harmonized,
        "daily": daily,
        "stats": stats,
        "mtimes_key": mtimes_key,
    }


def resolve_controls_bbox(
    controls: dict,
) -> tuple[str, float, float, float, float]:
    """Return (region_label, min_lon, min_lat, max_lon, max_lat)."""
    mode = controls["region_mode"]
    if mode == "Preset region":
        name = controls["region_name"]
        return name, *PRESET_REGIONS[name]
    if mode == "Manual bbox":
        min_lon = float(controls["min_lon"])
        min_lat = float(controls["min_lat"])
        max_lon = float(controls["max_lon"])
        max_lat = float(controls["max_lat"])
        if not (min_lon < max_lon and min_lat < max_lat):
            raise FirmsApiError(
                "Invalid bbox: require min_lon < max_lon and min_lat < max_lat."
            )
        return "Manual bbox", min_lon, min_lat, max_lon, max_lat

    bbox = resolve_place_bbox(controls["place_query"])
    label = (controls["place_query"] or "").strip().title() or "Custom"
    return label, *bbox


def _fetch_and_harmonize(
    map_key: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch all NRT sensors and run harmonization."""
    raw = fetch_firms_area(
        map_key=map_key,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start_date=start_date,
        end_date=end_date,
        sensors=AVAILABLE_SENSORS,
    )
    if raw.empty:
        return raw
    missing = validate_firms_columns(raw)
    if missing:
        raise FirmsApiError(
            "FIRMS response is missing columns: " + ", ".join(missing)
        )
    return harmonize_hotspots(raw)


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    if pct == float("inf"):
        return "+∞%"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def render_comparison(primary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Side-by-side stats with percent difference."""
    st.subheader("Period comparison")
    rows = compare_summaries(primary, comparison)
    labels = {
        "total": "Total",
        "matched": "Matched",
        "modis_only": "MODIS-only",
        "viirs_only": "VIIRS-only",
    }
    cols = st.columns(4)
    for col, key in zip(cols, labels):
        item = rows[key]
        col.metric(
            labels[key],
            f"{item['primary']}",
            delta=f"{item['delta']} ({_fmt_pct(item['pct'])}) vs comparison",
        )


def render_control_panel(
    source_mode: str,
    archive_date_min: date | None = None,
    archive_date_max: date | None = None,
) -> dict:
    """Bottom panel: region + dates."""
    st.header("Settings")

    today = date.today()
    default_start = today - timedelta(days=2)
    default_end = today
    compare_default_start = today - timedelta(days=9)
    compare_default_end = today - timedelta(days=7)
    date_max = today

    if source_mode == "Archive CSV" and archive_date_min and archive_date_max:
        date_max = archive_date_max
        august_year = archive_date_max.year
        candidate_end = date(august_year, 8, 15)
        if candidate_end > archive_date_max or candidate_end < archive_date_min:
            august_year -= 1
            candidate_end = date(august_year, 8, 15)
        if archive_date_min <= candidate_end <= archive_date_max:
            default_end = candidate_end
            default_start = max(archive_date_min, date(august_year, 8, 1))
        else:
            default_end = archive_date_max
            default_start = max(
                archive_date_min, archive_date_max - timedelta(days=14)
            )
        compare_default_end = max(
            archive_date_min, default_start - timedelta(days=1)
        )
        compare_default_start = max(
            archive_date_min, compare_default_end - timedelta(days=14)
        )

    compare_mode = st.toggle("Compare two periods", value=False)

    col_region, col_dates = st.columns((1.6, 1.2))

    with col_region:
        region_mode = st.radio(
            "Region",
            options=("Preset", "Place name", "Custom area"),
            horizontal=True,
        )
        place_query = ""
        min_lon = min_lat = max_lon = max_lat = None
        region_name = "Custom"

        if region_mode == "Preset":
            region_name = st.selectbox(
                "Choose a region",
                options=list(PRESET_REGIONS.keys()),
                index=list(PRESET_REGIONS.keys()).index("Turkey")
                if "Turkey" in PRESET_REGIONS
                else 0,
                label_visibility="collapsed",
            )
            min_lon, min_lat, max_lon, max_lat = PRESET_REGIONS[region_name]
        elif region_mode == "Place name":
            place_query = st.text_input(
                "City or region",
                placeholder="e.g. Ankara, Istanbul, Amazon",
                label_visibility="collapsed",
            )
            region_name = (place_query or "").strip() or "Custom"
        else:
            region_name = "Custom area"
            b1, b2 = st.columns(2)
            with b1:
                min_lon = st.number_input("West", value=-10.0, format="%.2f")
                min_lat = st.number_input("South", value=30.0, format="%.2f")
            with b2:
                max_lon = st.number_input("East", value=40.0, format="%.2f")
                max_lat = st.number_input("North", value=45.0, format="%.2f")

    with col_dates:
        primary_range = st.date_input(
            "Dates",
            value=(default_start, default_end),
            min_value=archive_date_min if source_mode == "Archive CSV" else None,
            max_value=date_max,
            key="primary_dates_v2",
        )
        start_date, end_date = _parse_date_range(primary_range, default_end)

        compare_start = compare_end = None
        if compare_mode:
            compare_range = st.date_input(
                "Compare with",
                value=(compare_default_start, compare_default_end),
                min_value=archive_date_min if source_mode == "Archive CSV" else None,
                max_value=date_max,
                key="compare_dates",
            )
            compare_start, compare_end = _parse_date_range(
                compare_range, compare_default_end
            )

        # Archive updates live on every control change; NRT still needs a click
        if source_mode == "NRT API":
            fetch_clicked = st.button(
                "Fetch live data",
                type="primary",
                use_container_width=True,
            )
        else:
            fetch_clicked = True

    # Map radio labels back to internal region modes used elsewhere
    region_mode_internal = {
        "Preset": "Preset region",
        "Place name": "Custom place",
        "Custom area": "Manual bbox",
    }[region_mode]

    return {
        "region_mode": region_mode_internal,
        "region_name": region_name,
        "place_query": place_query,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "start_date": start_date,
        "end_date": end_date,
        "compare_mode": compare_mode,
        "compare_start": compare_start,
        "compare_end": compare_end,
        "freq_choice": "Daily",
        "fetch_clicked": fetch_clicked,
    }


def _coerce_focus_dates(
    start: date,
    end: date,
    focus: tuple | list | None,
) -> tuple[date, date]:
    """Clamp a chart focus window inside [start, end]."""
    if not focus or len(focus) != 2:
        return start, end
    a, b = focus[0], focus[1]
    if hasattr(a, "date"):
        a = a.date()
    if hasattr(b, "date"):
        b = b.date()
    a = max(start, min(a, end))
    b = max(start, min(b, end))
    if a > b:
        a, b = b, a
    return a, b


def render_results(controls: dict) -> None:
    """Summary, map on top, detailed activity chart below with date focus."""
    meta = st.session_state.last_meta or {}
    harmonized: pd.DataFrame = st.session_state.harmonized
    comparison: pd.DataFrame | None = st.session_state.get("comparison")

    start = meta.get("start_date")
    end = meta.get("end_date")
    bbox = meta.get("bbox")
    if start is None or end is None:
        start = controls["start_date"]
        end = controls["end_date"]

    data_key = (
        str(meta.get("region")),
        str(start),
        str(end),
        str(meta.get("bbox")),
    )
    if st.session_state.get("_focus_data_key") != data_key:
        st.session_state._focus_data_key = data_key
        st.session_state.chart_focus = (start, end)
        # Reset slider widget state for the new loaded period
        if "chart_focus_slider" in st.session_state:
            del st.session_state["chart_focus_slider"]

    # Prefer live slider state (available on rerun before the widget runs)
    raw_focus = st.session_state.get("chart_focus_slider")
    if raw_focus is None:
        raw_focus = st.session_state.get("chart_focus", (start, end))
    focus_start, focus_end = _coerce_focus_dates(start, end, raw_focus)
    st.session_state.chart_focus = (focus_start, focus_end)

    focus_mask = (
        (harmonized["acq_datetime"] >= pd.Timestamp(focus_start))
        & (
            harmonized["acq_datetime"]
            < pd.Timestamp(focus_end) + pd.Timedelta(days=1)
        )
    )
    focused = harmonized.loc[focus_mask].copy()
    if focused.empty and not harmonized.empty:
        focused = harmonized

    # Full-range exact daily series from DuckDB (any Global / multi-year span)
    daily_series = st.session_state.get("daily_series")
    if isinstance(daily_series, pd.DataFrame) and not daily_series.empty:
        ds = daily_series.copy()
        ds["period"] = pd.to_datetime(ds["period"])
        daily_focus = ds[
            (ds["period"] >= pd.Timestamp(focus_start))
            & (ds["period"] < pd.Timestamp(focus_end) + pd.Timedelta(days=1))
        ].copy()
        if daily_focus.empty:
            daily_focus = ds
        # Recompute anomaly stats on the focused window
        positive = daily_focus.loc[daily_focus["total"] > 0, "total"]
        mean_total = float(positive.mean()) if len(positive) else 0.0
        thr = mean_total * 2.0
        daily_focus["mean_total"] = mean_total
        daily_focus["threshold"] = thr
        daily_focus["is_anomaly"] = (daily_focus["total"] > thr) & (mean_total > 0)
    else:
        daily_focus = daily_hotspot_counts(
            focused,
            start=pd.Timestamp(focus_start),
            end=pd.Timestamp(focus_end),
        )

    anomalies = count_anomalies(daily_focus)
    map_stats = summarize_categories(focused)
    exact_total = (
        int(daily_focus["total"].sum())
        if not daily_focus.empty and "total" in daily_focus.columns
        else map_stats["total"]
    )
    exact_modis = (
        int(daily_focus[MODIS_ONLY].sum())
        if not daily_focus.empty and MODIS_ONLY in daily_focus.columns
        else map_stats["modis_only"]
    )
    exact_viirs = (
        int(daily_focus[VIIRS_ONLY].sum())
        if not daily_focus.empty and VIIRS_ONLY in daily_focus.columns
        else map_stats["viirs_only"]
    )

    if meta:
        st.caption(
            f"{meta.get('region', '—')}  ·  {start} → {end}  ·  "
            f"focus {focus_start} → {focus_end}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{exact_total:,}")
    c2.metric("Matched (map)", f"{map_stats['matched']:,}")
    c3.metric("MODIS", f"{exact_modis:,}")
    c4.metric("VIIRS", f"{exact_viirs:,}")
    c5.metric("Busy days", anomalies)

    if comparison is not None and not comparison.empty:
        st.divider()
        render_comparison(focused, comparison)

    if anomalies and not daily_focus.empty:
        anomaly_days = daily_focus.loc[daily_focus["is_anomaly"], "period"]
        # Cap labels so a dense Global window doesn't flood the page
        labels = ", ".join(
            pd.Timestamp(d).strftime("%Y-%m-%d")
            for d in list(anomaly_days)[:12]
        )
        extra = len(anomaly_days) - 12
        if extra > 0:
            labels = f"{labels} (+{extra} more)"
        st.warning(f"Unusually busy days: {labels}")

    st.divider()

    # —— Map (full width, top) ——
    st.subheader("Map")
    st.caption("Matched · MODIS only · VIIRS only")
    map_fig = build_map_figure(focused, bbox=bbox)
    map_fig.update_layout(height=480)
    st.plotly_chart(
        map_fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="fire_map",
    )

    # —— Activity chart (full width, below map) ——
    st.subheader("Activity")
    head_l, head_r = st.columns((2.2, 1.0))
    with head_l:
        st.caption(
            "Global and multi-year ranges are fully supported. "
            "Use the slider to zoom the map focus."
        )
    with head_r:
        default_weekly = ((end - start).days >= 90)
        freq_choice = st.radio(
            "Detail",
            options=("Daily", "Weekly"),
            horizontal=True,
            index=1 if default_weekly else 0,
            key="chart_freq",
        )

    span_days = max((end - start).days, 0)
    if span_days == 0:
        st.caption(f"Single day: {start}")
    else:
        st.slider(
            "Date range (chart & map)",
            min_value=start,
            max_value=end,
            value=(focus_start, focus_end),
            format="YYYY-MM-DD",
            key="chart_focus_slider",
        )

    # Chart from exact aggregates, downsampled so Plotly stays light
    if daily_focus.empty:
        st.info("Not enough data for the chart.")
    else:
        chart_daily = daily_focus.copy()
        span_focus = max((focus_end - focus_start).days, 1)
        # Auto coarsen for long windows (still no selection limits)
        if freq_choice == "Weekly" or span_focus >= 90:
            tmp = chart_daily.set_index("period").sort_index()
            agg = {
                k: "sum"
                for k in ("total", MATCHED, MODIS_ONLY, VIIRS_ONLY)
                if k in tmp.columns
            }
            chart_daily = tmp.resample("W-MON").agg(agg).dropna(how="all").reset_index()
            positive = chart_daily.loc[chart_daily["total"] > 0, "total"]
            mean_total = float(positive.mean()) if len(positive) else 0.0
            chart_daily["mean_total"] = mean_total
            chart_daily["threshold"] = mean_total * 2.0
            chart_daily["is_anomaly"] = (chart_daily["total"] > chart_daily["threshold"]) & (
                mean_total > 0
            )
            freq_label = "Weekly"
        else:
            freq_label = "Daily"
        chart_daily = _downsample_chart_daily(chart_daily)

        ts = chart_daily[
            [
                c
                for c in ("period", MATCHED, MODIS_ONLY, VIIRS_ONLY)
                if c in chart_daily.columns
            ]
        ].copy()
        fig = build_activity_chart(
            ts,
            freq_label,
            daily=chart_daily,
            focus_start=focus_start,
            focus_end=focus_end,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key="fire_activity",
        )

    with st.expander("Map sample table", expanded=False):
        show_cols = [
            c
            for c in (
                "latitude",
                "longitude",
                "acq_datetime",
                "confidence",
                "satellite",
                "sensor",
                "category",
            )
            if c in focused.columns
        ]
        display = focused[show_cols].sort_values("acq_datetime").head(5_000)
        st.dataframe(display, use_container_width=True, hide_index=True)


def run_archive_pipeline(controls: dict) -> str | None:
    """
    Load archive for current region/dates into session.

    Skips reload when the selection is unchanged (critical: chart slider
    must not re-query Global data on every drag).
    """
    try:
        region_label, min_lon, min_lat, max_lon, max_lat = resolve_controls_bbox(
            controls
        )
    except FirmsApiError as exc:
        st.error(str(exc))
        return None

    load_key = (
        region_label,
        round(min_lon, 4),
        round(min_lat, 4),
        round(max_lon, 4),
        round(max_lat, 4),
        controls["start_date"].isoformat(),
        controls["end_date"].isoformat(),
        bool(controls.get("compare_mode")),
        (controls.get("compare_start") or date.min).isoformat()
        if controls.get("compare_mode")
        else "",
        (controls.get("compare_end") or date.min).isoformat()
        if controls.get("compare_mode")
        else "",
    )
    if (
        st.session_state.get("archive_load_key") == load_key
        and st.session_state.get("harmonized") is not None
        and st.session_state.get("daily_series") is not None
    ):
        return None

    mtimes = _archive_mtimes_key(DATA_PATH)
    start_iso = controls["start_date"].isoformat()
    end_iso = controls["end_date"].isoformat()

    try:
        _, rebuilt = ensure_hotspots_db(DATA_PATH)
    except ArchiveCsvError as exc:
        st.error(str(exc))
        return None
    if rebuilt:
        st.toast("Archive index ready.")

    try:
        payload = load_archive_slice(
            str(DATA_PATH),
            mtimes,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            start_iso,
            end_iso,
        )
    except ArchiveCsvError as exc:
        st.error(str(exc))
        return None
    except Exception:  # noqa: BLE001
        st.error("Could not load fire data for this selection.")
        return None

    primary = payload["harmonized"]
    stats = payload.get("stats") or {}
    daily = payload.get("daily")
    if daily is None:
        daily = pd.DataFrame()
    total_count = int(
        stats.get("total_count") or stats.get("rows_scanned") or len(primary)
    )
    if primary.empty and (
        daily.empty or int(daily.get("total", pd.Series(dtype=int)).sum()) == 0
    ):
        st.warning("No fires found for this region and date range.")
        st.session_state.harmonized = None
        st.session_state.daily_series = None
        st.session_state.archive_load_key = None
        st.session_state.comparison = None
        return None

    st.session_state.daily_series = daily
    st.session_state.archive_load_key = load_key
    if stats.get("truncated") and total_count > len(primary):
        st.caption(
            f"Map sample {len(primary):,} of {total_count:,} points — "
            "chart uses the full selection."
        )

    st.session_state.harmonized = primary
    st.session_state.last_meta = {
        "region": region_label,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "start_date": controls["start_date"],
        "end_date": controls["end_date"],
        "raw_rows": total_count,
        "source": "Archive CSV",
        "files": payload.get("files", []),
        "rows_scanned": int(stats.get("rows_scanned", 0)),
        "map_sample": int(len(primary)),
    }

    st.session_state.comparison = None
    st.session_state.comparison_meta = None

    if (
        controls["compare_mode"]
        and controls["compare_start"]
        and controls["compare_end"]
    ):
        with st.spinner("Loading comparison…"):
            try:
                cmp_payload = load_archive_slice(
                    str(DATA_PATH),
                    mtimes,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    controls["compare_start"].isoformat(),
                    controls["compare_end"].isoformat(),
                )
            except Exception:  # noqa: BLE001
                cmp_payload = None
        if cmp_payload is None or cmp_payload["harmonized"].empty:
            st.warning("No fires found in the comparison period.")
        else:
            st.session_state.comparison = cmp_payload["harmonized"]
            st.session_state.comparison_meta = {
                "start_date": controls["compare_start"],
                "end_date": controls["compare_end"],
                "raw_rows": int(len(cmp_payload["harmonized"])),
            }

    return None


def run_nrt_pipeline(controls: dict, map_key: str) -> None:
    """Fetch NRT API data into session results."""
    try:
        region_label, min_lon, min_lat, max_lon, max_lat = resolve_controls_bbox(
            controls
        )
    except FirmsApiError as exc:
        st.error(str(exc))
        st.session_state.harmonized = None
        st.session_state.comparison = None
        return

    with st.spinner("Updating map…"):
        try:
            primary = _fetch_and_harmonize(
                map_key,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                controls["start_date"],
                controls["end_date"],
            )
        except FirmsApiError as exc:
            st.error(str(exc))
            st.session_state.harmonized = None
            st.session_state.comparison = None
            return
        except Exception:  # noqa: BLE001
            st.error("Could not load fire data. Try again in a moment.")
            st.session_state.harmonized = None
            st.session_state.comparison = None
            return

    if primary.empty:
        st.warning("No fires found for this region and date range.")
        st.session_state.harmonized = None
        st.session_state.comparison = None
        return

    st.session_state.harmonized = primary
    st.session_state.last_meta = {
        "region": region_label,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "start_date": controls["start_date"],
        "end_date": controls["end_date"],
        "raw_rows": int(len(primary)),
        "source": "NRT API",
    }
    st.session_state.comparison = None
    st.session_state.comparison_meta = None

    if (
        controls["compare_mode"]
        and controls["compare_start"]
        and controls["compare_end"]
    ):
        with st.spinner("Loading comparison…"):
            try:
                secondary = _fetch_and_harmonize(
                    map_key,
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    controls["compare_start"],
                    controls["compare_end"],
                )
            except Exception:  # noqa: BLE001
                secondary = pd.DataFrame()

        if secondary.empty:
            st.warning("No fires found in the comparison period.")
        else:
            st.session_state.comparison = secondary
            st.session_state.comparison_meta = {
                "start_date": controls["compare_start"],
                "end_date": controls["compare_end"],
                "raw_rows": int(len(secondary)),
            }


@st.fragment
def live_archive_explorer(
    archive_date_min: date | None,
    archive_date_max: date | None,
) -> None:
    """
    Controls + results. Archive data reloads only when region/dates change;
    chart focus slider only re-filters in memory (no Global re-query).
    """
    controls = render_control_panel(
        "Archive CSV",
        archive_date_min=archive_date_min,
        archive_date_max=archive_date_max,
    )
    run_archive_pipeline(controls)
    if (
        st.session_state.get("harmonized") is not None
        or (
            isinstance(st.session_state.get("daily_series"), pd.DataFrame)
            and not st.session_state.daily_series.empty
        )
    ):
        render_results(controls)


def main() -> None:
    st.title("Unified Fire Activity Calendar")
    st.caption("MODIS and VIIRS fire activity on one map.")

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None
        st.session_state.comparison = None
        st.session_state.comparison_meta = None

    inventory = describe_csv_files(DATA_PATH)
    archive_date_min, archive_date_max = peek_archive_date_bounds(DATA_PATH)
    # Prefer local archive when present; otherwise live FIRMS (needs MAP_KEY)
    source_mode = "Archive CSV" if inventory else "NRT API"

    if source_mode == "NRT API":
        map_key = get_firms_map_key()
        if not map_key:
            st.warning(
                "No fire archive found and live data needs a FIRMS key in .env."
            )
            with st.expander("Add archive files"):
                uploads = st.file_uploader(
                    "Upload CSV",
                    type=["csv"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if uploads:
                    save_uploaded_csvs(DATA_PATH, uploads)
                    load_archive_slice.clear()
                    st.rerun()
            return

        st.divider()
        controls = render_control_panel(
            source_mode,
            archive_date_min=archive_date_min,
            archive_date_max=archive_date_max,
        )
        if controls["fetch_clicked"]:
            run_nrt_pipeline(controls, map_key)
        if st.session_state.harmonized is not None:
            render_results(controls)
        return

    # Archive: live updates as soon as region/dates change
    st.caption("Change region or dates — the map updates immediately.")
    st.divider()
    live_archive_explorer(archive_date_min, archive_date_max)


if __name__ == "__main__":
    main()
