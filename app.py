"""
Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.
Supports FIRMS Archive CSVs in data/ and live NRT Area API fetches.

FIRMS_MAP_KEY is loaded from .env for NRT mode only.
"""

from __future__ import annotations

import os
import time
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
    elevated_activity_last_days,
    harmonization_quality,
    latest_month_changes,
)
from src.archive_csv import (
    ArchiveCsvError,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    data_dir,
    describe_csv_files,
    ensure_hotspots_db,
    list_archive_csv_files,
    load_daily_activity,
    load_filtered_csvs,
    load_monthly_activity,
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

# NRT abuse guards — host MAP_KEY must not be an unbounded public proxy
NRT_MAX_FETCHES_PER_SESSION = 5
NRT_MIN_SECONDS_BETWEEN_FETCHES = 20


def active_data_path() -> Path:
    """Archive root for this session (shared data/ or per-visitor uploads)."""
    override = st.session_state.get("archive_root")
    if override:
        return Path(override)
    return DATA_PATH


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


def _is_hosted_deploy() -> bool:
    """True on Cursor/Vercel hosted runtimes (not local .env development)."""
    return bool(os.getenv("CURSOR_APP_ID") or os.getenv("VERCEL"))


def _allow_host_map_key() -> bool:
    """
    Host FIRMS_MAP_KEY is for local/dev by default.

    On a hosted deploy it stays off unless FIRMS_ALLOW_HOST_KEY is explicitly set,
    so anonymous visitors cannot burn the operator quota.
    """
    flag = (os.getenv("FIRMS_ALLOW_HOST_KEY") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return not _is_hosted_deploy()


def get_firms_map_key() -> str:
    """
    Resolve MAP_KEY for NRT calls.

    Prefer a session-entered key. The host env key is only used when allowed
    (local/dev, or FIRMS_ALLOW_HOST_KEY). Value is never logged.
    """
    session_key = (st.session_state.get("visitor_map_key") or "").strip()
    if session_key:
        return session_key
    if not _allow_host_map_key():
        return ""
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def _session_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx and getattr(ctx, "session_id", None):
            return str(ctx.session_id)
    except Exception:  # noqa: BLE001
        pass
    return "local"


def session_upload_dir() -> Path:
    """Per-visitor upload folder so archives are not a shared mutable bucket."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in _session_id())[
        :64
    ]
    path = (DATA_PATH / "_uploads" / safe_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    # Stay under data/ even if session_id were hostile
    data_root = DATA_PATH.resolve()
    if path != data_root and data_root not in path.parents:
        raise ArchiveCsvError("Invalid upload directory.")
    return path


def _nrt_rate_limit_ok() -> tuple[bool, str]:
    """Session rate limit for outbound FIRMS Area API calls."""
    now = time.monotonic()
    stamps: list[float] = list(st.session_state.get("nrt_fetch_times") or [])
    stamps = [t for t in stamps if now - t < 3600]
    if len(stamps) >= NRT_MAX_FETCHES_PER_SESSION:
        return (
            False,
            f"Live FIRMS limit reached ({NRT_MAX_FETCHES_PER_SESSION} fetches / session).",
        )
    if stamps and (now - stamps[-1]) < NRT_MIN_SECONDS_BETWEEN_FETCHES:
        wait = int(NRT_MIN_SECONDS_BETWEEN_FETCHES - (now - stamps[-1])) + 1
        return False, f"Please wait {wait}s before another live FIRMS request."
    return True, ""


def _record_nrt_fetch() -> None:
    now = time.monotonic()
    stamps: list[float] = list(st.session_state.get("nrt_fetch_times") or [])
    stamps.append(now)
    st.session_state.nrt_fetch_times = stamps[-NRT_MAX_FETCHES_PER_SESSION :]


def render_map_key_input() -> None:
    """Optional visitor MAP_KEY (session-only); never echoed back to logs."""
    with st.expander("FIRMS MAP_KEY (optional, session only)", expanded=False):
        st.caption(
            "Enter your own NASA FIRMS MAP_KEY for live NRT. "
            "It stays in this browser session and is not written to disk."
        )
        entered = st.text_input(
            "MAP_KEY",
            value="",
            type="password",
            key="visitor_map_key_input",
            autocomplete="off",
        )
        if entered.strip():
            st.session_state.visitor_map_key = entered.strip()
        elif "visitor_map_key" not in st.session_state:
            st.session_state.visitor_map_key = ""


def guard_nrt_fetch(map_key: str) -> bool:
    """Return True if an NRT fetch may proceed; otherwise show a warning."""
    if not map_key:
        st.warning(
            "Live FIRMS needs a MAP_KEY — enter yours above, or set FIRMS_MAP_KEY locally."
        )
        return False
    ok, reason = _nrt_rate_limit_ok()
    if not ok:
        st.warning(reason)
        return False
    return True


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
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return (max_lon - min_lon) >= 120 or (max_lat - min_lat) >= 100


# Map budgets — charts stay exact via DuckDB; the browser cannot paint 80k markers.
MAP_FULL_THRESHOLD = 80_000  # load every row when the selection is smaller
MAX_MAP_LOAD = MAP_FULL_THRESHOLD
MAX_MAP_RENDER = 3_000  # hard Plotly paint cap (Matched + high FRP first)
MAX_CHART_ANOMALIES = 15
MAX_CHART_POINTS = 180
# Skip expensive MODIS↔VIIRS matching above this map-sample size
MATCH_ROW_LIMIT = 8_000


def _map_point_limit(bbox: tuple[float, float, float, float] | None = None) -> int:
    return MAX_MAP_RENDER


def _map_load_budget(
    bbox: tuple[float, float, float, float] | None = None,
    *,
    known_total: int = 0,
) -> int:
    """Load all rows under the threshold; otherwise cap the DuckDB pull."""
    if known_total and 0 < known_total <= MAP_FULL_THRESHOLD:
        return int(known_total)
    if known_total and known_total > MAP_FULL_THRESHOLD:
        return MAX_MAP_LOAD
    return MAX_MAP_LOAD


def _sample_for_map(
    df: pd.DataFrame,
    limit: int = MAX_MAP_RENDER,
) -> pd.DataFrame:
    """
    Keep every point at or under the threshold.

    Above the limit: keep Matched and high-FRP first, then fill with a
    spatial grid so sparse regions are not erased.
    """
    if df is None or df.empty or len(df) <= limit:
        return df if df is not None else pd.DataFrame()

    keep_cols = [
        c
        for c in (
            "latitude",
            "longitude",
            "category",
            "frp",
            "acq_datetime",
            "confidence",
            "sensor",
        )
        if c in df.columns
    ]
    work = df[keep_cols].copy()
    work["_frp"] = (
        pd.to_numeric(work["frp"], errors="coerce").fillna(0.0)
        if "frp" in work.columns
        else 0.0
    )
    work["_matched"] = (
        (work["category"] == MATCHED).astype("int8")
        if "category" in work.columns
        else 0
    )

    # 1) Priority pool: Matched first, then highest FRP
    priority_n = max(int(limit * 0.7), 1)
    priority = work.sort_values(
        ["_matched", "_frp"],
        ascending=[False, False],
        kind="mergesort",
    ).head(priority_n)

    # 2) Spatial fill from the remainder for geographic coverage
    leftover = work.loc[~work.index.isin(priority.index)]
    need = limit - len(priority)
    if need > 0 and not leftover.empty:
        lon_span = float(leftover["longitude"].max() - leftover["longitude"].min()) + 1e-6
        lat_span = float(leftover["latitude"].max() - leftover["latitude"].min()) + 1e-6
        area = lon_span * lat_span
        cell = max(0.2, min((area / max(need, 1)) ** 0.5, 2.0))
        leftover = leftover.copy()
        leftover["_gx"] = (leftover["longitude"] / cell).floordiv(1).astype("int32")
        leftover["_gy"] = (leftover["latitude"] / cell).floordiv(1).astype("int32")
        leftover = leftover.sort_values(
            ["_matched", "_frp"],
            ascending=[False, False],
            kind="mergesort",
        )
        spaced = leftover.groupby(["_gx", "_gy"], sort=False, as_index=False).head(1)
        if len(spaced) >= need:
            extra = spaced.head(need)
        else:
            rest = leftover.loc[~leftover.index.isin(spaced.index)]
            fill = rest.head(need - len(spaced))
            extra = pd.concat([spaced, fill], ignore_index=False)
        out = pd.concat([priority, extra], ignore_index=True)
    else:
        out = priority

    if len(out) > limit:
        out = out.head(limit)
    return out.drop(
        columns=["_frp", "_matched", "_gx", "_gy"],
        errors="ignore",
    )


def build_map_figure(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
) -> go.Figure:
    """
    Map markers without tile wrap.

    Data may be loaded up to MAP_FULL_THRESHOLD; Plotly paints at most
    MAX_MAP_RENDER points (Matched + high FRP first).

    Wide / Global → Scattergeo (Natural Earth). Regional → OSM with bounds.
    """
    plot_df = _sample_for_map(df, limit=_map_point_limit(bbox))

    empty_geo = dict(
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor="#1a1a1a",
        showocean=True,
        oceancolor="#0b1220",
        showcountries=True,
        countrycolor="#333333",
    )
    if plot_df is None or plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            geo=empty_geo,
            height=440,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    lons = ((plot_df["longitude"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
    lats = plot_df["latitude"].to_numpy(dtype=float).clip(-90, 90)
    cats = plot_df["category"].to_numpy()

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon = float(lons.min())
        max_lon = float(lons.max())
        min_lat = float(lats.min())
        max_lat = float(lats.max())

    fig = go.Figure()
    # Draw rarer categories last so they stay visible on top
    draw_order = [VIIRS_ONLY, MODIS_ONLY, MATCHED]
    wide = _is_wide_bbox((min_lon, min_lat, max_lon, max_lat))
    n_pts = int(len(plot_df))
    # Shrink markers as density grows so 10k–80k stays readable
    if n_pts > 40_000:
        geo_size, map_size, opacity = 2, 3, 0.55
    elif n_pts > 12_000:
        geo_size, map_size, opacity = 2, 4, 0.65
    elif n_pts > 3_000:
        geo_size, map_size, opacity = 3, 5, 0.75
    else:
        geo_size, map_size, opacity = 4, 6, 0.85

    if wide:
        for cat in draw_order:
            mask = cats == cat
            if not mask.any():
                continue
            fig.add_trace(
                go.Scattergeo(
                    lon=lons[mask],
                    lat=lats[mask],
                    mode="markers",
                    name=cat,
                    hoverinfo="skip",
                    marker=dict(
                        size=geo_size,
                        opacity=opacity,
                        color=CATEGORY_COLORS.get(cat, "#ffffff"),
                        line=dict(width=0),
                    ),
                )
            )
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
            lataxis_range=[max(-85.0, min_lat - 2), min(85.0, max_lat + 2)],
            lonaxis_range=[max(-180.0, min_lon), min(180.0, max_lon)],
            visible=True,
            resolution=110,
        )
        fig.update_layout(
            height=440,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.02, x=0.01),
            paper_bgcolor="rgba(0,0,0,0)",
            geo_bgcolor="rgba(0,0,0,0)",
            uirevision="fire-map-geo",
            dragmode=False,
        )
        return fig

    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    zoom = float(_map_zoom_for_bbox(min_lon, min_lat, max_lon, max_lat))
    pad_lon = max((max_lon - min_lon) * 0.05, 0.3)
    pad_lat = max((max_lat - min_lat) * 0.05, 0.3)
    west = max(-180.0, min_lon - pad_lon)
    east = min(180.0, max_lon + pad_lon)
    south = max(-85.0, min_lat - pad_lat)
    north = min(85.0, max_lat + pad_lat)

    for cat in draw_order:
        mask = cats == cat
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattermap(
                lon=lons[mask],
                lat=lats[mask],
                mode="markers",
                name=cat,
                hoverinfo="skip",
                marker=dict(
                    size=map_size,
                    opacity=opacity,
                    color=CATEGORY_COLORS.get(cat, "#ffffff"),
                ),
            )
        )
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
            bounds=dict(west=west, east=east, south=south, north=north),
        ),
        height=440,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
        uirevision="fire-map-region",
        dragmode=False,
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
        keep_cols = [
            c
            for c in (
                "period",
                "total",
                "frp_sum",
                "is_anomaly",
                "mean_total",
                "threshold",
            )
            if c in daily.columns
        ]
        base = daily[keep_cols].copy()
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

    has_frp = "frp_sum" in work.columns and float(work["frp_sum"].fillna(0).sum()) > 0
    if has_frp:
        fig.add_trace(
            go.Scatter(
                x=work["period"],
                y=work["frp_sum"],
                mode="lines",
                name="FRP (MW)",
                yaxis="y2",
                line=dict(color="#fbbf24", width=2.2),
                hovertemplate="%{x|%Y-%m-%d}<br>FRP: %{y:,.1f} MW<extra></extra>",
            )
        )

    layout_kwargs = dict(
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
        margin=dict(l=50, r=55 if has_frp else 20, t=70, b=40),
        height=440,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack",
        bargap=0.15,
        uirevision="activity-chart-lite",
    )
    if has_frp:
        layout_kwargs["yaxis2"] = dict(
            title=dict(text="FRP (MW)", font=dict(color="#fbbf24")),
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            tickfont=dict(color="#fbbf24"),
        )
    fig.update_layout(**layout_kwargs)
    return fig


def build_monthly_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Monthly hotspot counts + FRP with a dual axis."""
    fig = go.Figure()
    if monthly is None or monthly.empty:
        fig.update_layout(
            template="plotly_dark",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    work = monthly.sort_values("period").copy()
    work["period"] = pd.to_datetime(work["period"])
    fig.add_trace(
        go.Bar(
            x=work["period"],
            y=work["total"],
            name="Hotspots",
            marker_color="#14919b",
            opacity=0.85,
            hovertemplate="%{x|%Y-%m}<br>Hotspots: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["period"],
            y=work["frp_sum"],
            mode="lines+markers",
            name="FRP (MW)",
            yaxis="y2",
            line=dict(color="#fbbf24", width=2.4),
            marker=dict(size=7, color="#fbbf24"),
            hovertemplate="%{x|%Y-%m}<br>FRP: %{y:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Monthly trend", x=0, xanchor="left", font=dict(size=16)),
        xaxis=dict(
            title="Month",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            type="date",
            dtick="M1",
            tickformat="%Y-%m",
        ),
        yaxis=dict(
            title="Hot spots",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            rangemode="tozero",
        ),
        yaxis2=dict(
            title=dict(text="FRP (MW)", font=dict(color="#fbbf24")),
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
            tickfont=dict(color="#fbbf24"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hovermode="x unified",
        margin=dict(l=50, r=55, t=70, b=40),
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        uirevision="monthly-trend",
    )
    return fig


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    if value == float("inf"):
        return "new"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


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
        "frp_sum": "sum",
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


def _trim_map_columns(harmonized: pd.DataFrame) -> pd.DataFrame:
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
        return harmonized[keep].copy()
    return harmonized


def _cached_mtimes_key() -> tuple:
    """Avoid re-statting multi-GB CSVs on every slider tick."""
    cached = st.session_state.get("_archive_mtimes")
    if cached is not None:
        return cached
    key = _archive_mtimes_key(active_data_path())
    st.session_state._archive_mtimes = key
    return key


@st.cache_data(show_spinner=False, max_entries=24, ttl=3600)
def load_map_slice(
    folder_str: str,
    mtimes_key: tuple,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_iso: str,
    end_iso: str,
    known_total: int = 0,
    max_rows: int = MAX_MAP_LOAD,
) -> dict:
    """
    Fire points for the map focus window.

    Loads every row when the selection is under MAP_FULL_THRESHOLD;
    otherwise priority-samples (high FRP + spatial coverage), then
    ranks Matched + high FRP for display.
    """
    folder = Path(folder_str)
    raw, names, stats = load_filtered_csvs(
        folder,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start_iso,
        end=end_iso,
        max_rows=int(max_rows),
        known_total=known_total or None,
    )
    harmonized = _classify_map_points(raw)
    if not harmonized.empty:
        # Carry FRP through classify when present on the raw frame
        if "frp" in raw.columns and "frp" not in harmonized.columns:
            # Align by position after prepare_firms may drop rows — merge on coords/time when possible
            harmonized = harmonized.copy()
            if len(harmonized) == len(raw):
                harmonized["frp"] = pd.to_numeric(raw["frp"], errors="coerce")
            elif {"latitude", "longitude"}.issubset(raw.columns):
                frp_lookup = raw[["latitude", "longitude"]].copy()
                frp_lookup["frp"] = pd.to_numeric(raw["frp"], errors="coerce")
                frp_lookup = frp_lookup.drop_duplicates(
                    subset=["latitude", "longitude"], keep="first"
                )
                harmonized = harmonized.merge(
                    frp_lookup, on=["latitude", "longitude"], how="left"
                )
        cols = [
            c
            for c in (
                "latitude",
                "longitude",
                "category",
                "acq_datetime",
                "confidence",
                "frp",
            )
            if c in harmonized.columns
        ]
        harmonized = harmonized[cols].copy()
    return {
        "harmonized": harmonized,
        "files": names,
        "stats": stats,
        "mtimes_key": mtimes_key,
    }


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
    - exact daily + monthly activity (SQL aggregates)
    - point sample for the map / quality score
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
    monthly = load_monthly_activity(
        folder,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start_iso,
        end=end_iso,
    )
    known_total = int(daily_stats.get("total_count") or 0)
    map_payload = load_map_slice(
        folder_str,
        mtimes_key,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        start_iso,
        end_iso,
        known_total,
        _map_load_budget(
            (min_lon, min_lat, max_lon, max_lat),
            known_total=known_total,
        ),
    )
    quality = harmonization_quality(map_payload["harmonized"])
    stats = {
        **(map_payload.get("stats") or {}),
        **daily_stats,
        "harmonization_quality": quality,
    }
    return {
        "raw_rows": int(daily_stats.get("total_count") or 0),
        "files": map_payload.get("files", []),
        "harmonized": map_payload["harmonized"],
        "daily": daily,
        "monthly": monthly,
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
    """Side-by-side stats with percent difference (point samples)."""
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


def render_archive_vs_live(
    *,
    archive_total: int,
    archive_start: date,
    archive_end: date,
    live_total: int,
    live_start: date,
    live_end: date,
    live_label: str = "Live NRT",
) -> None:
    """
    Compare uploaded archive span with the short live FIRMS window.

    Uses daily averages so multi-year archive totals are comparable to
    a 5–10 day NRT slice.
    """
    st.subheader("Archive vs live")
    arch_days = max((archive_end - archive_start).days + 1, 1)
    live_days = max((live_end - live_start).days + 1, 1)
    arch_avg = archive_total / arch_days
    live_avg = live_total / live_days
    delta_avg = live_avg - arch_avg
    pct = None
    if arch_avg > 0:
        pct = (delta_avg / arch_avg) * 100.0
    elif live_avg > 0:
        pct = float("inf")

    st.caption(
        f"Archive **{archive_start} → {archive_end}** ({arch_days} days) vs "
        f"{live_label} **{live_start} → {live_end}** ({live_days} days). "
        "Delta is live daily average vs archive daily average."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Archive total", f"{archive_total:,}")
    c2.metric("Archive / day", f"{arch_avg:,.1f}")
    c3.metric(f"{live_label} total", f"{live_total:,}")
    c4.metric(
        f"{live_label} / day",
        f"{live_avg:,.1f}",
        delta=f"{delta_avg:+,.1f} ({_fmt_pct(pct)})",
    )


def _nrt_window_from_end(end_date: date, window_days: int) -> tuple[date, date]:
    """Slide a 1…MAX_DAY_RANGE day window that always ends on end_date."""
    today = date.today()
    nrt_floor = today - timedelta(days=365)
    end = min(max(end_date, nrt_floor), today)
    days = max(1, min(int(window_days), MAX_DAY_RANGE))
    start = end - timedelta(days=days - 1)
    if start < nrt_floor:
        start = nrt_floor
    return start, end


def render_control_panel(
    source_mode: str,
    archive_date_min: date | None = None,
    archive_date_max: date | None = None,
    *,
    nrt_compare_available: bool = False,
) -> dict:
    """Bottom panel: region + dates (+ optional live NRT compare)."""
    st.header("Settings")

    today = date.today()
    default_start = today - timedelta(days=2)
    default_end = today
    date_max = today
    nrt_min = today - timedelta(days=365)

    if source_mode == "Archive CSV" and archive_date_min and archive_date_max:
        # Default to the full uploaded archive span (e.g. 2023 → 2026)
        date_max = archive_date_max
        default_start = archive_date_min
        default_end = archive_date_max

    compare_source = "none"
    compare_mode = False
    if source_mode == "Archive CSV" and nrt_compare_available:
        compare_mode = st.toggle(
            "Compare with live FIRMS",
            value=True,
            help=(
                "Uses your uploaded archive for Archive dates, then fetches "
                f"the Live dates window (max {MAX_DAY_RANGE} days) via MAP_KEY."
            ),
        )
        compare_source = "nrt" if compare_mode else "none"
    elif source_mode == "NRT API":
        compare_mode = False
    else:
        compare_mode = st.toggle("Compare two archive periods", value=False)
        compare_source = "archive" if compare_mode else "none"

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
        compare_start = compare_end = None
        if source_mode == "NRT API":
            end_date = st.date_input(
                "End date",
                value=default_end,
                min_value=nrt_min,
                max_value=today,
                key="nrt_end_date",
            )
            window_days = st.slider(
                "Window (days)",
                min_value=1,
                max_value=MAX_DAY_RANGE,
                value=MAX_DAY_RANGE,
                key="nrt_window_days",
            )
            start_date, end_date = _nrt_window_from_end(end_date, window_days)
            st.caption(
                f"Live window: **{start_date} → {end_date}** "
                f"({(end_date - start_date).days + 1} days)."
            )
            fetch_clicked = st.button(
                "Fetch live data",
                type="primary",
                use_container_width=True,
            )
        else:
            primary_range = st.date_input(
                "Archive dates",
                value=(default_start, default_end),
                min_value=archive_date_min,
                max_value=date_max,
                key="primary_dates_v3",
            )
            start_date, end_date = _parse_date_range(primary_range, default_end)

            if compare_source == "nrt":
                # Pick the live NRT window with the same date control style
                # as Archive dates (not a separate "days" slider).
                live_default_end = today
                live_default_start = today - timedelta(days=min(5, MAX_DAY_RANGE) - 1)
                live_range = st.date_input(
                    "Live dates (MAP_KEY)",
                    value=(live_default_start, live_default_end),
                    min_value=nrt_min,
                    max_value=today,
                    key="archive_nrt_live_dates",
                    help=(
                        "Select the live FIRMS window here (max "
                        f"{MAX_DAY_RANGE} days). Same date picker as Archive dates."
                    ),
                )
                compare_start, compare_end = _parse_date_range(
                    live_range, live_default_end
                )
                span = (compare_end - compare_start).days + 1
                if span > MAX_DAY_RANGE:
                    compare_start = compare_end - timedelta(days=MAX_DAY_RANGE - 1)
                    st.caption(
                        f"NRT limit: clipped to **{compare_start} → {compare_end}** "
                        f"({MAX_DAY_RANGE} days)."
                    )
                else:
                    st.caption(
                        f"Live compare: **{compare_start} → {compare_end}** "
                        f"({span} day{'s' if span != 1 else ''} via MAP_KEY)."
                    )
            elif compare_source == "archive":
                compare_range = st.date_input(
                    "Compare with",
                    value=(
                        max(archive_date_min or default_start, default_start - timedelta(days=30)),
                        max(archive_date_min or default_start, default_start - timedelta(days=1)),
                    ),
                    min_value=archive_date_min,
                    max_value=date_max,
                    key="compare_dates",
                )
                compare_start, compare_end = _parse_date_range(
                    compare_range, default_end
                )
            fetch_clicked = True

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
        "compare_source": compare_source,
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


@st.fragment
def render_map_focus_panel(
    *,
    bbox: tuple[float, float, float, float] | None,
    start: date,
    end: date,
    focus_start: date,
    focus_end: date,
    exact_total: int,
    harmonized_fallback: pd.DataFrame | None,
    use_archive_map: bool = True,
    show_focus_slider: bool = False,
) -> None:
    """
    Map only (optionally with a focus slider). Isolated so map redraws do not
    rebuild the activity chart or re-run the archive pipeline.
    """
    st.subheader("Map")
    st.caption("Matched · MODIS only · VIIRS only")

    span_days = max((end - start).days, 0)
    if show_focus_slider and span_days > 0:
        cur_start, cur_end = st.slider(
            "Date range (map focus)",
            min_value=start,
            max_value=end,
            value=(focus_start, focus_end),
            format="YYYY-MM-DD",
            key="chart_focus_slider",
        )
        cur_start, cur_end = _coerce_focus_dates(start, end, (cur_start, cur_end))
    else:
        cur_start, cur_end = focus_start, focus_end
        if span_days == 0:
            st.caption(f"Single day: {start}")
        else:
            st.caption(f"Map focus: {cur_start} → {cur_end}")
    st.session_state.chart_focus = (cur_start, cur_end)

    focused = harmonized_fallback if harmonized_fallback is not None else pd.DataFrame()
    map_stats: dict = {}
    if use_archive_map and bbox is not None:
        min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox)
        try:
            payload = load_map_slice(
                str(active_data_path()),
                _cached_mtimes_key(),
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                cur_start.isoformat(),
                cur_end.isoformat(),
                0,  # recount for this focus window (fast; avoids stale totals)
                _map_load_budget(bbox, known_total=0),
            )
            if payload.get("harmonized") is not None:
                focused = payload["harmonized"]
            map_stats = payload.get("stats") or {}
        except Exception:  # noqa: BLE001
            if (
                harmonized_fallback is not None
                and not harmonized_fallback.empty
                and "acq_datetime" in harmonized_fallback.columns
            ):
                mask = (
                    (harmonized_fallback["acq_datetime"] >= pd.Timestamp(cur_start))
                    & (
                        harmonized_fallback["acq_datetime"]
                        < pd.Timestamp(cur_end) + pd.Timedelta(days=1)
                    )
                )
                focused = harmonized_fallback.loc[mask]
                if focused.empty:
                    focused = harmonized_fallback
    elif (
        harmonized_fallback is not None
        and not harmonized_fallback.empty
        and "acq_datetime" in harmonized_fallback.columns
    ):
        # NRT / in-memory path: filter the fetched frame to the focus window
        mask = (
            (harmonized_fallback["acq_datetime"] >= pd.Timestamp(cur_start))
            & (
                harmonized_fallback["acq_datetime"]
                < pd.Timestamp(cur_end) + pd.Timedelta(days=1)
            )
        )
        focused = harmonized_fallback.loc[mask]
        if focused.empty:
            focused = harmonized_fallback

    map_n = int(len(focused)) if focused is not None else 0
    scanned_n = int(map_stats.get("rows_scanned") or exact_total or map_n)
    plot_cap = _map_point_limit(bbox)
    plotted = min(map_n, plot_cap)
    truncated = (
        bool(map_stats.get("truncated"))
        or scanned_n > MAP_FULL_THRESHOLD
        or map_n > plot_cap
    )
    if plotted:
        st.caption(
            f"{plotted:,} fires on the map for {cur_start} → {cur_end}"
            + (f" (of {scanned_n:,} in focus)" if scanned_n > plotted else "")
            + "."
        )

    map_fig = build_map_figure(
        focused if focused is not None else pd.DataFrame(),
        bbox=bbox,
    )
    st.plotly_chart(
        map_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
        },
        key="fire_map",
    )
    if truncated:
        st.caption(
            "Map shows a performance sample (Matched + high FRP first). "
            "Charts and totals use the full selection."
        )

    with st.expander("Map sample table", expanded=False):
        if focused is None or focused.empty:
            st.caption("No map points in this focus window.")
        else:
            show_cols = [
                c
                for c in ("latitude", "longitude", "acq_datetime", "category")
                if c in focused.columns
            ]
            st.dataframe(
                focused[show_cols].head(500),
                use_container_width=True,
                hide_index=True,
            )


def render_results(controls: dict) -> None:
    """Summary, activity chart, then map underneath with shared date focus."""
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

    # Full-range exact daily series from DuckDB (any Global / multi-year span)
    daily_series = st.session_state.get("daily_series")
    if isinstance(daily_series, pd.DataFrame) and not daily_series.empty:
        ds = daily_series.copy()
        ds["period"] = pd.to_datetime(ds["period"])
        if elevated_activity_last_days(ds, days=7):
            st.warning("Elevated fire activity detected")
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
        daily_focus = pd.DataFrame()

    anomalies = count_anomalies(daily_focus) if not daily_focus.empty else 0
    exact_total = (
        int(daily_focus["total"].sum())
        if not daily_focus.empty and "total" in daily_focus.columns
        else 0
    )
    exact_modis = (
        int(daily_focus[MODIS_ONLY].sum())
        if not daily_focus.empty and MODIS_ONLY in daily_focus.columns
        else 0
    )
    exact_viirs = (
        int(daily_focus[VIIRS_ONLY].sum())
        if not daily_focus.empty and VIIRS_ONLY in daily_focus.columns
        else 0
    )
    frp_sum = (
        float(daily_focus["frp_sum"].sum())
        if not daily_focus.empty and "frp_sum" in daily_focus.columns
        else float(meta.get("frp_sum") or 0.0)
    )
    frp_avg = (frp_sum / exact_total) if exact_total else 0.0
    quality = meta.get("harmonization_quality")
    if quality is None and harmonized is not None:
        quality = harmonization_quality(harmonized)

    if meta:
        st.caption(
            f"{meta.get('region', '—')}  ·  {start} → {end}  ·  "
            f"focus {focus_start} → {focus_end}"
        )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total", f"{exact_total:,}")
    c2.metric("FRP total", f"{frp_sum:,.0f} MW")
    c3.metric("FRP avg", f"{frp_avg:,.2f} MW")
    c4.metric("MODIS", f"{exact_modis:,}")
    c5.metric("VIIRS", f"{exact_viirs:,}")
    c6.metric(
        "Harmonization Quality",
        f"{quality:.0f}%" if quality is not None else "—",
    )

    cmp_meta = st.session_state.get("comparison_meta") or {}
    if cmp_meta.get("source") == "NRT API":
        st.divider()
        live_total = int(cmp_meta.get("raw_rows") or 0)
        if comparison is not None and not comparison.empty and live_total <= 0:
            live_total = int(len(comparison))
        render_archive_vs_live(
            archive_total=int(meta.get("raw_rows") or exact_total),
            archive_start=start,
            archive_end=end,
            live_total=live_total,
            live_start=cmp_meta.get("start_date") or start,
            live_end=cmp_meta.get("end_date") or end,
        )
    elif comparison is not None and not comparison.empty and harmonized is not None:
        st.divider()
        render_comparison(harmonized, comparison)

    if anomalies and not daily_focus.empty:
        anomaly_days = daily_focus.loc[daily_focus["is_anomaly"], "period"]
        labels = ", ".join(
            pd.Timestamp(d).strftime("%Y-%m-%d")
            for d in list(anomaly_days)[:12]
        )
        extra = len(anomaly_days) - 12
        if extra > 0:
            labels = f"{labels} (+{extra} more)"
        st.warning(f"Unusually busy days: {labels}")

    st.divider()

    # —— Activity chart first; map sits underneath ——
    st.subheader("Activity")
    head_l, head_r = st.columns((2.2, 1.0))
    with head_l:
        st.caption(
            "Exact totals from your uploaded archive. "
            "Long ranges auto-aggregate so the chart stays readable."
        )
    with head_r:
        span_days = max((end - start).days, 0)
        if span_days >= 365:
            detail_opts = ("Monthly", "Weekly")
            default_idx = 0
        elif span_days >= 90:
            detail_opts = ("Weekly", "Daily")
            default_idx = 0
        else:
            detail_opts = ("Daily", "Weekly")
            default_idx = 0
        freq_choice = st.radio(
            "Detail",
            options=detail_opts,
            horizontal=True,
            index=default_idx,
            key="chart_freq_v2",
        )

    span_days = max((end - start).days, 0)
    if span_days > 0:
        raw_focus = st.session_state.get("chart_focus", (start, end))
        # Default chart focus: full archive selection (not a tiny 10-day slice)
        if (
            st.session_state.get("_focus_data_key") == data_key
            and isinstance(raw_focus, (tuple, list))
            and len(raw_focus) == 2
        ):
            pass
        else:
            raw_focus = (start, end)
        focus_start, focus_end = st.slider(
            "Chart / map focus",
            min_value=start,
            max_value=end,
            value=_coerce_focus_dates(start, end, raw_focus),
            format="YYYY-MM-DD",
            key="chart_focus_slider",
        )
        focus_start, focus_end = _coerce_focus_dates(
            start, end, (focus_start, focus_end)
        )
        st.session_state.chart_focus = (focus_start, focus_end)
    else:
        focus_start, focus_end = start, end
        st.session_state.chart_focus = (focus_start, focus_end)

    if isinstance(daily_series, pd.DataFrame) and not daily_series.empty:
        ds = daily_series.copy()
        ds["period"] = pd.to_datetime(ds["period"])
        daily_focus = ds[
            (ds["period"] >= pd.Timestamp(focus_start))
            & (ds["period"] < pd.Timestamp(focus_end) + pd.Timedelta(days=1))
        ].copy()
        if daily_focus.empty:
            daily_focus = ds
        positive = daily_focus.loc[daily_focus["total"] > 0, "total"]
        mean_total = float(positive.mean()) if len(positive) else 0.0
        daily_focus["mean_total"] = mean_total
        daily_focus["threshold"] = mean_total * 2.0
        daily_focus["is_anomaly"] = (daily_focus["total"] > mean_total * 2.0) & (
            mean_total > 0
        )

    if daily_focus.empty:
        st.info("Not enough data for the chart.")
    else:
        chart_daily = daily_focus.copy()
        span_focus = max((focus_end - focus_start).days, 1)
        if freq_choice == "Monthly" or span_focus >= 700:
            tmp = chart_daily.set_index("period").sort_index()
            agg = {
                k: "sum"
                for k in ("total", "frp_sum", MATCHED, MODIS_ONLY, VIIRS_ONLY)
                if k in tmp.columns
            }
            chart_daily = tmp.resample("MS").agg(agg).dropna(how="all").reset_index()
            freq_label = "Monthly"
        elif freq_choice == "Weekly" or span_focus >= 90:
            tmp = chart_daily.set_index("period").sort_index()
            agg = {
                k: "sum"
                for k in ("total", "frp_sum", MATCHED, MODIS_ONLY, VIIRS_ONLY)
                if k in tmp.columns
            }
            chart_daily = tmp.resample("W-MON").agg(agg).dropna(how="all").reset_index()
            freq_label = "Weekly"
        else:
            freq_label = "Daily"
        positive = chart_daily.loc[chart_daily["total"] > 0, "total"]
        mean_total = float(positive.mean()) if len(positive) else 0.0
        chart_daily["mean_total"] = mean_total
        chart_daily["threshold"] = mean_total * 2.0
        chart_daily["is_anomaly"] = (chart_daily["total"] > chart_daily["threshold"]) & (
            mean_total > 0
        )
        chart_daily = _downsample_chart_daily(chart_daily)

        ts = chart_daily[
            [
                c
                for c in ("period", MATCHED, MODIS_ONLY, VIIRS_ONLY, "frp_sum")
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

    st.divider()
    # Map under the chart — uses the same focus window
    render_map_focus_panel(
        bbox=bbox if isinstance(bbox, tuple) else None,
        start=start,
        end=end,
        focus_start=focus_start,
        focus_end=focus_end,
        exact_total=exact_total,
        harmonized_fallback=harmonized,
        use_archive_map=meta.get("source") != "NRT API",
        show_focus_slider=False,
    )


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
        str(controls.get("compare_source") or "none"),
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

    if st.session_state.get("_archive_mtimes") is None:
        st.session_state._archive_mtimes = _archive_mtimes_key(active_data_path())
    mtimes = st.session_state._archive_mtimes
    start_iso = controls["start_date"].isoformat()
    end_iso = controls["end_date"].isoformat()

    try:
        _, rebuilt = ensure_hotspots_db(active_data_path())
    except ArchiveCsvError as exc:
        st.error(str(exc))
        return None
    if rebuilt:
        st.session_state._archive_mtimes = _archive_mtimes_key(active_data_path())
        mtimes = st.session_state._archive_mtimes
        st.toast("Archive index ready.")

    try:
        payload = load_archive_slice(
            str(active_data_path()),
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
        st.session_state.monthly_series = None
        st.session_state.archive_load_key = None
        st.session_state.comparison = None
        return None

    st.session_state.daily_series = daily
    st.session_state.monthly_series = payload.get("monthly")
    st.session_state.archive_load_key = load_key
    if stats.get("truncated") and total_count > len(primary):
        st.caption(
            f"Map sample {len(primary):,} of {total_count:,} points — "
            "chart uses the full selection."
        )

    # Keep session light — Plotly/fallback map never needs > MAX_MAP_RENDER rows
    st.session_state.harmonized = _sample_for_map(primary, MAX_MAP_RENDER)
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
        "frp_sum": float(stats.get("frp_sum") or 0.0),
        "frp_avg": float(stats.get("frp_avg") or 0.0),
        "harmonization_quality": stats.get("harmonization_quality"),
    }

    st.session_state.comparison = None
    st.session_state.comparison_meta = None

    if (
        controls["compare_mode"]
        and controls["compare_start"]
        and controls["compare_end"]
    ):
        cmp_source = controls.get("compare_source") or "archive"
        with st.spinner(
            "Fetching live FIRMS…"
            if cmp_source == "nrt"
            else "Loading comparison…"
        ):
            if cmp_source == "nrt":
                map_key = get_firms_map_key()
                if not guard_nrt_fetch(map_key):
                    cmp_payload = None
                else:
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
                        _record_nrt_fetch()
                        cmp_payload = {
                            "harmonized": secondary,
                            "raw_rows": int(len(secondary)),
                        }
                    except FirmsApiError as exc:
                        st.warning(str(exc))
                        cmp_payload = None
                    except Exception:  # noqa: BLE001
                        st.warning("Could not fetch live FIRMS for comparison.")
                        cmp_payload = None
            else:
                try:
                    cmp_payload = load_archive_slice(
                        str(active_data_path()),
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
            secondary = cmp_payload["harmonized"]
            st.session_state.comparison = _sample_for_map(
                secondary, MAX_MAP_RENDER
            )
            raw_n = int(
                cmp_payload.get("raw_rows")
                or cmp_payload.get("stats", {}).get("total_count")
                or len(secondary)
            )
            st.session_state.comparison_meta = {
                "start_date": controls["compare_start"],
                "end_date": controls["compare_end"],
                "raw_rows": raw_n,
                "source": "NRT API" if cmp_source == "nrt" else "Archive CSV",
            }

    return None


def run_nrt_pipeline(controls: dict, map_key: str) -> None:
    """Fetch NRT API data into session results."""
    if not guard_nrt_fetch(map_key):
        st.session_state.harmonized = None
        st.session_state.comparison = None
        return
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
            _record_nrt_fetch()
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

    daily = daily_hotspot_counts(
        primary,
        start=pd.Timestamp(controls["start_date"]),
        end=pd.Timestamp(controls["end_date"]),
    )
    if "frp" in primary.columns and not daily.empty:
        frp_daily = (
            primary.assign(_day=pd.to_datetime(primary["acq_datetime"]).dt.normalize())
            .groupby("_day", as_index=False)["frp"]
            .sum()
            .rename(columns={"_day": "period", "frp": "frp_sum"})
        )
        daily = daily.merge(frp_daily, on="period", how="left")
        daily["frp_sum"] = daily["frp_sum"].fillna(0.0)
    elif not daily.empty:
        daily["frp_sum"] = 0.0

    monthly = pd.DataFrame()
    if not daily.empty:
        tmp = daily.set_index("period").sort_index()
        monthly = (
            tmp.resample("MS")
            .agg({"total": "sum", "frp_sum": "sum"})
            .dropna(how="all")
            .reset_index()
        )
        prev_total = monthly["total"].shift(1)
        prev_frp = monthly["frp_sum"].shift(1)
        monthly["pct_change_total"] = (
            (monthly["total"] - prev_total) / prev_total.replace(0, pd.NA) * 100.0
        )
        monthly["pct_change_frp"] = (
            (monthly["frp_sum"] - prev_frp) / prev_frp.replace(0, pd.NA) * 100.0
        )

    frp_sum = float(primary["frp"].sum()) if "frp" in primary.columns else 0.0
    st.session_state.harmonized = primary
    st.session_state.daily_series = daily
    st.session_state.monthly_series = monthly
    st.session_state.last_meta = {
        "region": region_label,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "start_date": controls["start_date"],
        "end_date": controls["end_date"],
        "raw_rows": int(len(primary)),
        "source": "NRT API",
        "frp_sum": frp_sum,
        "frp_avg": (frp_sum / len(primary)) if len(primary) else 0.0,
        "harmonization_quality": harmonization_quality(primary),
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
    *,
    nrt_compare_available: bool = False,
) -> None:
    """
    Controls + results. Archive data reloads only when region/dates change;
    chart focus slider only re-filters in memory (no Global re-query).
    """
    controls = render_control_panel(
        "Archive CSV",
        archive_date_min=archive_date_min,
        archive_date_max=archive_date_max,
        nrt_compare_available=nrt_compare_available,
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
    st.caption(
        "Uploaded archive (e.g. 2023–2026) on the chart; "
        "optional live FIRMS window for comparison; map below."
    )

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None
        st.session_state.comparison = None
        st.session_state.comparison_meta = None
        st.session_state.nrt_fetch_times = []
        st.session_state.visitor_map_key = ""

    render_map_key_input()

    # Prefer session uploads when present; otherwise shared operator archive
    upload_dir = session_upload_dir()
    session_inventory = describe_csv_files(upload_dir)
    archive_root = upload_dir if session_inventory else DATA_PATH
    inventory = describe_csv_files(archive_root)
    archive_date_min, archive_date_max = peek_archive_date_bounds(archive_root)
    map_key = get_firms_map_key()
    has_archive = bool(inventory)
    has_nrt = bool(map_key)

    if not has_archive and not has_nrt:
        st.warning(
            "No fire archive found. Upload CSVs or enter a FIRMS MAP_KEY for live data."
        )
        with st.expander("Add archive files"):
            st.caption(
                f"Up to {MAX_UPLOAD_FILES} CSV files, "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MiB each. "
                "Saved only for this session."
            )
            uploads = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploads:
                try:
                    save_uploaded_csvs(upload_dir, uploads)
                except ArchiveCsvError as exc:
                    st.error(str(exc))
                    return
                load_archive_slice.clear()
                load_map_slice.clear()
                st.rerun()
        return

    st.session_state.archive_root = str(archive_root)

    if has_archive:
        if archive_date_min and archive_date_max:
            st.caption(
                f"Archive coverage: **{archive_date_min} → {archive_date_max}**"
                + (
                    f". Live compare uses MAP_KEY (last 5–{MAX_DAY_RANGE} days)."
                    if has_nrt
                    else "."
                )
            )
        st.divider()
        live_archive_explorer(
            archive_date_min,
            archive_date_max,
            nrt_compare_available=has_nrt,
        )
        return

    # NRT-only fallback when no CSVs are present
    st.caption(
        "No archive CSVs — live NRT only "
        f"(max {MAX_DAY_RANGE} days ending on the date you pick)."
    )
    st.divider()
    controls = render_control_panel(
        "NRT API",
        archive_date_min=archive_date_min,
        archive_date_max=archive_date_max,
    )
    if controls["fetch_clicked"]:
        run_nrt_pipeline(controls, map_key)
    if st.session_state.harmonized is not None:
        render_results(controls)


if __name__ == "__main__":
    main()
