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

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

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
    list_archive_csv_files,
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


MAX_MAP_POINTS = 2_500


def _sample_for_map(df: pd.DataFrame, limit: int = MAX_MAP_POINTS) -> pd.DataFrame:
    """Keep the map responsive: prefer Matched points, then sample the rest."""
    if len(df) <= limit:
        return df
    matched = df[df["category"] == MATCHED]
    others = df[df["category"] != MATCHED]
    keep_matched = min(len(matched), max(limit // 3, 400))
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


def build_map(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
) -> folium.Map:
    """Folium map with OpenStreetMap tiles (no API key)."""
    plot_df = _sample_for_map(df)
    if plot_df.empty:
        center_lat, center_lon, zoom = 20.0, 0.0, 2
    elif bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        zoom = _map_zoom_for_bbox(min_lon, min_lat, max_lon, max_lat)
    else:
        center_lat = float(plot_df["latitude"].mean())
        center_lon = float(plot_df["longitude"].mean())
        zoom = 7

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        control_scale=True,
    )

    for category, color in CATEGORY_COLORS.items():
        subset = plot_df[plot_df["category"] == category]
        if subset.empty:
            continue
        full_n = int((df["category"] == category).sum()) if not df.empty else len(subset)
        layer = folium.FeatureGroup(name=f"{category} ({full_n})")
        # Vectorized-ish add via itertuples (faster than iterrows)
        for row in subset.itertuples(index=False):
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=5 if category == MATCHED else 4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=1,
                tooltip=str(category),
            ).add_to(layer)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def build_activity_chart(
    ts: pd.DataFrame,
    freq_label: str,
    daily: pd.DataFrame | None = None,
) -> go.Figure:
    """Category lines + optional anomaly markers on daily totals."""
    fig = go.Figure()
    for col, color in CATEGORY_COLORS.items():
        if col not in ts.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=ts["period"],
                y=ts[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
            )
        )

    if daily is not None and not daily.empty:
        fig.add_trace(
            go.Bar(
                x=daily["period"],
                y=daily["total"],
                name="Daily total",
                marker_color="rgba(0, 229, 212, 0.25)",
                opacity=0.55,
            )
        )
        anomalies = daily[daily["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies["period"],
                    y=anomalies["total"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        color=ANOMALY_COLOR,
                        size=14,
                        symbol="diamond",
                        line=dict(width=1, color="#ffffff"),
                    ),
                )
            )
            if "threshold" in daily.columns and len(daily):
                thr = float(daily["threshold"].iloc[0])
                fig.add_hline(
                    y=thr,
                    line_dash="dot",
                    line_color=ANOMALY_COLOR,
                    annotation_text="2× mean",
                    annotation_position="top left",
                )

    fig.update_layout(
        title=f"Fire Activity ({freq_label})",
        xaxis_title="Period",
        yaxis_title="Hot spot count",
        legend_title="Category",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
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


@st.cache_data(show_spinner=False)
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
    Stream archive CSVs, keep rows inside bbox + date window, then harmonize.

    Cached by folder mtimes and the filter key so Apply reuses prior slices.
    """
    raw, names, stats = load_filtered_csvs(
        Path(folder_str),
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start_iso,
        end=end_iso,
    )
    if raw.empty:
        harmonized = raw
    else:
        harmonized = harmonize_hotspots(raw)
    return {
        "raw_rows": int(len(raw)),
        "files": names,
        "harmonized": harmonized,
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

    col_region, col_dates, col_action = st.columns((1.5, 1.4, 0.7))

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
            key="primary_dates",
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

    with col_action:
        st.write("")
        st.write("")
        fetch_clicked = st.button(
            "Update map",
            type="primary",
            use_container_width=True,
        )

    freq_choice = st.radio(
        "Calendar",
        options=("Daily", "Weekly"),
        horizontal=True,
    )

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
        "freq_choice": freq_choice,
        "fetch_clicked": fetch_clicked,
    }


def render_results(controls: dict) -> None:
    """Summary, anomalies, map, calendar, optional comparison."""
    meta = st.session_state.last_meta or {}
    harmonized: pd.DataFrame = st.session_state.harmonized
    comparison: pd.DataFrame | None = st.session_state.get("comparison")

    start = meta.get("start_date")
    end = meta.get("end_date")
    bbox = meta.get("bbox")

    daily = daily_hotspot_counts(
        harmonized,
        start=pd.Timestamp(start) if start else None,
        end=pd.Timestamp(end) if end else None,
    )
    anomalies = count_anomalies(daily)
    stats = summarize_categories(harmonized)

    if meta:
        st.caption(f"{meta.get('region', '—')}  ·  {start} → {end}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{stats['total']:,}")
    c2.metric("Matched", f"{stats['matched']:,}")
    c3.metric("MODIS only", f"{stats['modis_only']:,}")
    c4.metric("VIIRS only", f"{stats['viirs_only']:,}")
    c5.metric("Busy days", anomalies)

    if comparison is not None and not comparison.empty:
        st.divider()
        render_comparison(harmonized, comparison)

    if anomalies and not daily.empty:
        anomaly_days = daily.loc[daily["is_anomaly"], "period"]
        labels = ", ".join(
            pd.Timestamp(d).strftime("%Y-%m-%d") for d in anomaly_days
        )
        st.warning(f"Unusually busy days: {labels}")

    st.divider()

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Map")
        st.caption("Matched · MODIS only · VIIRS only")
        fmap = build_map(harmonized, bbox=bbox)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Activity")
        freq_choice = controls["freq_choice"]
        freq = "D" if freq_choice == "Daily" else "W-MON"
        ts = activity_timeseries(
            harmonized,
            freq=freq,
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )
        if ts.empty and daily.empty:
            st.info("Not enough data for the chart.")
        else:
            fig = build_activity_chart(
                ts if not ts.empty else pd.DataFrame(columns=["period"]),
                freq_choice,
                daily=daily,
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Harmonized table", expanded=False):
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
                "firms_source",
                "archive_file",
            )
            if c in harmonized.columns
        ]
        display = harmonized[show_cols].sort_values("acq_datetime")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="harmonized_hotspots.csv",
            mime="text/csv",
        )

    if not daily.empty:
        with st.expander("Daily totals & anomalies", expanded=False):
            show = daily[
                ["period", "total", "is_anomaly", "mean_total", "threshold"]
            ]
            st.dataframe(show, use_container_width=True, hide_index=True)


def run_archive_pipeline(controls: dict) -> str | None:
    """
    Stream-filter archive CSVs by bbox + dates, harmonize, store in session.

    Returns a short banner string for the results area, or None on failure.
    """
    try:
        region_label, min_lon, min_lat, max_lon, max_lat = resolve_controls_bbox(
            controls
        )
    except FirmsApiError as exc:
        st.error(str(exc))
        return None

    mtimes = _archive_mtimes_key(DATA_PATH)
    start_iso = controls["start_date"].isoformat()
    end_iso = controls["end_date"].isoformat()

    with st.spinner("Updating map…"):
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
    if primary.empty:
        st.warning("No fires found for this region and date range.")
        st.session_state.harmonized = None
        st.session_state.comparison = None
        return None

    if stats.get("truncated"):
        st.warning(
            "Too many points for one view — try a smaller region or shorter dates."
        )

    st.session_state.harmonized = primary
    st.session_state.last_meta = {
        "region": region_label,
        "bbox": (min_lon, min_lat, max_lon, max_lat),
        "start_date": controls["start_date"],
        "end_date": controls["end_date"],
        "raw_rows": int(len(primary)),
        "source": "Archive CSV",
        "files": payload.get("files", []),
        "rows_scanned": int(stats.get("rows_scanned", 0)),
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

    results_area = st.container()
    st.divider()
    controls = render_control_panel(
        source_mode,
        archive_date_min=archive_date_min,
        archive_date_max=archive_date_max,
    )

    if source_mode == "Archive CSV":
        should_run = (
            controls["fetch_clicked"]
            or st.session_state.harmonized is None
            or (st.session_state.last_meta or {}).get("source") != "Archive CSV"
        )
        if should_run:
            run_archive_pipeline(controls)
    elif controls["fetch_clicked"]:
        run_nrt_pipeline(controls, get_firms_map_key())

    if st.session_state.harmonized is not None:
        with results_area:
            render_results(controls)


if __name__ == "__main__":
    main()
