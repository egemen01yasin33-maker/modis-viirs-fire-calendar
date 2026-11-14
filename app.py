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
    summary_sentence,
)
from src.archive_csv import (
    ArchiveCsvError,
    data_dir,
    filter_by_bbox,
    filter_by_dates,
    list_csv_files,
    load_and_merge_csvs,
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


def build_map(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
) -> folium.Map:
    """Folium map with OpenStreetMap tiles (no API key)."""
    if df.empty:
        center_lat, center_lon, zoom = 20.0, 0.0, 2
    elif bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        zoom = _map_zoom_for_bbox(min_lon, min_lat, max_lon, max_lat)
    else:
        center_lat = float(df["latitude"].mean())
        center_lon = float(df["longitude"].mean())
        zoom = 7

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    for category, color in CATEGORY_COLORS.items():
        subset = df[df["category"] == category]
        if subset.empty:
            continue
        layer = folium.FeatureGroup(name=f"{category} ({len(subset)})")
        for _, row in subset.iterrows():
            popup = (
                f"<b>{category}</b><br>"
                f"Sensor: {row.get('sensor', '')}<br>"
                f"Satellite: {row.get('satellite', '')}<br>"
                f"Time: {row.get('acq_datetime', '')}<br>"
                f"Confidence: {row.get('confidence', '')}<br>"
                f"Lat/Lon: {row['latitude']:.4f}, {row['longitude']:.4f}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=7 if category == MATCHED else 6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup, max_width=280),
                tooltip=f"{category} · {row.get('sensor', '')}",
            ).add_to(layer)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
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


@st.cache_data(show_spinner=False)
def load_archive_harmonized(folder_str: str, mtimes_key: tuple) -> dict:
    """
    Load + merge + harmonize all CSVs under data/.

    Cached by folder path and file mtimes so edits reload automatically.
    """
    folder = Path(folder_str)
    raw, names = load_and_merge_csvs(folder)
    harmonized = harmonize_hotspots(raw)
    return {
        "raw_rows": int(len(raw)),
        "files": names,
        "harmonized": harmonized,
        "mtimes_key": mtimes_key,
    }


def _archive_mtimes_key(folder: Path) -> tuple:
    files = list_csv_files(folder)
    return tuple((p.name, p.stat().st_mtime_ns) for p in files)


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


def render_control_panel(source_mode: str) -> dict:
    """Bottom panel: region + dates (+ NRT-only limits)."""
    st.header("FIRMS Settings")

    today = date.today()
    default_start = today - timedelta(days=2)
    compare_default_start = today - timedelta(days=9)
    compare_default_end = today - timedelta(days=7)

    # Archive data can span years — use a wider default when possible
    if source_mode == "Archive CSV":
        default_start = today - timedelta(days=30)

    compare_mode = st.toggle(
        "Comparison mode",
        value=False,
        help="Compare two date ranges for the same region.",
    )

    col_region, col_dates, col_action = st.columns((1.5, 1.4, 0.7))

    with col_region:
        region_mode = st.radio(
            "Region",
            options=("Preset region", "Custom place", "Manual bbox"),
            horizontal=True,
        )
        place_query = ""
        min_lon = min_lat = max_lon = max_lat = None
        region_name = "Custom"

        if region_mode == "Preset region":
            region_name = st.selectbox(
                "Preset regions",
                options=list(PRESET_REGIONS.keys()),
                index=list(PRESET_REGIONS.keys()).index("Turkey")
                if "Turkey" in PRESET_REGIONS
                else 0,
            )
            min_lon, min_lat, max_lon, max_lat = PRESET_REGIONS[region_name]
            st.caption(f"bbox: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
        elif region_mode == "Custom place":
            place_query = st.text_input(
                "City or region name",
                placeholder="e.g. Ankara, Istanbul, Izmir",
            )
            region_name = (place_query or "").strip() or "Custom"
        else:
            region_name = "Manual bbox"
            b1, b2 = st.columns(2)
            with b1:
                min_lon = st.number_input("min_lon", value=-10.0, format="%.4f")
                min_lat = st.number_input("min_lat", value=30.0, format="%.4f")
            with b2:
                max_lon = st.number_input("max_lon", value=40.0, format="%.4f")
                max_lat = st.number_input("max_lat", value=45.0, format="%.4f")

    with col_dates:
        help_dates = (
            f"FIRMS NRT supports up to {MAX_DAY_RANGE} days per request."
            if source_mode == "NRT API"
            else "Filter the loaded archive CSVs by acquisition date."
        )
        primary_range = st.date_input(
            "Primary date range",
            value=(default_start, today),
            max_value=today,
            help=help_dates,
            key="primary_dates",
        )
        start_date, end_date = _parse_date_range(primary_range, today)
        span = (end_date - start_date).days + 1
        if source_mode == "NRT API":
            st.caption(f"Primary: {span} day(s) · max {MAX_DAY_RANGE}")
        else:
            st.caption(f"Primary: {span} day(s)")

        compare_start = compare_end = None
        if compare_mode:
            compare_range = st.date_input(
                "Comparison date range",
                value=(compare_default_start, compare_default_end),
                max_value=today,
                key="compare_dates",
            )
            compare_start, compare_end = _parse_date_range(compare_range, today)

    with col_action:
        st.write("")
        st.write("")
        action_label = (
            "Apply / Refresh" if source_mode == "Archive CSV" else "Fetch Data"
        )
        fetch_clicked = st.button(
            action_label,
            type="primary",
            use_container_width=True,
        )

    freq_choice = st.radio(
        "Calendar frequency",
        options=("Daily", "Weekly"),
        horizontal=True,
    )

    st.caption(
        "Sensors: MODIS_NRT + VIIRS_NOAA20_NRT + VIIRS_NOAA21_NRT (automatic) · "
        "Map: OpenStreetMap · "
        "[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) · Space Apps 2026"
    )

    return {
        "region_mode": region_mode,
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


def render_results(
    controls: dict,
    *,
    source_banner: str | None = None,
) -> None:
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

    if source_banner:
        st.info(source_banner)
    else:
        st.success(
            summary_sentence(stats["total"], stats["matched"], anomalies)
        )

    if meta:
        st.caption(
            f"Region: **{meta.get('region', '—')}** · "
            f"{start} → {end} · "
            f"Source: {meta.get('source', '—')} · "
            f"Rows: {meta.get('raw_rows', '—')}"
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", stats["total"])
    c2.metric("Matched", stats["matched"])
    c3.metric("MODIS-only", stats["modis_only"])
    c4.metric("VIIRS-only", stats["viirs_only"])
    c5.metric("Anomalies", anomalies)

    if comparison is not None and not comparison.empty:
        st.divider()
        render_comparison(harmonized, comparison)
        cmp_meta = st.session_state.get("comparison_meta") or {}
        st.caption(
            f"Comparison window: {cmp_meta.get('start_date', '—')} → "
            f"{cmp_meta.get('end_date', '—')}"
        )

    if anomalies and not daily.empty:
        anomaly_days = daily.loc[daily["is_anomaly"], "period"]
        labels = ", ".join(
            pd.Timestamp(d).strftime("%Y-%m-%d") for d in anomaly_days
        )
        st.warning(f"Anomaly days (> 2× mean daily total): {labels}")

    st.divider()

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Harmonized Map")
        st.caption(
            "Turquoise = Matched · Teal = MODIS-only · Cyan = VIIRS-only · "
            "Tiles: OpenStreetMap"
        )
        fmap = build_map(harmonized, bbox=bbox)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Fire Activity Calendar")
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


def _slice_archive(
    full: pd.DataFrame,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Apply bbox + date filters to a harmonized archive frame."""
    scoped = filter_by_bbox(full, min_lon, min_lat, max_lon, max_lat)
    return filter_by_dates(scoped, start_date, end_date)


def run_archive_pipeline(controls: dict, archive_payload: dict) -> None:
    """Filter cached archive harmonization into session results."""
    try:
        region_label, min_lon, min_lat, max_lon, max_lat = resolve_controls_bbox(
            controls
        )
    except FirmsApiError as exc:
        st.error(str(exc))
        return

    full: pd.DataFrame = archive_payload["harmonized"]
    if full.empty:
        st.warning("Archive CSVs loaded but produced no valid MODIS/VIIRS rows.")
        st.session_state.harmonized = None
        return

    primary = _slice_archive(
        full,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        controls["start_date"],
        controls["end_date"],
    )
    if primary.empty:
        st.warning(
            "No archive hot spots in this region / date range. "
            "Widen the filters or check your CSV dates."
        )
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
        "source": "Archive CSV",
        "files": archive_payload.get("files", []),
    }

    st.session_state.comparison = None
    st.session_state.comparison_meta = None

    if (
        controls["compare_mode"]
        and controls["compare_start"]
        and controls["compare_end"]
    ):
        secondary = _slice_archive(
            full,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            controls["compare_start"],
            controls["compare_end"],
        )
        if secondary.empty:
            st.warning("No archive hot spots in the comparison date range.")
        else:
            st.session_state.comparison = secondary
            st.session_state.comparison_meta = {
                "start_date": controls["compare_start"],
                "end_date": controls["compare_end"],
                "raw_rows": int(len(secondary)),
            }


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

    with st.spinner("Fetching primary period from FIRMS NRT…"):
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
            st.error(
                "An unexpected error occurred. Check your connection "
                "and parameters, then try again."
            )
            st.session_state.harmonized = None
            st.session_state.comparison = None
            return

    if primary.empty:
        st.warning("No hot spots found for this region and primary date range.")
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
        with st.spinner("Fetching comparison period from FIRMS NRT…"):
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
            except FirmsApiError as exc:
                st.error(f"Comparison fetch failed: {exc}")
                secondary = pd.DataFrame()
            except Exception:  # noqa: BLE001
                st.error("Comparison fetch failed unexpectedly.")
                secondary = pd.DataFrame()

        if secondary.empty:
            st.warning("No hot spots in the comparison date range.")
        else:
            st.session_state.comparison = secondary
            st.session_state.comparison_meta = {
                "start_date": controls["compare_start"],
                "end_date": controls["compare_end"],
                "raw_rows": int(len(secondary)),
            }


def main() -> None:
    st.title("Unified Fire Activity Calendar")
    st.markdown(
        """
        Harmonize MODIS / VIIRS hot spots (**1 km**, **±6 hours**) from
        **FIRMS Archive CSVs** in `data/` or live **NRT API** pulls.
        Anomaly detection flags days above **2×** mean activity.
        """
    )

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None
        st.session_state.comparison = None
        st.session_state.comparison_meta = None

    source_mode = st.radio(
        "Data source",
        options=("Archive CSV", "NRT API"),
        horizontal=True,
        help=(
            "Archive CSV: load every .csv under data/. "
            "NRT API: live FIRMS Area API (needs FIRMS_MAP_KEY in .env)."
        ),
    )

    archive_payload = None
    archive_banner = None

    if source_mode == "Archive CSV":
        st.markdown(
            f"**CSV folder:** `{DATA_PATH}` — drop files below or copy them "
            f"into this `data/` directory."
        )
        uploads = st.file_uploader(
            "Upload FIRMS Archive CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help=(
                "Required columns: latitude, longitude, acq_date, "
                "acq_time, confidence, satellite"
            ),
        )
        if uploads:
            saved = save_uploaded_csvs(DATA_PATH, uploads)
            load_archive_harmonized.clear()
            st.success(
                "Saved to data/: " + ", ".join(saved) + ". Reloading…"
            )
            st.rerun()

        csv_files = list_csv_files(DATA_PATH)
        if not csv_files:
            st.warning(
                "No CSV files in data/ yet. "
                "Use the uploader above, or put files in the data/ folder."
            )
            st.info(
                "Required columns: "
                "`latitude, longitude, acq_date, acq_time, confidence, satellite`."
            )
            return

        try:
            mtimes = _archive_mtimes_key(DATA_PATH)
            with st.spinner("Loading and harmonizing archive CSVs…"):
                archive_payload = load_archive_harmonized(
                    str(DATA_PATH), mtimes
                )
        except ArchiveCsvError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load archive CSVs: {exc}")
            return

        full = archive_payload["harmonized"]
        stats_all = summarize_categories(full)
        daily_all = daily_hotspot_counts(full)
        anomalies_all = count_anomalies(daily_all)
        archive_banner = (
            f"Loaded {archive_payload['raw_rows']} records from data/ folder, "
            f"{stats_all['matched']} matched, "
            f"{anomalies_all} anomalies."
        )
        st.caption(
            "Files: " + ", ".join(archive_payload["files"])
        )
    else:
        map_key = get_firms_map_key()
        if not map_key:
            st.warning(
                "NRT mode needs `FIRMS_MAP_KEY` in your `.env` file. "
                "Or switch to Archive CSV and drop files into data/."
            )
            return

    results_area = st.container()
    st.divider()
    controls = render_control_panel(source_mode)

    # Archive: auto-apply filters on first load and when Apply is clicked
    if source_mode == "Archive CSV" and archive_payload is not None:
        should_run = (
            controls["fetch_clicked"]
            or st.session_state.harmonized is None
            or (st.session_state.last_meta or {}).get("source") != "Archive CSV"
        )
        if should_run:
            run_archive_pipeline(controls, archive_payload)

    if source_mode == "NRT API" and controls["fetch_clicked"]:
        run_nrt_pipeline(controls, get_firms_map_key())

    if st.session_state.harmonized is not None:
        with results_area:
            # Prefer the required archive banner wording when in CSV mode
            banner = None
            if source_mode == "Archive CSV" and archive_banner:
                view_stats = summarize_categories(st.session_state.harmonized)
                view_daily = daily_hotspot_counts(
                    st.session_state.harmonized,
                    start=pd.Timestamp(
                        (st.session_state.last_meta or {}).get("start_date")
                    )
                    if (st.session_state.last_meta or {}).get("start_date")
                    else None,
                    end=pd.Timestamp(
                        (st.session_state.last_meta or {}).get("end_date")
                    )
                    if (st.session_state.last_meta or {}).get("end_date")
                    else None,
                )
                view_anom = count_anomalies(view_daily)
                # Banner uses full folder load counts as requested
                banner = archive_banner
                # Also show filtered-view summary underneath via metrics
                _ = (view_stats, view_anom)
            render_results(controls, source_banner=banner)


if __name__ == "__main__":
    main()
