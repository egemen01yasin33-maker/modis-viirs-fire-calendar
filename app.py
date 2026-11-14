"""
Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.
Fetches FIRMS Area API data, matches within 1 km / ±6 hours, and shows
map + activity calendar views.

FIRMS_MAP_KEY is loaded from .env; it is never logged or committed.
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

from src.firms_api import (
    AVAILABLE_SENSORS,
    PRESET_REGIONS,
    FirmsApiError,
    fetch_firms_area,
    resolve_place_bbox,
)

# NRT day-window limit (also defined in firms_api; keep a local fallback)
try:
    from src.firms_api import MAX_DAY_RANGE
except ImportError:  # pragma: no cover — older firms_api without the constant
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


def build_map(df: pd.DataFrame) -> folium.Map:
    """Build a Folium map colored by harmonization category (OpenStreetMap)."""
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
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


def build_activity_chart(ts: pd.DataFrame, freq_label: str) -> go.Figure:
    """Line chart of Matched / MODIS-only / VIIRS-only activity."""
    fig = go.Figure()
    for col, color in CATEGORY_COLORS.items():
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
    fig.update_layout(
        title=f"Fire Activity ({freq_label})",
        xaxis_title="Period",
        yaxis_title="Hot spot count",
        legend_title="Category",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_control_panel() -> dict:
    """Bottom panel: region + date range. Sensors are always all three NRT."""
    st.header("FIRMS Settings")

    today = date.today()
    default_start = today - timedelta(days=2)

    col_region, col_dates, col_action = st.columns((1.4, 1.2, 0.7))

    with col_region:
        region_mode = st.radio(
            "Region",
            options=("Preset region", "Custom place"),
            horizontal=True,
        )
        if region_mode == "Preset region":
            region_name = st.selectbox(
                "Preset regions",
                options=list(PRESET_REGIONS.keys()),
                index=0,
            )
            min_lon, min_lat, max_lon, max_lat = PRESET_REGIONS[region_name]
            place_query = ""
            st.caption(f"bbox: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
        else:
            place_query = st.text_input(
                "City or region name",
                placeholder="e.g. Ankara, Istanbul, Izmir",
                help="Uses a built-in bbox for known places in Turkey.",
            )
            region_name = (place_query or "").strip() or "Custom"
            min_lon = min_lat = max_lon = max_lat = None

    with col_dates:
        date_range = st.date_input(
            "Date range",
            value=(default_start, today),
            max_value=today,
            help=f"FIRMS NRT supports up to {MAX_DAY_RANGE} days.",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range if isinstance(date_range, date) else today

        span = (end_date - start_date).days + 1
        st.caption(f"{span} day(s) selected · max {MAX_DAY_RANGE} for NRT")

    with col_action:
        st.write("")
        st.write("")
        fetch_clicked = st.button(
            "Fetch Data",
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
        "Source: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) · "
        "Space Apps 2026"
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
        "freq_choice": freq_choice,
        "fetch_clicked": fetch_clicked,
    }


def resolve_controls_bbox(controls: dict) -> tuple[str, float, float, float, float]:
    """Return (region_label, min_lon, min_lat, max_lon, max_lat)."""
    if controls["region_mode"] == "Preset region":
        name = controls["region_name"]
        return name, *PRESET_REGIONS[name]

    bbox = resolve_place_bbox(controls["place_query"])
    label = (controls["place_query"] or "").strip().title() or "Custom"
    return label, *bbox


def render_results(harmonized: pd.DataFrame, controls: dict) -> None:
    """Map, calendar, and table for the current harmonized dataset."""
    meta = st.session_state.last_meta or {}
    if meta:
        st.caption(
            f"Region: **{meta.get('region', '—')}** · "
            f"{meta.get('start_date', '—')} → {meta.get('end_date', '—')} · "
            f"Raw rows: {meta.get('raw_rows', '—')}"
        )

    stats = summarize_categories(harmonized)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", stats["total"])
    c2.metric("Matched", stats["matched"])
    c3.metric("MODIS-only", stats["modis_only"])
    c4.metric("VIIRS-only", stats["viirs_only"])

    st.divider()

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Harmonized Map")
        st.caption(
            "Turquoise = Matched (High Confidence) · "
            "Teal = MODIS-only · Cyan = VIIRS-only · Tiles: OpenStreetMap"
        )
        fmap = build_map(harmonized)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Fire Activity Calendar")
        freq_choice = controls["freq_choice"]
        freq = "D" if freq_choice == "Daily" else "W-MON"
        start = meta.get("start_date")
        end = meta.get("end_date")
        ts = activity_timeseries(
            harmonized,
            freq=freq,
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )
        if ts.empty:
            st.info("Not enough data for the chart.")
        else:
            fig = build_activity_chart(ts, freq_choice)
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
            )
            if c in harmonized.columns
        ]
        display = harmonized[show_cols].sort_values("acq_datetime")
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="harmonized_hotspots.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("Unified Fire Activity Calendar")
    st.markdown(
        """
        Pulls live NASA FIRMS **MODIS** and **VIIRS** hot spots and merges
        detections within **1 km** and **±6 hours**. Matches are labeled
        *Matched (High Confidence)*; the rest appear as *MODIS-only* /
        *VIIRS-only* on the map and calendar.
        """
    )

    map_key = get_firms_map_key()
    if not map_key:
        st.warning(
            "Add `FIRMS_MAP_KEY` to your `.env` file. "
            "See `.env.example` in the project root."
        )
        with st.expander("How to set up .env"):
            st.markdown(
                """
                1. Get a free key from
                   [FIRMS MAP Key](https://firms.modaps.eosdis.nasa.gov/api/key/).
                2. Create a `.env` file in the project folder:
                ```
                FIRMS_MAP_KEY=your_key_here
                ```
                3. Restart the app. `.env` is gitignored and never committed.
                """
            )
        return

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None

    results_area = st.container()

    st.divider()
    controls = render_control_panel()

    if controls["fetch_clicked"]:
        try:
            region_label, min_lon, min_lat, max_lon, max_lat = resolve_controls_bbox(
                controls
            )
        except FirmsApiError as exc:
            st.error(str(exc))
            st.session_state.harmonized = None
            return

        with st.spinner("Fetching data from FIRMS…"):
            try:
                raw = fetch_firms_area(
                    map_key=map_key,
                    min_lon=min_lon,
                    min_lat=min_lat,
                    max_lon=max_lon,
                    max_lat=max_lat,
                    start_date=controls["start_date"],
                    end_date=controls["end_date"],
                    sensors=AVAILABLE_SENSORS,
                )
            except FirmsApiError as exc:
                st.error(str(exc))
                st.session_state.harmonized = None
                return
            except Exception:  # noqa: BLE001 — avoid leaking the key
                st.error(
                    "An unexpected error occurred. Check your connection "
                    "and parameters, then try again."
                )
                st.session_state.harmonized = None
                return

        if raw.empty:
            st.warning(
                "No hot spots found for this region and date range. "
                "Try a wider region or different dates."
            )
            st.session_state.harmonized = None
            return

        missing = validate_firms_columns(raw)
        if missing:
            st.error(
                "FIRMS response is missing columns: " + ", ".join(missing)
            )
            st.session_state.harmonized = None
            return

        with st.spinner("Harmonizing hot spots…"):
            harmonized = harmonize_hotspots(raw)

        if harmonized.empty:
            st.warning(
                "No valid MODIS/VIIRS rows could be parsed. "
                "Check the satellite / instrument fields in the response."
            )
            st.session_state.harmonized = None
            return

        st.session_state.harmonized = harmonized
        st.session_state.last_meta = {
            "region": region_label,
            "bbox": (min_lon, min_lat, max_lon, max_lat),
            "start_date": controls["start_date"],
            "end_date": controls["end_date"],
            "raw_rows": int(len(raw)),
        }

    harmonized = st.session_state.harmonized
    if harmonized is not None:
        with results_area:
            render_results(harmonized, controls)


if __name__ == "__main__":
    main()
