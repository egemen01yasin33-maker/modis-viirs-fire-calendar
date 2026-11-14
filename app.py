"""
Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.
Fetches FIRMS Area API data, matches within 1 km / ±6 hours, and shows
map + activity calendar views.

FIRMS_MAP_KEY is loaded from .env; it is never logged or committed.
"""

from __future__ import annotations

import os
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
)
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
    initial_sidebar_state="expanded",
)


def get_firms_map_key() -> str:
    """Read FIRMS_MAP_KEY from the environment (value is never logged)."""
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def build_map(df: pd.DataFrame) -> folium.Map:
    """Build a Folium map colored by harmonization category."""
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB dark_matter",
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


def render_sidebar() -> dict:
    """Sidebar: region, sensors, days — MAP_KEY comes from .env."""
    with st.sidebar:
        st.header("FIRMS Settings")

        st.subheader("Region")
        region_mode = st.radio(
            "Region mode",
            options=("Preset region", "Manual bbox"),
            horizontal=True,
        )

        with st.form("firms_fetch_form", clear_on_submit=False):
            if region_mode == "Preset region":
                region_name = st.selectbox(
                    "Preset regions",
                    options=list(PRESET_REGIONS.keys()),
                    index=0,
                )
                min_lon, min_lat, max_lon, max_lat = PRESET_REGIONS[region_name]
                st.caption(
                    f"bbox: {min_lon}, {min_lat}, {max_lon}, {max_lat}"
                )
            else:
                region_name = "Manual"
                c1, c2 = st.columns(2)
                with c1:
                    min_lon = st.number_input("min_lon", value=26.0, format="%.4f")
                    min_lat = st.number_input("min_lat", value=36.0, format="%.4f")
                with c2:
                    max_lon = st.number_input("max_lon", value=30.0, format="%.4f")
                    max_lat = st.number_input("max_lat", value=40.0, format="%.4f")

            sensors = st.multiselect(
                "Sensors",
                options=list(AVAILABLE_SENSORS),
                default=list(AVAILABLE_SENSORS),
                help="Select one or more sensors; results are combined.",
            )

            days = st.slider("Number of days", min_value=1, max_value=10, value=3)

            fetch_clicked = st.form_submit_button(
                "Fetch Data",
                type="primary",
                use_container_width=True,
            )

        freq_choice = st.radio(
            "Calendar frequency",
            options=("Daily", "Weekly"),
            horizontal=True,
        )

        st.divider()
        st.caption(
            "Source: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) · "
            "Space Apps 2026 – Hot Spot Harmonization"
        )

    return {
        "region_name": region_name,
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "sensors": sensors,
        "days": int(days),
        "freq_choice": freq_choice,
        "fetch_clicked": fetch_clicked,
    }


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

    controls = render_sidebar()

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None

    if controls["fetch_clicked"]:
        if not controls["sensors"]:
            st.error("Select at least one sensor.")
            st.session_state.harmonized = None
        else:
            with st.spinner("Fetching data from FIRMS…"):
                try:
                    raw = fetch_firms_area(
                        map_key=map_key,
                        sensors=controls["sensors"],
                        min_lon=controls["min_lon"],
                        min_lat=controls["min_lat"],
                        max_lon=controls["max_lon"],
                        max_lat=controls["max_lat"],
                        days=controls["days"],
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
                    "No hot spots found for this region, sensors, and day range. "
                    "Try a wider bbox or more days."
                )
                st.session_state.harmonized = None
                return

            missing = validate_firms_columns(raw)
            if missing:
                st.error(
                    "FIRMS response is missing columns: "
                    + ", ".join(missing)
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
                "region": controls["region_name"],
                "bbox": (
                    controls["min_lon"],
                    controls["min_lat"],
                    controls["max_lon"],
                    controls["max_lat"],
                ),
                "sensors": list(controls["sensors"]),
                "days": controls["days"],
                "raw_rows": int(len(raw)),
            }

    harmonized = st.session_state.harmonized
    if harmonized is None:
        st.info(
            "Choose a region and sensors, then press **Fetch Data**. "
            "The API key is loaded from `FIRMS_MAP_KEY` in `.env`."
        )
        return

    meta = st.session_state.last_meta or {}
    if meta:
        st.caption(
            f"Region: **{meta.get('region', '—')}** · "
            f"Sensors: {', '.join(meta.get('sensors', []))} · "
            f"Last {meta.get('days', '—')} days · "
            f"Raw rows: {meta.get('raw_rows', '—')}"
        )

    min_d = harmonized["acq_datetime"].min().date()
    max_d = harmonized["acq_datetime"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    mask = (
        (harmonized["acq_datetime"].dt.date >= start_date)
        & (harmonized["acq_datetime"].dt.date <= end_date)
    )
    filtered = harmonized.loc[mask].copy()
    if filtered.empty:
        st.warning("No points in the selected date range.")
        return

    stats = summarize_categories(filtered)

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
            "Teal = MODIS-only · Cyan = VIIRS-only"
        )
        fmap = build_map(filtered)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Fire Activity Calendar")
        freq_choice = controls["freq_choice"]
        freq = "D" if freq_choice == "Daily" else "W-MON"
        ts = activity_timeseries(
            filtered,
            freq=freq,
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
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
            if c in filtered.columns
        ]
        display = filtered[show_cols].sort_values("acq_datetime")
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv_bytes = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="harmonized_hotspots.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
