"""
Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS ve VIIRS hot spot harmonizasyonu.
FIRMS Area API'den veri çeker; 1 km / ±6 saat ile eşleştirir; harita ve
aktivite takviminde gösterir.

FIRMS_MAP_KEY .env dosyasından okunur; loglanmaz ve Git'e eklenmez.
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

# Proje kökündeki .env — anahtar yalnızca ortam değişkeninde tutulur
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Unified Fire Activity Calendar",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_firms_map_key() -> str:
    """FIRMS_MAP_KEY'i ortamdan okur (değer loglanmaz)."""
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def build_map(df: pd.DataFrame) -> folium.Map:
    """Kategori renkleriyle Folium haritası oluşturur."""
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
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
    """Matched / MODIS-only / VIIRS-only çizgili aktivite grafiği."""
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
        title=f"Yangın Aktivitesi ({freq_label})",
        xaxis_title="Dönem",
        yaxis_title="Hot spot sayısı",
        legend_title="Kategori",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        template="plotly_white",
    )
    return fig


def render_sidebar() -> dict:
    """Kenar çubuğu: bölge, sensör, gün — MAP_KEY yok (.env'den okunur)."""
    with st.sidebar:
        st.header("FIRMS Ayarları")

        # Form dışında: anında UI değişimi (Hazır ↔ Manuel)
        st.subheader("Bölge")
        region_mode = st.radio(
            "Bölge seçimi",
            options=("Hazır bölge", "Manuel bbox"),
            horizontal=True,
        )

        with st.form("firms_fetch_form", clear_on_submit=False):
            if region_mode == "Hazır bölge":
                region_name = st.selectbox(
                    "Hazır bölgeler",
                    options=list(PRESET_REGIONS.keys()),
                    index=0,
                )
                min_lon, min_lat, max_lon, max_lat = PRESET_REGIONS[region_name]
                st.caption(
                    f"bbox: {min_lon}, {min_lat}, {max_lon}, {max_lat}"
                )
            else:
                region_name = "Manuel"
                c1, c2 = st.columns(2)
                with c1:
                    min_lon = st.number_input("min_lon", value=26.0, format="%.4f")
                    min_lat = st.number_input("min_lat", value=36.0, format="%.4f")
                with c2:
                    max_lon = st.number_input("max_lon", value=30.0, format="%.4f")
                    max_lat = st.number_input("max_lat", value=40.0, format="%.4f")

            sensors = st.multiselect(
                "Sensörler",
                options=list(AVAILABLE_SENSORS),
                default=list(AVAILABLE_SENSORS),
                help="Birden fazla sensör seçilebilir; sonuçlar birleştirilir.",
            )

            days = st.slider("Gün sayısı", min_value=1, max_value=10, value=3)

            fetch_clicked = st.form_submit_button(
                "Veriyi Getir",
                type="primary",
                use_container_width=True,
            )

        freq_choice = st.radio(
            "Takvim frekansı",
            options=("Günlük", "Haftalık"),
            horizontal=True,
        )

        st.divider()
        st.caption(
            "Kaynak: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) · "
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
        NASA FIRMS **MODIS** ve **VIIRS** sıcak noktalarını canlı olarak çeker;
        **1 km** mesafe ve **±6 saat** penceresinde birleştirir.
        Eşleşenler *Matched (High Confidence)*; diğerleri *MODIS-only* /
        *VIIRS-only* olarak haritada ve takvimde gösterilir.
        """
    )

    map_key = get_firms_map_key()
    if not map_key:
        st.warning(
            "`.env` dosyasına `FIRMS_MAP_KEY` ekleyin. "
            "Örnek: projede `.env.example` dosyasına bakın."
        )
        with st.expander(".env nasıl ayarlanır?"):
            st.markdown(
                """
                1. [FIRMS MAP Key](https://firms.modaps.eosdis.nasa.gov/api/key/) alın.
                2. Proje klasöründe `.env` oluşturun:
                ```
                FIRMS_MAP_KEY=senin_keyin_buraya
                ```
                3. Uygulamayı yeniden başlatın. `.env` Git'e eklenmez.
                """
            )
        return

    controls = render_sidebar()

    if "harmonized" not in st.session_state:
        st.session_state.harmonized = None
        st.session_state.last_meta = None

    if controls["fetch_clicked"]:
        if not controls["sensors"]:
            st.error("En az bir sensör seçin.")
            st.session_state.harmonized = None
        else:
            with st.spinner("FIRMS API'den veri getiriliyor…"):
                try:
                    # map_key yalnızca bu çağrıya verilir; loglanmaz / state'e yazılmaz
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
                except Exception:  # noqa: BLE001 — anahtar sızıntısını önle
                    st.error(
                        "Beklenmeyen bir hata oluştu. Bağlantınızı ve "
                        "parametreleri kontrol edip yeniden deneyin."
                    )
                    st.session_state.harmonized = None
                    return

            if raw.empty:
                st.warning(
                    "Seçilen bölge, sensörler ve gün aralığında hot spot bulunamadı. "
                    "Daha geniş bir bbox veya daha fazla gün deneyin."
                )
                st.session_state.harmonized = None
                return

            missing = validate_firms_columns(raw)
            if missing:
                st.error(
                    "FIRMS yanıtında eksik sütunlar: "
                    + ", ".join(missing)
                )
                st.session_state.harmonized = None
                return

            with st.spinner("Hot spotlar harmonize ediliyor…"):
                harmonized = harmonize_hotspots(raw)

            if harmonized.empty:
                st.warning(
                    "Geçerli MODIS/VIIRS satırı çıkarılamadı. "
                    "Yanıttaki satellite / instrument alanlarını kontrol edin."
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
            "Bölge ve sensörleri seçip **Veriyi Getir**’e basın. "
            "API anahtarı `.env` içindeki `FIRMS_MAP_KEY` ile yüklenir."
        )
        return

    meta = st.session_state.last_meta or {}
    if meta:
        st.caption(
            f"Bölge: **{meta.get('region', '—')}** · "
            f"Sensörler: {', '.join(meta.get('sensors', []))} · "
            f"Son {meta.get('days', '—')} gün · "
            f"Ham satır: {meta.get('raw_rows', '—')}"
        )

    min_d = harmonized["acq_datetime"].min().date()
    max_d = harmonized["acq_datetime"].max().date()
    date_range = st.date_input(
        "Tarih aralığı",
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
        st.warning("Seçilen tarih aralığında nokta yok.")
        return

    stats = summarize_categories(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam", stats["total"])
    c2.metric("Matched", stats["matched"])
    c3.metric("MODIS-only", stats["modis_only"])
    c4.metric("VIIRS-only", stats["viirs_only"])

    st.divider()

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Harmonize Harita")
        st.caption(
            "Yeşil = Matched (High Confidence) · "
            "Turuncu = MODIS-only · Mavi = VIIRS-only"
        )
        fmap = build_map(filtered)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Yangın Aktivitesi Takvimi")
        freq_choice = controls["freq_choice"]
        freq = "D" if freq_choice == "Günlük" else "W-MON"
        ts = activity_timeseries(
            filtered,
            freq=freq,
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
        )
        if ts.empty:
            st.info("Grafik için yeterli veri yok.")
        else:
            fig = build_activity_chart(ts, freq_choice)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Harmonize edilmiş tablo", expanded=False):
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
            "CSV indir",
            data=csv_bytes,
            file_name="harmonized_hotspots.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
