"""
Unified Fire Activity Calendar – Harmonized MODIS & VIIRS

NASA Space Apps: FIRMS CSV yükle → 1 km / ±6 saat ile harmonize et →
Folium harita + günlük/haftalık aktivite grafiği + özet istatistikler.
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

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
SAMPLE_CSV = ROOT / "data" / "sample_firms.csv"

st.set_page_config(
    page_title="Unified Fire Activity Calendar",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_csv(uploaded) -> pd.DataFrame:
    """Yüklenen veya örnek CSV'yi DataFrame olarak okur."""
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return pd.read_csv(SAMPLE_CSV)


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
                f"Sensor: {row['sensor']}<br>"
                f"Satellite: {row['satellite']}<br>"
                f"Time: {row['acq_datetime']}<br>"
                f"Confidence: {row['confidence']}<br>"
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
                tooltip=f"{category} · {row['sensor']}",
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


def main() -> None:
    st.title("Unified Fire Activity Calendar – Harmonized MODIS & VIIRS")
    st.markdown(
        """
        NASA FIRMS formatındaki MODIS ve VIIRS sıcak noktalarını **1 km** mesafe
        ve **±6 saat** zaman penceresinde birleştirir. Eşleşenler
        *Matched (High Confidence)* olarak işaretlenir; tek sensör tespitleri
        haritada ve takvimde ayrı renklerle gösterilir.
        """
    )

    # --- Kenar çubuğu ---
    with st.sidebar:
        st.header("Veri & Ayarlar")
        uploaded = st.file_uploader(
            "FIRMS CSV yükle",
            type=["csv"],
            help=(
                "Gerekli sütunlar: latitude, longitude, acq_date, "
                "acq_time, confidence, satellite"
            ),
        )
        use_sample = st.checkbox(
            "Örnek veriyi kullan",
            value=uploaded is None,
            help="Yükleme yoksa Ege/Akdeniz örnek hot spotları yüklenir.",
        )
        st.divider()
        distance_km = st.slider("Mesafe eşiği (km)", 0.5, 5.0, 1.0, 0.1)
        time_hours = st.slider("Zaman penceresi (± saat)", 1.0, 12.0, 6.0, 0.5)
        freq_choice = st.radio(
            "Takvim frekansı",
            options=("Günlük", "Haftalık"),
            horizontal=True,
        )
        st.caption(
            "Kaynak: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) · "
            "Space Apps – Hot Spot Harmonization"
        )

    # --- Veri yükleme ---
    if uploaded is None and not use_sample:
        st.info("Başlamak için bir FIRMS CSV yükleyin veya örnek veriyi açın.")
        return

    try:
        raw = load_csv(uploaded if uploaded is not None else None)
    except Exception as exc:  # noqa: BLE001 — kullanıcıya göster
        st.error(f"CSV okunamadı: {exc}")
        return

    missing = validate_firms_columns(raw)
    if missing:
        st.error("Eksik sütunlar: " + ", ".join(missing))
        st.write("Beklenen sütunlar:", ", ".join(
            [
                "latitude",
                "longitude",
                "acq_date",
                "acq_time",
                "confidence",
                "satellite",
            ]
        ))
        return

    with st.spinner("Hot spotlar harmonize ediliyor…"):
        harmonized = harmonize_hotspots(
            raw,
            distance_km=distance_km,
            time_window_hours=time_hours,
        )

    if harmonized.empty:
        st.warning(
            "Geçerli MODIS/VIIRS satırı bulunamadı. "
            "`satellite` alanının Terra/Aqua (MODIS) veya N/N20/N21 (VIIRS) "
            "içerdiğinden emin olun."
        )
        return

    # --- Tarih aralığı filtresi ---
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

    # --- İstatistikler ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam", stats["total"])
    c2.metric("Matched", stats["matched"])
    c3.metric("MODIS-only", stats["modis_only"])
    c4.metric("VIIRS-only", stats["viirs_only"])

    st.divider()

    # --- Harita + takvim ---
    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        st.subheader("Harmonize Harita")
        st.caption(
            "🟢 Matched (High Confidence) · 🟠 MODIS-only · 🔵 VIIRS-only"
        )
        fmap = build_map(filtered)
        st_folium(fmap, width=None, height=480, returned_objects=[])

    with right:
        st.subheader("Yangın Aktivitesi Takvimi")
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

    # --- Tablo ---
    with st.expander("Harmonize edilmiş tablo", expanded=False):
        show_cols = [
            "latitude",
            "longitude",
            "acq_datetime",
            "confidence",
            "satellite",
            "sensor",
            "category",
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
