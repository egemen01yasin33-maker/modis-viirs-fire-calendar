"""ASGI entry for Cursor publish host (and uvicorn).

HTTP-only FastAPI UI — Streamlit needs WebSockets that Cursor Apps
do not upgrade, so the public site uses this stack instead.

Local Streamlit (optional):  streamlit run streamlit_app.py
Hosted / ASGI:               uvicorn app:app --host 0.0.0.0 --port ${PORT:-8529}
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from folium.plugins import MarkerCluster

from src.firms_api import (
    AVAILABLE_SENSORS,
    MAX_DAY_RANGE,
    PRESET_REGIONS,
    FirmsApiError,
    fetch_firms_area,
)
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

TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))
SAMPLE_CSV = ROOT / "data" / "sample_firms.csv"

# Simple process-wide rate limit for public FIRMS use of the host key
_NRT_HITS: list[float] = []
_NRT_MAX_PER_HOUR = 30
_NRT_MIN_GAP_SEC = 8.0

app = FastAPI(title="Unified Fire Activity Calendar")
# Cursor publish also looks for these aliases
application = app
handler = app


def _firms_key() -> str:
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def _nrt_allowed() -> tuple[bool, str]:
    now = time.monotonic()
    # prune
    while _NRT_HITS and now - _NRT_HITS[0] > 3600:
        _NRT_HITS.pop(0)
    if len(_NRT_HITS) >= _NRT_MAX_PER_HOUR:
        return False, "Live FIRMS rate limit reached for this hour. Try sample data."
    if _NRT_HITS and (now - _NRT_HITS[-1]) < _NRT_MIN_GAP_SEC:
        wait = int(_NRT_MIN_GAP_SEC - (now - _NRT_HITS[-1])) + 1
        return False, f"Please wait {wait}s before another live FIRMS request."
    return True, ""


def _record_nrt() -> None:
    _NRT_HITS.append(time.monotonic())


def _load_sample() -> pd.DataFrame:
    if not SAMPLE_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_CSV)


def _build_map(df: pd.DataFrame) -> str:
    if df.empty:
        m = folium.Map(location=[38.0, 35.0], zoom_start=5, tiles="OpenStreetMap")
        return m._repr_html_()

    lat0 = float(df["latitude"].mean())
    lon0 = float(df["longitude"].mean())
    # OpenStreetMap tiles — no third-party API key required on public demos
    m = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        cat = str(row.get("category") or "")
        color = CATEGORY_COLORS.get(cat, "#88aabb")
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{cat}</b><br>{row.get('acq_date', '')} {row.get('acq_time', '')}",
                max_width=220,
            ),
        ).add_to(cluster)
    return m._repr_html_()


def _build_chart(series: pd.DataFrame) -> str:
    if series is None or series.empty:
        return "<p class='muted'>No activity series.</p>"
    fig = go.Figure()
    x = series["period"].astype(str)
    for col, label in (
        (MATCHED, "Matched"),
        (MODIS_ONLY, "MODIS-only"),
        (VIIRS_ONLY, "VIIRS-only"),
    ):
        if col in series.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=series[col],
                    name=label,
                    line=dict(color=CATEGORY_COLORS[col], width=2),
                )
            )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#070b10",
        plot_bgcolor="#0e151c",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.12),
        height=360,
        font=dict(color="#e8f1f4"),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _run_pipeline(raw: pd.DataFrame) -> dict:
    missing = validate_firms_columns(raw)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    harm = harmonize_hotspots(raw)
    summary = summarize_categories(harm)
    series = activity_timeseries(harm)
    return {
        "harmonized": harm,
        "summary": summary,
        "series": series,
        "map_html": _build_map(harm),
        "chart_html": _build_chart(series),
        "rows": int(len(harm)),
    }


def _page_context(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    mode: str = "sample",
    region: str = "Turkey",
    days: int = 5,
) -> dict:
    today = date.today()
    return {
        "request": request,
        "presets": list(PRESET_REGIONS.keys()),
        "region": region,
        "days": days,
        "has_key": bool(_firms_key()),
        "mode": mode,
        "error": error,
        "result": result,
        "today": today.isoformat(),
        "max_days": MAX_DAY_RANGE,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    error = None
    result = None
    try:
        raw = _load_sample()
        if raw.empty:
            error = "Sample FIRMS CSV missing from data/sample_firms.csv."
        else:
            result = _run_pipeline(raw)
            result["source_label"] = "Sample FIRMS CSV (bundled demo)"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    ctx = _page_context(request, result=result, error=error, mode="sample")
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context=ctx)


@app.post("/run", response_class=HTMLResponse)
async def run(
    request: Request,
    mode: str = Form("sample"),
    region: str = Form("Turkey"),
    days: int = Form(5),
) -> HTMLResponse:
    error = None
    result = None
    days = max(1, min(int(days), MAX_DAY_RANGE))
    region = region if region in PRESET_REGIONS else "Turkey"

    try:
        if mode == "nrt":
            key = _firms_key()
            if not key:
                raise FirmsApiError(
                    "FIRMS_MAP_KEY is not configured on this host. Use sample mode."
                )
            ok, reason = _nrt_allowed()
            if not ok:
                raise FirmsApiError(reason)
            bbox = PRESET_REGIONS[region]
            end = date.today()
            start = end - timedelta(days=days - 1)
            _record_nrt()
            raw = fetch_firms_area(
                key,
                *bbox,
                start,
                end,
                sensors=AVAILABLE_SENSORS,
            )
            result = _run_pipeline(raw)
            result["source_label"] = (
                f"Live FIRMS NRT · {region} · {start.isoformat()} → {end.isoformat()}"
            )
        else:
            raw = _load_sample()
            if raw.empty:
                raise ValueError("Sample FIRMS CSV missing.")
            result = _run_pipeline(raw)
            result["source_label"] = "Sample FIRMS CSV (bundled demo)"
    except (FirmsApiError, ValueError) as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        error = f"Unexpected error: {exc}"

    ctx = _page_context(
        request,
        result=result,
        error=error,
        mode=mode,
        region=region,
        days=days,
    )
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/healthz")
@app.get("/_stcore/health")
async def health() -> dict:
    return {"status": "ok"}
