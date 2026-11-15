"""ASGI entry for Cursor publish host (and uvicorn).

HTTP-only FastAPI UI — Streamlit needs WebSockets that Cursor Apps
do not upgrade, so the public site uses this stack instead.

Local Streamlit (optional):  streamlit run streamlit_app.py
Hosted / ASGI:               uvicorn app:app --host 0.0.0.0 --port ${PORT:-8529}
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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

_NRT_HITS: list[float] = []
_NRT_MAX_PER_HOUR = 30
_NRT_MIN_GAP_SEC = 8.0
_MAP_POINT_CAP = 400


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Warm sample pipeline once at boot so the first visitor is fast
    try:
        cached_sample_result()
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="Unified Fire Activity Calendar", lifespan=_lifespan)
application = app
handler = app


def _firms_key() -> str:
    return (os.getenv("FIRMS_MAP_KEY") or "").strip()


def _nrt_allowed() -> tuple[bool, str]:
    now = time.monotonic()
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


@lru_cache(maxsize=1)
def _load_sample() -> pd.DataFrame:
    if not SAMPLE_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_CSV)


def _points_json(df: pd.DataFrame) -> str:
    if df.empty:
        return "[]"
    work = df
    if len(work) > _MAP_POINT_CAP:
        work = work.nlargest(_MAP_POINT_CAP, "frp") if "frp" in work.columns else work.head(
            _MAP_POINT_CAP
        )
    pts = []
    for row in work.itertuples(index=False):
        cat = str(getattr(row, "category", "") or "")
        pts.append(
            {
                "lat": float(row.latitude),
                "lon": float(row.longitude),
                "c": cat,
                "color": CATEGORY_COLORS.get(cat, "#88aabb"),
            }
        )
    return json.dumps(pts, separators=(",", ":"))


def _series_json(series: pd.DataFrame) -> str:
    if series is None or series.empty:
        return json.dumps({"labels": [], "matched": [], "modis": [], "viirs": []})
    labels = [str(p)[:10] for p in series["period"].tolist()]

    def _col(name: str) -> list[int]:
        if name not in series.columns:
            return [0] * len(labels)
        return [int(x) for x in series[name].tolist()]

    return json.dumps(
        {
            "labels": labels,
            "matched": _col(MATCHED),
            "modis": _col(MODIS_ONLY),
            "viirs": _col(VIIRS_ONLY),
            "colors": {
                "matched": CATEGORY_COLORS[MATCHED],
                "modis": CATEGORY_COLORS[MODIS_ONLY],
                "viirs": CATEGORY_COLORS[VIIRS_ONLY],
            },
        },
        separators=(",", ":"),
    )


def _run_pipeline(raw: pd.DataFrame) -> dict:
    missing = validate_firms_columns(raw)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    harm = harmonize_hotspots(raw)
    summary = summarize_categories(harm)
    series = activity_timeseries(harm)
    lat0 = float(harm["latitude"].mean()) if not harm.empty else 38.0
    lon0 = float(harm["longitude"].mean()) if not harm.empty else 35.0
    return {
        "summary": summary,
        "rows": int(len(harm)),
        "points_json": _points_json(harm),
        "series_json": _series_json(series),
        "map_center": [lat0, lon0],
        "map_zoom": 6 if not harm.empty else 5,
    }


@lru_cache(maxsize=1)
def cached_sample_result() -> dict:
    raw = _load_sample()
    if raw.empty:
        raise ValueError("Sample FIRMS CSV missing from data/sample_firms.csv.")
    result = _run_pipeline(raw)
    result["source_label"] = "Sample FIRMS CSV (bundled demo)"
    return result


def _page_context(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    mode: str = "sample",
    region: str = "Turkey",
    days: int = 5,
) -> dict:
    return {
        "request": request,
        "presets": list(PRESET_REGIONS.keys()),
        "region": region,
        "days": days,
        "has_key": bool(_firms_key()),
        "mode": mode,
        "error": error,
        "result": result,
        "max_days": MAX_DAY_RANGE,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    error = None
    result = None
    try:
        result = cached_sample_result()
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
            result = cached_sample_result()
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
