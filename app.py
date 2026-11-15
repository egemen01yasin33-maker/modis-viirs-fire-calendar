"""ASGI entry for Cursor publish host (and uvicorn).

HTTP-only FastAPI UI (Streamlit WebSockets are not upgraded on Cursor Apps).

Public sample demo is always available. Live FIRMS NRT stays available but is
gated: CSRF cookie + shared operator access code (NRT_GATE_PASSWORD) + per-IP
rate limits. MAP_KEY never appears in the browser.

Local Streamlit (optional):  streamlit run streamlit_app.py
Hosted / ASGI:               uvicorn app:app --host 0.0.0.0 --port ${PORT:-8529}
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
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
_MAP_POINT_CAP = 400

_CSRF_COOKIE = "fc_csrf"

# Per-IP NRT limits (in-memory; resets on process restart)
_IP_HITS: dict[str, list[float]] = defaultdict(list)
_IP_MAX_PER_HOUR = 3
_IP_MIN_GAP_SEC = 60.0
_GLOBAL_HITS: list[float] = []
_GLOBAL_MAX_PER_HOUR = 40


@asynccontextmanager
async def _lifespan(_app: FastAPI):
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


def _gate_password() -> str:
    return (os.getenv("NRT_GATE_PASSWORD") or "").strip()


def _nrt_configured() -> bool:
    return bool(_firms_key() and _gate_password())


def _client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded[:64]
    if request.client and request.client.host:
        return request.client.host[:64]
    return "unknown"


def _issue_csrf() -> str:
    return secrets.token_urlsafe(24)


def _csrf_ok(request: Request, submitted: str) -> bool:
    cookie = request.cookies.get(_CSRF_COOKIE) or ""
    if not cookie or not submitted or not hmac.compare_digest(cookie, submitted):
        return False
    return len(cookie) >= 16


def _cookie_secure() -> bool:
    return bool(os.getenv("CURSOR_APP_ID") or os.getenv("VERCEL"))


def _nrt_rate_ok(ip: str) -> tuple[bool, str]:
    now = time.monotonic()
    hits = [t for t in _IP_HITS[ip] if now - t < 3600]
    _IP_HITS[ip] = hits
    if len(hits) >= _IP_MAX_PER_HOUR:
        return False, "Live FIRMS limit for this network reached. Try again later."
    if hits and (now - hits[-1]) < _IP_MIN_GAP_SEC:
        wait = int(_IP_MIN_GAP_SEC - (now - hits[-1])) + 1
        return False, f"Please wait {wait}s before another live FIRMS request."

    global_hits = [t for t in _GLOBAL_HITS if now - t < 3600]
    _GLOBAL_HITS[:] = global_hits
    if len(global_hits) >= _GLOBAL_MAX_PER_HOUR:
        return False, "Live FIRMS is busy right now. Use sample mode or retry later."
    return True, ""


def _record_nrt(ip: str) -> None:
    now = time.monotonic()
    _IP_HITS[ip].append(now)
    _GLOBAL_HITS.append(now)


def _gate_ok(submitted: str) -> bool:
    expected = _gate_password()
    if not expected:
        return False
    return hmac.compare_digest(submitted.strip(), expected)


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
        work = (
            work.nlargest(_MAP_POINT_CAP, "frp")
            if "frp" in work.columns
            else work.head(_MAP_POINT_CAP)
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
    csrf: str,
) -> dict:
    return {
        "request": request,
        "presets": list(PRESET_REGIONS.keys()),
        "region": region,
        "days": days,
        "mode": mode,
        "error": error,
        "result": result,
        "max_days": MAX_DAY_RANGE,
        "nrt_available": _nrt_configured(),
        "csrf": csrf,
    }


def _html_response(
    request: Request,
    ctx: dict,
    *,
    csrf: str,
) -> Response:
    resp = TEMPLATES.TemplateResponse(request=request, name="index.html", context=ctx)
    resp.set_cookie(
        key=_CSRF_COOKIE,
        value=csrf,
        httponly=True,
        samesite="strict",
        secure=_cookie_secure(),
        max_age=3600,
    )
    return resp


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    error = None
    result = None
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()
    try:
        result = cached_sample_result()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    ctx = _page_context(request, result=result, error=error, csrf=csrf)
    return _html_response(request, ctx, csrf=csrf)


@app.post("/run", response_class=HTMLResponse)
async def run(
    request: Request,
    mode: str = Form("sample"),
    region: str = Form("Turkey"),
    days: int = Form(5),
    csrf_token: str = Form(""),
    access_code: str = Form(""),
) -> Response:
    error = None
    result = None
    days = max(1, min(int(days), MAX_DAY_RANGE))
    region = region if region in PRESET_REGIONS else "Turkey"
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()

    if not _csrf_ok(request, csrf_token):
        error = "Invalid or missing form token. Reload the page and try again."
        ctx = _page_context(
            request, result=None, error=error, mode=mode, region=region, days=days, csrf=csrf
        )
        return _html_response(request, ctx, csrf=csrf)

    try:
        if mode == "nrt":
            if not _nrt_configured():
                raise FirmsApiError(
                    "Live FIRMS is not enabled on this host "
                    "(needs FIRMS_MAP_KEY and NRT_GATE_PASSWORD)."
                )
            if not _gate_ok(access_code):
                raise FirmsApiError("Invalid access code for live FIRMS.")
            ip = _client_ip(request)
            ok, reason = _nrt_rate_ok(ip)
            if not ok:
                raise FirmsApiError(reason)
            key = _firms_key()
            bbox = PRESET_REGIONS[region]
            end = date.today()
            start = end - timedelta(days=days - 1)
            _record_nrt(ip)
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

    # Rotate CSRF after successful POSTs
    csrf = _issue_csrf()
    ctx = _page_context(
        request,
        result=result,
        error=error,
        mode=mode,
        region=region,
        days=days,
        csrf=csrf,
    )
    return _html_response(request, ctx, csrf=csrf)


@app.get("/healthz")
@app.get("/_stcore/health")
async def health() -> dict:
    return {"status": "ok"}
