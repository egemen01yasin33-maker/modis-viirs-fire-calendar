"""ASGI entry for Cursor publish host (and uvicorn).

HTTP-only FastAPI UI (Streamlit WebSockets are not upgraded on Cursor Apps).

Modes:
  - archive: 2023–2026 FIRMS window with date pickers + period compare
  - sample: tiny bundled demo CSV
  - nrt: live FIRMS (CSRF + NRT_GATE_PASSWORD + per-IP rate limits)

MAP_KEY never appears in the browser.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from src.archive_csv import ArchiveCsvError
from src.bundled_archive import (
    ARCHIVE_DATE_MAX,
    ARCHIVE_DATE_MIN,
    archive_date_bounds,
    daily_to_series_json,
    list_archive_regions,
    load_archive_window,
    pick_bundled_region_for_bbox,
)
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
_MAP_POINT_CAP = 3000

_CSRF_COOKIE = "fc_csrf"

# Rate limits use the direct TCP peer only — never X-Forwarded-For (spoofable).
_NRT_HITS: dict[str, list[float]] = defaultdict(list)
_NRT_MAX_PER_HOUR = 3
_NRT_MIN_GAP_SEC = 60.0
_NRT_GLOBAL_HITS: list[float] = []
_NRT_GLOBAL_MAX_PER_HOUR = 20

_GATE_FAILS: dict[str, list[float]] = defaultdict(list)
_GATE_FAIL_MAX = 5
_GATE_FAIL_WINDOW_SEC = 900.0  # 15 minutes

_ARCHIVE_HITS: dict[str, list[float]] = defaultdict(list)
_ARCHIVE_MAX_PER_MIN = 12
_ARCHIVE_MAX_PER_HOUR = 120
_ARCHIVE_GLOBAL_HITS: list[float] = []
_ARCHIVE_GLOBAL_MAX_PER_MIN = 60


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
    """
    Rate-limit identity = direct connection peer only.

    Public clients can forge X-Forwarded-For / X-Real-IP; those headers are
    ignored so one password cannot mint unlimited per-IP FIRMS quotas.
    """
    if request.client and request.client.host:
        return request.client.host[:64]
    return "unknown"


def _prune(hits: list[float], now: float, window: float) -> list[float]:
    return [t for t in hits if now - t < window]


def _issue_csrf() -> str:
    return secrets.token_urlsafe(24)


def _csrf_ok(request: Request, submitted: str) -> bool:
    cookie = request.cookies.get(_CSRF_COOKIE) or ""
    if not cookie or not submitted or not hmac.compare_digest(cookie, submitted):
        return False
    return len(cookie) >= 16


def _cookie_secure() -> bool:
    return bool(os.getenv("CURSOR_APP_ID") or os.getenv("VERCEL"))


def _public_error(exc: BaseException) -> str:
    """User-facing message — never raw unexpected exception text."""
    if isinstance(exc, (FirmsApiError, ArchiveCsvError, ValueError)):
        text = str(exc).strip()
        if text and len(text) <= 240 and "\n" not in text:
            return text
        if isinstance(exc, FirmsApiError):
            return "Live FIRMS request could not be completed. Try again later."
        if isinstance(exc, ArchiveCsvError):
            return "Archive query failed for this region or date window."
        return "Invalid request. Check dates, region, or coordinates."
    return "Something went wrong. Reload and try again."


def _nrt_rate_ok(ip: str) -> tuple[bool, str]:
    now = time.monotonic()
    hits = _prune(_NRT_HITS[ip], now, 3600)
    _NRT_HITS[ip] = hits
    if len(hits) >= _NRT_MAX_PER_HOUR:
        return False, "Live FIRMS limit for this network reached. Try again later."
    if hits and (now - hits[-1]) < _NRT_MIN_GAP_SEC:
        wait = int(_NRT_MIN_GAP_SEC - (now - hits[-1])) + 1
        return False, f"Please wait {wait}s before another live FIRMS request."

    global_hits = _prune(_NRT_GLOBAL_HITS, now, 3600)
    _NRT_GLOBAL_HITS[:] = global_hits
    if len(global_hits) >= _NRT_GLOBAL_MAX_PER_HOUR:
        return False, "Live FIRMS is busy right now. Use archive mode or retry later."
    return True, ""


def _record_nrt(ip: str) -> None:
    now = time.monotonic()
    _NRT_HITS[ip].append(now)
    _NRT_GLOBAL_HITS.append(now)


def _gate_fail_ok(ip: str) -> tuple[bool, str]:
    now = time.monotonic()
    fails = _prune(_GATE_FAILS[ip], now, _GATE_FAIL_WINDOW_SEC)
    _GATE_FAILS[ip] = fails
    if len(fails) >= _GATE_FAIL_MAX:
        return False, "Too many invalid access codes. Try again later."
    return True, ""


def _record_gate_fail(ip: str) -> None:
    _GATE_FAILS[ip].append(time.monotonic())


def _gate_ok(submitted: str) -> bool:
    expected = _gate_password()
    if not expected:
        return False
    # Constant-time compare; empty / short submissions still go through digest path
    got = (submitted or "").strip()
    if len(got) != len(expected):
        # Still compare against expected to keep timing closer
        hmac.compare_digest(expected, expected)
        return False
    return hmac.compare_digest(got, expected)


def _archive_rate_ok(ip: str) -> tuple[bool, str]:
    now = time.monotonic()
    hits = _prune(_ARCHIVE_HITS[ip], now, 3600)
    _ARCHIVE_HITS[ip] = hits
    recent = [t for t in hits if now - t < 60]
    if len(recent) >= _ARCHIVE_MAX_PER_MIN:
        return False, "Too many archive requests. Wait a moment and try again."
    if len(hits) >= _ARCHIVE_MAX_PER_HOUR:
        return False, "Archive request limit reached for this network. Try later."

    global_hits = _prune(_ARCHIVE_GLOBAL_HITS, now, 60)
    _ARCHIVE_GLOBAL_HITS[:] = global_hits
    if len(global_hits) >= _ARCHIVE_GLOBAL_MAX_PER_MIN:
        return False, "Server is busy. Please retry shortly."
    return True, ""


def _record_archive(ip: str) -> None:
    now = time.monotonic()
    _ARCHIVE_HITS[ip].append(now)
    _ARCHIVE_GLOBAL_HITS.append(now)


def _parse_date(value: str | date | None, fallback: date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return fallback
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return fallback


def _clamp_archive_dates(start: date, end: date) -> tuple[date, date]:
    lo, hi = archive_date_bounds(ROOT)
    lo = max(lo, ARCHIVE_DATE_MIN)
    hi = min(hi, ARCHIVE_DATE_MAX)
    start = min(max(start, lo), hi)
    end = min(max(end, lo), hi)
    if start > end:
        start, end = end, start
    return start, end


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
        frp_val = getattr(row, "frp", None)
        try:
            frp_f = float(frp_val) if frp_val is not None and frp_val == frp_val else None
        except (TypeError, ValueError):
            frp_f = None
        pts.append(
            {
                "lat": float(row.latitude),
                "lon": float(row.longitude),
                "c": cat,
                "frp": frp_f,
                "color": CATEGORY_COLORS.get(cat, "#88aabb"),
            }
        )
    return json.dumps(pts, separators=(",", ":"))


def _series_json(series: pd.DataFrame) -> str:
    if series is None or series.empty:
        return json.dumps(
            {
                "labels": [],
                "matched": [],
                "modis": [],
                "viirs": [],
                "frp": [],
                "colors": {
                    "matched": CATEGORY_COLORS[MATCHED],
                    "modis": CATEGORY_COLORS[MODIS_ONLY],
                    "viirs": CATEGORY_COLORS[VIIRS_ONLY],
                    "frp": "#f0c674",
                },
            }
        )
    labels = [str(p)[:10] for p in series["period"].tolist()]

    def _col(name: str) -> list[int]:
        if name not in series.columns:
            return [0] * len(labels)
        return [int(x) for x in series[name].tolist()]

    if "frp_sum" in series.columns:
        frp = [round(float(x or 0), 2) for x in series["frp_sum"].tolist()]
    else:
        frp = [0.0] * len(labels)

    return json.dumps(
        {
            "labels": labels,
            "matched": _col(MATCHED),
            "modis": _col(MODIS_ONLY),
            "viirs": _col(VIIRS_ONLY),
            "frp": frp,
            "colors": {
                "matched": CATEGORY_COLORS[MATCHED],
                "modis": CATEGORY_COLORS[MODIS_ONLY],
                "viirs": CATEGORY_COLORS[VIIRS_ONLY],
                "frp": "#f0c674",
            },
        },
        separators=(",", ":"),
    )


def _auto_mode(start: date, end: date) -> str:
    """
    Pick data source from the date window:
    - Live NRT when the window is recent (≤ MAX_DAY_RANGE days, ends near today)
      and NRT is configured
    - Otherwise Archive 2023–2026
    """
    today = date.today()
    span = (end - start).days + 1
    recent = end >= (today - timedelta(days=1)) and start >= (today - timedelta(days=365))
    if _nrt_configured() and recent and 1 <= span <= MAX_DAY_RANGE:
        return "nrt"
    return "archive"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    if pct == float("inf"):
        return "+∞%"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _comparison_rows_from_summaries(
    primary_summary: dict,
    comparison_summary: dict,
) -> list[dict]:
    """Side-by-side metrics from cached archive summaries (no raw frames)."""
    from src.analytics import pct_change

    labels = {
        "total": "Total",
        "matched": "Matched",
        "modis_only": "MODIS-only",
        "viirs_only": "VIIRS-only",
    }
    out = []
    for key, label in labels.items():
        # Prefer exact window total; sample categories otherwise
        if key == "total":
            p = int(primary_summary.get("total") or 0)
            c = int(comparison_summary.get("total") or 0)
        else:
            p = int(primary_summary.get(key) or 0)
            c = int(comparison_summary.get(key) or 0)
        out.append(
            {
                "key": key,
                "label": label,
                "primary": p,
                "comparison": c,
                "delta": p - c,
                "pct": _fmt_pct(pct_change(p, c)),
            }
        )
    return out


def _run_pipeline(raw: pd.DataFrame, *, daily: pd.DataFrame | None = None) -> dict:
    missing = validate_firms_columns(raw)
    if missing and not raw.empty:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if raw.empty:
        harm = raw
        summary = {"matched": 0, "modis_only": 0, "viirs_only": 0, "total": 0}
        series_payload = daily_to_series_json(daily) if daily is not None else {
            "labels": [],
            "matched": [],
            "modis": [],
            "viirs": [],
        }
        series_json = json.dumps(series_payload, separators=(",", ":"))
    else:
        harm = harmonize_hotspots(raw)
        summary = summarize_categories(harm)
        if daily is not None and not daily.empty:
            series_json = json.dumps(daily_to_series_json(daily), separators=(",", ":"))
        else:
            series_json = _series_json(activity_timeseries(harm))
    lat0 = float(harm["latitude"].mean()) if not harm.empty else 38.0
    lon0 = float(harm["longitude"].mean()) if not harm.empty else 35.0
    return {
        "summary": summary,
        "rows": int(len(harm)),
        "points_json": _points_json(harm),
        "series_json": series_json,
        "map_center": [lat0, lon0],
        "map_zoom": 6 if not harm.empty else 5,
        "harm": harm,
    }


def _parse_bbox(
    *,
    area_mode: str,
    region: str,
    min_lon: str | float,
    min_lat: str | float,
    max_lon: str | float,
    max_lat: str | float,
) -> tuple[str, tuple[float, float, float, float]]:
    """Return (label, bbox). Custom mode validates numeric west/south/east/north."""
    if area_mode == "custom":
        try:
            west = float(min_lon)
            south = float(min_lat)
            east = float(max_lon)
            north = float(max_lat)
        except (TypeError, ValueError) as exc:
            raise ValueError("Coordinates must be numbers (lon/lat).") from exc
        if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
            raise ValueError(
                "Invalid bbox: need west < east and south < north within world bounds."
            )
        if (east - west) > 80 or (north - south) > 60:
            raise ValueError("Custom area is too large — shrink the box (max ~80°×60°).")
        return "Custom coordinates", (west, south, east, north)

    regions = list_archive_regions(ROOT)
    if region not in PRESET_REGIONS:
        region = "Turkey" if "Turkey" in PRESET_REGIONS else next(iter(PRESET_REGIONS))
    if region not in regions and regions:
        # Hosted bundle may omit Global etc. — fall back to Turkey
        region = "Turkey" if "Turkey" in regions else regions[0]
    return region, PRESET_REGIONS[region]


@lru_cache(maxsize=64)
def _archive_display_cached(
    region_label: str,
    start_iso: str,
    end_iso: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
    """Cached JSON for an archive window (no DataFrames — safe to reuse)."""
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    if region_label in PRESET_REGIONS and region_label != "Custom coordinates":
        bundle_key = region_label
    else:
        bundle_key = pick_bundled_region_for_bbox(
            ROOT, min_lon, min_lat, max_lon, max_lat
        )
    payload = load_archive_window(
        ROOT,
        region=bundle_key,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        start=start,
        end=end,
    )
    result = _run_pipeline(payload["raw"], daily=payload.get("daily"))
    total = int(payload.get("total_count") or result["rows"])
    daily = payload.get("daily")
    frp_total = 0.0
    if daily is not None and not daily.empty and "frp_sum" in daily.columns:
        frp_total = float(daily["frp_sum"].sum())
    elif not payload["raw"].empty and "frp" in payload["raw"].columns:
        frp_total = float(
            pd.to_numeric(payload["raw"]["frp"], errors="coerce").fillna(0).sum()
        )
    summary = {
        **result["summary"],
        "total": total,
        "frp_sum": round(frp_total, 1),
    }
    map_center = result["map_center"]
    map_zoom = result["map_zoom"]
    if total == 0 or payload["raw"].empty:
        map_center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
        map_zoom = 6
    # Drop non-JSON harm frame before caching
    result.pop("harm", None)
    display = {
        "summary": summary,
        "rows": total,
        "points_json": result["points_json"],
        "series_json": result["series_json"],
        "map_center": map_center,
        "map_zoom": map_zoom,
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "map_bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "source_label": (
            f"Archive · {region_label} · {start_iso} → {end_iso} "
            f"({payload.get('source', 'archive')})"
        ),
        "period_label": f"{start_iso} → {end_iso}",
    }
    return json.dumps(display, separators=(",", ":"))


def _archive_result(
    region_label: str,
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
) -> dict:
    min_lon, min_lat, max_lon, max_lat = (
        round(float(bbox[0]), 4),
        round(float(bbox[1]), 4),
        round(float(bbox[2]), 4),
        round(float(bbox[3]), 4),
    )
    raw = _archive_display_cached(
        region_label,
        start.isoformat(),
        end.isoformat(),
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )
    result = json.loads(raw)
    # Placeholder so compare path can use summary-only deltas
    result["harm"] = pd.DataFrame()
    result["_summary_only"] = True
    return result


@lru_cache(maxsize=1)
def cached_sample_result() -> dict:
    raw = _load_sample()
    if raw.empty:
        raise ValueError("Sample FIRMS CSV missing from data/sample_firms.csv.")
    result = _run_pipeline(raw)
    result["source_label"] = "Sample FIRMS CSV (bundled demo)"
    result.pop("harm", None)
    return result


def _default_form() -> dict:
    lo, hi = archive_date_bounds(ROOT)
    lo = max(lo, ARCHIVE_DATE_MIN)
    hi = min(hi, ARCHIVE_DATE_MAX)
    # Default primary: latest full summer season in range (or last 90 days)
    end = hi
    start = max(lo, date(end.year, 6, 1)) if end.month >= 6 else max(lo, date(end.year - 1, 6, 1))
    if start > end:
        start = lo
    # Default compare: same season previous year
    c_end = date(start.year - 1, min(end.month, 12), min(end.day, 28))
    c_start = date(start.year - 1, start.month, min(start.day, 28))
    c_start, c_end = _clamp_archive_dates(c_start, c_end)
    turkey = PRESET_REGIONS.get("Turkey", (25.5, 35.8, 45.0, 42.3))
    return {
        "mode": "archive",
        "mode_label": "Archive 2023–2026 (auto from dates)",
        "area_mode": "preset",
        "region": "Turkey",
        "min_lon": f"{turkey[0]:.2f}",
        "min_lat": f"{turkey[1]:.2f}",
        "max_lon": f"{turkey[2]:.2f}",
        "max_lat": f"{turkey[3]:.2f}",
        "days": 5,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "compare": "1",
        "compare_start": c_start.isoformat(),
        "compare_end": c_end.isoformat(),
        "archive_min": lo.isoformat(),
        "archive_max": hi.isoformat(),
    }


def _page_context(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    form: dict | None = None,
    csrf: str,
) -> dict:
    defaults = _default_form()
    form = {**defaults, **(form or {})}
    archive_regions = list_archive_regions(ROOT) or ["Turkey"]
    return {
        "request": request,
        "presets": archive_regions if form.get("mode") == "archive" else list(PRESET_REGIONS.keys()),
        "archive_regions": archive_regions,
        "all_presets": list(PRESET_REGIONS.keys()),
        "preset_bboxes": {k: list(v) for k, v in PRESET_REGIONS.items()},
        "form": form,
        "error": error,
        "result": result,
        "max_days": MAX_DAY_RANGE,
        "nrt_available": _nrt_configured(),
        "archive_available": bool(archive_regions),
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
        samesite="lax",
        secure=_cookie_secure(),
        max_age=3600,
    )
    return resp


def _run_with_form(form: dict) -> tuple[dict | None, str | None, dict]:
    """Execute archive/sample from form dates; mode is derived from the window."""
    error = None
    result = None
    try:
        label, bbox = _parse_bbox(
            area_mode=form.get("area_mode") or "preset",
            region=form.get("region") or "Turkey",
            min_lon=form.get("min_lon") or 25.5,
            min_lat=form.get("min_lat") or 35.8,
            max_lon=form.get("max_lon") or 45.0,
            max_lat=form.get("max_lat") or 42.3,
        )
        form["min_lon"] = f"{bbox[0]:.4f}".rstrip("0").rstrip(".")
        form["min_lat"] = f"{bbox[1]:.4f}".rstrip("0").rstrip(".")
        form["max_lon"] = f"{bbox[2]:.4f}".rstrip("0").rstrip(".")
        form["max_lat"] = f"{bbox[3]:.4f}".rstrip("0").rstrip(".")
        if form.get("area_mode") != "custom":
            form["region"] = label

        start = _parse_date(form.get("start_date"), ARCHIVE_DATE_MIN)
        end = _parse_date(form.get("end_date"), ARCHIVE_DATE_MAX)
        start, end = _clamp_archive_dates(start, end)
        form["start_date"] = start.isoformat()
        form["end_date"] = end.isoformat()
        form["mode"] = _auto_mode(start, end)
        form["mode_label"] = (
            "Live FIRMS NRT (auto from dates)"
            if form["mode"] == "nrt"
            else "Archive 2023–2026 (auto from dates)"
        )

        if form["mode"] == "nrt":
            # NRT needs access code — fall back to archive for anonymous GET/chips
            form["mode"] = "archive"
            form["mode_label"] = "Archive 2023–2026 (auto — live needs access code)"

        if not list_archive_regions(ROOT):
            result = cached_sample_result()
            form["mode"] = "sample"
            form["mode_label"] = "Sample demo"
            return result, None, form

        result = _archive_result(label, start, end, bbox)
        if form.get("compare") == "1":
            cs = _parse_date(form["compare_start"], start)
            ce = _parse_date(form["compare_end"], end)
            cs, ce = _clamp_archive_dates(cs, ce)
            form["compare_start"] = cs.isoformat()
            form["compare_end"] = ce.isoformat()
            other = _archive_result(label, cs, ce, bbox)
            rows = _comparison_rows_from_summaries(result["summary"], other["summary"])
            result["compare"] = {
                "label": other["period_label"],
                "rows": rows,
                "source_label": other["source_label"],
                "summary": other["summary"],
            }
        result.pop("harm", None)
        result.pop("_summary_only", None)
    except (FirmsApiError, ArchiveCsvError, ValueError) as exc:
        error = _public_error(exc)
    except Exception as exc:  # noqa: BLE001
        error = _public_error(exc)
    return result, error, form


@app.get("/", response_class=HTMLResponse)
@app.get("/run", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()
    form = _default_form()
    ip = _client_ip(request)
    ok, reason = _archive_rate_ok(ip)
    if not ok:
        ctx = _page_context(
            request,
            result=cached_sample_result() if SAMPLE_CSV.is_file() else None,
            error=reason,
            form=form,
            csrf=csrf,
        )
        return _html_response(request, ctx, csrf=csrf)
    _record_archive(ip)
    result, error, form = _run_with_form(form)
    if error and result is None:
        try:
            result = cached_sample_result()
        except Exception:  # noqa: BLE001
            result = None
    ctx = _page_context(request, result=result, error=error, form=form, csrf=csrf)
    return _html_response(request, ctx, csrf=csrf)


@app.post("/run", response_class=HTMLResponse)
async def run(
    request: Request,
    area_mode: str = Form("preset"),
    region: str = Form("Turkey"),
    min_lon: str = Form("25.5"),
    min_lat: str = Form("35.8"),
    max_lon: str = Form("45.0"),
    max_lat: str = Form("42.3"),
    days: int = Form(5),
    start_date: str = Form(""),
    end_date: str = Form(""),
    compare: str = Form(""),
    compare_start: str = Form(""),
    compare_end: str = Form(""),
    csrf_token: str = Form(""),
    access_code: str = Form(""),
    force_nrt: str = Form(""),
) -> Response:
    days = max(1, min(int(days), MAX_DAY_RANGE))
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()
    form = {
        **_default_form(),
        "area_mode": "custom" if area_mode == "custom" else "preset",
        "region": region if region in PRESET_REGIONS else "Turkey",
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "days": days,
        "start_date": start_date,
        "end_date": end_date,
        "compare": "1" if compare in {"1", "on", "true", "yes"} else "",
        "compare_start": compare_start,
        "compare_end": compare_end,
    }

    start = _parse_date(form["start_date"], ARCHIVE_DATE_MIN)
    end = _parse_date(form["end_date"], ARCHIVE_DATE_MAX)
    start, end = _clamp_archive_dates(start, end)
    form["start_date"] = start.isoformat()
    form["end_date"] = end.isoformat()
    auto = _auto_mode(start, end)
    want_nrt = force_nrt in {"1", "on", "true", "yes"} or auto == "nrt"

    # Archive/sample are read-only public queries — soft CSRF (tunnel/cookie quirks).
    # Live NRT still requires a valid CSRF + access code.
    csrf_valid = _csrf_ok(request, csrf_token)
    ip = _client_ip(request)
    if want_nrt and _nrt_configured() and access_code:
        if not csrf_valid:
            error = "Invalid or missing form token. Reload the page and try again."
            form["mode"] = "nrt"
            form["mode_label"] = "Live FIRMS NRT"
            ctx = _page_context(request, result=None, error=error, form=form, csrf=csrf)
            return _html_response(request, ctx, csrf=csrf)
        error = None
        result = None
        try:
            gate_ok, gate_reason = _gate_fail_ok(ip)
            if not gate_ok:
                raise FirmsApiError(gate_reason)
            if not _gate_ok(access_code):
                _record_gate_fail(ip)
                raise FirmsApiError("Invalid access code for live FIRMS.")
            ok, reason = _nrt_rate_ok(ip)
            if not ok:
                raise FirmsApiError(reason)
            label, bbox = _parse_bbox(
                area_mode=form["area_mode"],
                region=form["region"],
                min_lon=form["min_lon"],
                min_lat=form["min_lat"],
                max_lon=form["max_lon"],
                max_lat=form["max_lat"],
            )
            form["min_lon"] = str(bbox[0])
            form["min_lat"] = str(bbox[1])
            form["max_lon"] = str(bbox[2])
            form["max_lat"] = str(bbox[3])
            nrt_end = date.today()
            nrt_start = nrt_end - timedelta(days=days - 1)
            # Record quota before the outbound call so aborts still count
            _record_nrt(ip)
            raw = fetch_firms_area(
                _firms_key(),
                *bbox,
                nrt_start,
                nrt_end,
                sensors=AVAILABLE_SENSORS,
            )
            result = _run_pipeline(raw)
            result["bbox"] = list(bbox)
            result["map_bounds"] = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
            if not raw.empty and "frp" in raw.columns:
                frp_total = float(
                    pd.to_numeric(raw["frp"], errors="coerce").fillna(0).sum()
                )
                result["summary"] = {
                    **result["summary"],
                    "frp_sum": round(frp_total, 1),
                }
            result["source_label"] = (
                f"Live FIRMS NRT · {label} · "
                f"{nrt_start.isoformat()} → {nrt_end.isoformat()}"
            )
            result.pop("harm", None)
            form["mode"] = "nrt"
            form["mode_label"] = "Live FIRMS NRT (auto from dates)"
        except (FirmsApiError, ValueError) as exc:
            error = _public_error(exc)
            result, _, form = _run_with_form(form)
        except Exception as exc:  # noqa: BLE001
            error = _public_error(exc)
    else:
        ok, reason = _archive_rate_ok(ip)
        if not ok:
            error = reason
            result = None
            try:
                result = cached_sample_result()
            except Exception:  # noqa: BLE001
                pass
            form["mode"] = "archive"
        else:
            _record_archive(ip)
            if not csrf_valid:
                csrf = _issue_csrf()
            result, error, form = _run_with_form(form)

    csrf = _issue_csrf()
    ctx = _page_context(
        request,
        result=result,
        error=error,
        form=form,
        csrf=csrf,
    )
    return _html_response(request, ctx, csrf=csrf)


@app.get("/healthz")
@app.get("/_stcore/health")
async def health() -> dict:
    return {"status": "ok"}
