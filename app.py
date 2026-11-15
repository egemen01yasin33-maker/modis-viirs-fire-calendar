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

from src.analytics import compare_summaries
from src.archive_csv import ArchiveCsvError
from src.bundled_archive import (
    ARCHIVE_DATE_MAX,
    ARCHIVE_DATE_MIN,
    archive_date_bounds,
    daily_to_series_json,
    list_archive_regions,
    load_archive_window,
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
_MAP_POINT_CAP = 400

_CSRF_COOKIE = "fc_csrf"

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
        return False, "Live FIRMS is busy right now. Use archive mode or retry later."
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


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    if pct == float("inf"):
        return "+∞%"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _comparison_rows(primary: pd.DataFrame, comparison: pd.DataFrame) -> list[dict]:
    rows = compare_summaries(primary, comparison)
    labels = {
        "total": "Total",
        "matched": "Matched",
        "modis_only": "MODIS-only",
        "viirs_only": "VIIRS-only",
    }
    out = []
    for key, label in labels.items():
        item = rows[key]
        out.append(
            {
                "key": key,
                "label": label,
                "primary": int(item["primary"]),
                "comparison": int(item["comparison"]),
                "delta": int(item["delta"]),
                "pct": _fmt_pct(item["pct"]),  # type: ignore[arg-type]
            }
        )
    return out


def _overlay_exact_totals(
    rows: list[dict],
    primary_summary: dict,
    comparison_summary: dict,
) -> list[dict]:
    """Replace Total with exact daily-aggregate window counts when available."""
    from src.analytics import pct_change

    p = int(primary_summary.get("total") or 0)
    c = int(comparison_summary.get("total") or 0)
    out = []
    for row in rows:
        if row.get("key") == "total":
            delta = p - c
            out.append(
                {
                    **row,
                    "primary": p,
                    "comparison": c,
                    "delta": delta,
                    "pct": _fmt_pct(pct_change(p, c)),
                }
            )
        else:
            out.append(row)
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


def _archive_result(
    region: str,
    start: date,
    end: date,
) -> dict:
    regions = list_archive_regions(ROOT)
    if region not in regions:
        region = "Turkey" if "Turkey" in regions else (regions[0] if regions else region)
    if region not in PRESET_REGIONS:
        raise ArchiveCsvError(f"Unknown region: {region}")
    bbox = PRESET_REGIONS[region]
    payload = load_archive_window(
        ROOT,
        region=region,
        min_lon=bbox[0],
        min_lat=bbox[1],
        max_lon=bbox[2],
        max_lat=bbox[3],
        start=start,
        end=end,
    )
    result = _run_pipeline(payload["raw"], daily=payload.get("daily"))
    total = int(payload.get("total_count") or result["rows"])
    result["summary"] = {
        **result["summary"],
        "total": total,
    }
    result["rows"] = total
    result["source_label"] = (
        f"Archive · {region} · {start.isoformat()} → {end.isoformat()} "
        f"({payload.get('source', 'archive')})"
    )
    result["period_label"] = f"{start.isoformat()} → {end.isoformat()}"
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
    return {
        "mode": "archive",
        "region": "Turkey",
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
        samesite="strict",
        secure=_cookie_secure(),
        max_age=3600,
    )
    return resp


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    error = None
    result = None
    compare = None
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()
    form = _default_form()
    try:
        start = _parse_date(form["start_date"], ARCHIVE_DATE_MIN)
        end = _parse_date(form["end_date"], ARCHIVE_DATE_MAX)
        start, end = _clamp_archive_dates(start, end)
        form["start_date"] = start.isoformat()
        form["end_date"] = end.isoformat()
        if list_archive_regions(ROOT):
            result = _archive_result(form["region"], start, end)
            if form.get("compare") == "1":
                cs = _parse_date(form["compare_start"], start)
                ce = _parse_date(form["compare_end"], end)
                cs, ce = _clamp_archive_dates(cs, ce)
                form["compare_start"] = cs.isoformat()
                form["compare_end"] = ce.isoformat()
                other = _archive_result(form["region"], cs, ce)
                rows = _comparison_rows(result["harm"], other["harm"])
                # Prefer exact window totals from daily aggregates for Total
                rows = _overlay_exact_totals(rows, result["summary"], other["summary"])
                result["compare"] = {
                    "label": other["period_label"],
                    "rows": rows,
                    "source_label": other["source_label"],
                    "summary": other["summary"],
                }
            result.pop("harm", None)
        else:
            result = cached_sample_result()
            form["mode"] = "sample"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        try:
            result = cached_sample_result()
        except Exception:  # noqa: BLE001
            result = None
    ctx = _page_context(request, result=result, error=error, form=form, csrf=csrf)
    return _html_response(request, ctx, csrf=csrf)


@app.post("/run", response_class=HTMLResponse)
async def run(
    request: Request,
    mode: str = Form("archive"),
    region: str = Form("Turkey"),
    days: int = Form(5),
    start_date: str = Form(""),
    end_date: str = Form(""),
    compare: str = Form(""),
    compare_start: str = Form(""),
    compare_end: str = Form(""),
    csrf_token: str = Form(""),
    access_code: str = Form(""),
) -> Response:
    error = None
    result = None
    days = max(1, min(int(days), MAX_DAY_RANGE))
    if region not in PRESET_REGIONS:
        region = "Turkey"
    csrf = request.cookies.get(_CSRF_COOKIE) or _issue_csrf()
    form = {
        **_default_form(),
        "mode": mode if mode in {"archive", "sample", "nrt"} else "archive",
        "region": region,
        "days": days,
        "start_date": start_date,
        "end_date": end_date,
        "compare": "1" if compare in {"1", "on", "true", "yes"} else "",
        "compare_start": compare_start,
        "compare_end": compare_end,
    }

    if not _csrf_ok(request, csrf_token):
        error = "Invalid or missing form token. Reload the page and try again."
        ctx = _page_context(request, result=None, error=error, form=form, csrf=csrf)
        return _html_response(request, ctx, csrf=csrf)

    try:
        if form["mode"] == "nrt":
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
            result.pop("harm", None)
        elif form["mode"] == "sample":
            result = cached_sample_result()
        else:
            start = _parse_date(form["start_date"], ARCHIVE_DATE_MIN)
            end = _parse_date(form["end_date"], ARCHIVE_DATE_MAX)
            start, end = _clamp_archive_dates(start, end)
            form["start_date"] = start.isoformat()
            form["end_date"] = end.isoformat()
            result = _archive_result(region, start, end)
            if form["compare"] == "1":
                cs = _parse_date(form["compare_start"], start)
                ce = _parse_date(form["compare_end"], end)
                cs, ce = _clamp_archive_dates(cs, ce)
                form["compare_start"] = cs.isoformat()
                form["compare_end"] = ce.isoformat()
                other = _archive_result(region, cs, ce)
                rows = _comparison_rows(result["harm"], other["harm"])
                rows = _overlay_exact_totals(rows, result["summary"], other["summary"])
                result["compare"] = {
                    "label": other["period_label"],
                    "rows": rows,
                    "source_label": other["source_label"],
                    "summary": other["summary"],
                }
            result.pop("harm", None)
    except (FirmsApiError, ArchiveCsvError, ValueError) as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        error = f"Unexpected error: {exc}"

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
