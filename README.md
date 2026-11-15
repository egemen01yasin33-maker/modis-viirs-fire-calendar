# Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.

Dark Streamlit UI. Primary path:

1. **Archive CSV** — drop FIRMS Archive downloads into `data/` (e.g. 2023–2026);
   exact activity chart + sampled map for the selected region.
2. **Live compare** — with `FIRMS_MAP_KEY`, compare that archive span to the
   last 5–10 NRT days (daily averages so the periods stay comparable).

Matched (High Confidence) = within **1 km** and **±6 hours**.

## NASA Space Apps submission links

`.github.io` is **not required**. Judges need any **public HTTPS** links:

| Submission field | What to paste |
| --- | --- |
| **Project Demo** | 7-slide pitch (English): https://egemen01yasin33-maker.github.io/modis-viirs-fire-calendar/ |
| **Final Project** | Source repo: https://github.com/egemen01yasin33-maker/modis-viirs-fire-calendar |
| **NASA Data** | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (MODIS + VIIRS Archive / NRT) |

Open each link in a private window before submit. Live Streamlit hosting is optional for judging; the demo deck + public GitHub satisfy the form.

## Features

- Preset regions (Global, Amazon, Australia, California, Mediterranean,
  Siberia, Indonesia, Turkey sub-regions) + place name + manual bbox
- Activity chart above the map (Monthly/Weekly/Daily auto for long spans)
- Archive vs live NRT comparison (MAP_KEY)
- Anomaly days (>2× mean activity)
- Controls at the bottom (no left sidebar)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set FIRMS_MAP_KEY for NRT mode
streamlit run streamlit_app.py --server.port 8529
# or ASGI (same entry Cursor publish uses):
# uvicorn app:app --host 0.0.0.0 --port 8529
```

### Archive CSVs

Place NASA FIRMS Archive CSVs in `data/` (gitignored except the sample).

**Full dataset (~6.5 GB, MODIS + VIIRS)** is on a GitHub Release (not in git):

- Release: https://github.com/egemen01yasin33-maker/modis-viirs-fire-calendar/releases/tag/archive-data-v1
- Download:

```bash
bash scripts/download_archives.sh
# or:
gh release download archive-data-v1 -R egemen01yasin33-maker/modis-viirs-fire-calendar -D data
```

Required columns: `latitude`, `longitude`, `acq_date`, `acq_time`,
`confidence`, `satellite`.

In Archive mode pick a region and a date window that falls inside your CSV
coverage, then click **Apply / Refresh**. Worldwide multi-year dumps are not
loaded into RAM in full.

### Live NRT compare

1. Get a [FIRMS MAP Key](https://firms.modaps.eosdis.nasa.gov/api/key/).
2. Put `FIRMS_MAP_KEY=...` in `.env`.
3. Keep **Compare with live FIRMS** on, pick 5–10 days — the app fetches that
   window (chunked if needed) and shows archive vs live daily averages.

Without archive CSVs the app falls back to live-only NRT mode.

## Layout

```
app.py                 # ASGI entry (st.App)
streamlit_app.py       # Streamlit UI
src/firms_api.py       # FIRMS Area API client
src/harmonize.py       # Distance / time matching
src/archive_csv.py     # Chunked archive loaders
src/analytics.py       # Anomalies + comparison
data/                  # Archive CSVs (large files gitignored)
tests/
```

## Harmonization rule

A MODIS and VIIRS point are *Matched* when both hold:

- Haversine distance ≤ 1 km
- `|acq_datetime_MODIS − acq_datetime_VIIRS|` ≤ 6 hours

Greedy bipartite match: each MODIS takes the nearest unused VIIRS in its
time window.

## Pitch deck (NASA Space Apps)

7-slide fullscreen HTML deck (Turkish + technical English terms):

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory presentation
# open http://127.0.0.1:8765/
```

Keys: ← → / Space · `F` fullscreen.

## Tests

```bash
python tests/test_harmonize.py
python tests/test_firms_api.py
python tests/test_archive_csv.py
```

## Security

`FIRMS_MAP_KEY` lives in `.env` (gitignored). It is not logged, not stored in
`session_state`, and not echoed in error messages.
