# Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS & VIIRS hot spot harmonization.

Dark turquoise / black Streamlit UI. Two data sources:

1. **Archive CSV** — drop FIRMS Archive downloads into `data/`; the app streams
   multi-GB files and keeps only the selected region + date range.
2. **NRT API** — live FIRMS Area API via `FIRMS_MAP_KEY` in `.env`.

Matched (High Confidence) = within **1 km** and **±6 hours**.

## Features

- Preset regions (Global, Amazon, Australia, California, Mediterranean,
  Siberia, Indonesia, Turkey sub-regions) + place name + manual bbox
- Auto sensors: MODIS_NRT + VIIRS NOAA-20/21 (NRT mode)
- Anomaly days (>2× mean daily activity)
- Comparison mode (two date ranges, same region)
- OpenStreetMap Folium map + Plotly activity calendar
- Controls at the bottom (no left sidebar)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set FIRMS_MAP_KEY for NRT mode
streamlit run app.py --server.port 8529
```

### Archive CSVs

Place NASA FIRMS Archive CSVs in `data/` (gitignored except the sample). Example:

- `archive_modis_terra_aqua.csv`
- `archive_viirs_noaa20.csv`
- `archive_viirs_noaa21.csv`
- `archive_viirs_snpp.csv`

Required columns: `latitude`, `longitude`, `acq_date`, `acq_time`,
`confidence`, `satellite`.

In Archive mode pick a region and a date window that falls inside your CSV
coverage, then click **Apply / Refresh**. Worldwide multi-year dumps are not
loaded into RAM in full.

### NRT API

1. Get a [FIRMS MAP Key](https://firms.modaps.eosdis.nasa.gov/api/key/).
2. Put `FIRMS_MAP_KEY=...` in `.env`.
3. Choose region + dates (≤10 days) and click **Fetch Data**.

## Layout

```
app.py                 # Streamlit UI
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
