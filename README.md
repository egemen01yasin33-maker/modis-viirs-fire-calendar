# Unified Fire Activity Calendar

**NASA Space Apps Challenge 2026**  
Challenge: **Harmonization of MODIS and VIIRS Hot Spots**

Interactive Streamlit application that harmonizes active fire hotspot data from MODIS and VIIRS satellites into a single burning activity calendar.

## Features
- Direct FIRMS API integration (MAP_KEY required)
- Spatial + temporal matching of MODIS and VIIRS detections (1 km / ±6 hours)
- Interactive map (Matched / MODIS-only / VIIRS-only)
- Fire activity calendar (daily/weekly)
- Basic statistics

## Data Source
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/

## How to Run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Enter your FIRMS MAP_KEY in the sidebar

## Project Structure
- `sprint-reports/` — Sprint reports
- `docs/` — Documentation
- `app.py` — Main application
- `requirements.txt`
