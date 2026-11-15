# Sprint 3 Report

**Project:** Unified Fire Activity Calendar  
**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**NASA Space Apps Challenge 2026**

## What we built
An interactive Streamlit web application with the following features:

- **Region selection:** Preset regions, place name search, or custom area.
- **Date range picker** with optional **two-period comparison** mode.
- **Interactive map** (OpenStreetMap) showing hotspot locations.
- **Charts:** Daily and monthly fire activity.
- **Anomaly detection:** Highlights unusually busy days.
- **FRP (Fire Radiative Power)** summary statistics.
- **Harmonization indicators:** Total, Matched, MODIS-only, VIIRS-only counts.
- **Busy days** summary.

## Key capabilities
- Users can compare two different time periods side-by-side.
- Unusually high activity days are automatically flagged.
- Statistics update based on the selected region and date range.
- Designed to support early awareness of elevated fire activity.

## Current limitations
- Large global datasets require sampling for map display.
- Matching performance can be improved for very large selections.
- Deployment is currently local; cloud deployment is planned if time allows.

## Next steps
- Refine the matching algorithm so Matched counts are accurate.
- Add more analytical views (trends, quality score).
- Prepare demo video and final documentation.
