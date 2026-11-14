# Sprint 2 Report

**Project:** Unified Fire Activity Calendar  
**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**NASA Space Apps Challenge 2026**

## What we did
- Used NASA FIRMS Archive Download to obtain long-term active fire data (2023–2026).
- Downloaded CSV files for:
  - MODIS C6.1
  - VIIRS NOAA-20
  - VIIRS NOAA-21
- Implemented spatial-temporal harmonization:
  - Points within **1 km** and **±6 hours** are labeled as **Matched (High Confidence)**.
  - Unmatched detections are retained as MODIS-only or VIIRS-only.
- Built a Streamlit web application that loads these CSVs and performs the matching.

## Why it matters
MODIS (1 km) and VIIRS (375 m) have different resolutions and sensitivities, so their records cannot be compared directly. Harmonization creates a consistent time series, allowing users to focus on fire patterns, trends, and anomalies rather than sensor differences.

## Issues encountered
- Global archive files are very large (tens to hundreds of millions of points).
- Full matching on the entire dataset is computationally heavy; the map uses a sample while statistics use the full selection.
- Matched count can appear as 0 if the matching logic or sampling needs adjustment.

## Next steps
- Optimize matching performance.
- Add clearer quality metrics for the harmonization.
