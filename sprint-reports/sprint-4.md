# Sprint 4 Report

**Project:** Unified Fire Activity Calendar  
**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**NASA Space Apps Challenge 2026**  
**Date:** 15 November 2026

## Results
We delivered a working Streamlit web application that:

- Loads long-term NASA FIRMS active fire data (MODIS + VIIRS, 2023–2026).
- Performs spatial-temporal harmonization (1 km + ±6 hours) and separates Matched / MODIS-only / VIIRS-only detections.
- Provides an interactive map, date range selection, and two-period comparison.
- Detects unusually busy days (anomalies).
- Shows summary statistics including total hotspots, matched count, and FRP.
- Supports preset regions, place names, and custom areas.

The prototype demonstrates that MODIS and VIIRS records can be combined into a single, explorable burning activity calendar.

## Limitations
- Global datasets are extremely large. For performance, the map displays a sample of points (e.g. 50,000), so the map can look inconsistent or incomplete compared with the full statistics.
- Matched counts can be low or zero depending on the selected region, time window, and exact matching parameters.
- Currently runs locally; cloud deployment was not completed due to data size.
- Relies on pre-downloaded FIRMS Archive CSVs for multi-year analysis.

## Lessons Learned
- FIRMS Archive is the practical way to obtain multi-year data.
- Harmonization logic must balance accuracy and performance.
- Clear statistics and anomaly highlighting are more valuable to users than raw point counts.
- Map sampling is necessary for large areas but should be clearly communicated to the user.

## Future Work
- Improve performance on large datasets and make map sampling more transparent.
- Deploy a lighter version to Streamlit Community Cloud.
- Add confidence and FRP-based filtering.
- Include a simple downloadable summary report.

## Conclusion
The project meets the core goal of the challenge: a web tool that harmonizes MODIS and VIIRS hotspots and helps users examine fire patterns and unusual activity over time.
