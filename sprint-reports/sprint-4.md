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
- Global datasets are extremely large; full matching is slow and the map uses sampling.
- Matched counts can be low or zero depending on the selected region and exact matching parameters.
- Currently runs locally; cloud deployment was not completed due to data size.
- No automated export of full harmonized tables in the final version.
- Relies on pre-downloaded FIRMS Archive CSVs rather than fully live long-term API access.

## Lessons Learned
- FIRMS Archive is the practical way to obtain multi-year data.
- Harmonization logic must balance accuracy and performance.
- Clear statistics and anomaly highlighting are more valuable to users than raw point counts.

## Future Work
- Optimize matching for larger datasets.
- Deploy a lighter version to Streamlit Community Cloud.
- Add confidence and FRP-based filtering.
- Include a simple downloadable summary report.

## Conclusion
The project meets the core goal of the challenge: a web tool that harmonizes MODIS and VIIRS hotspots and helps users examine fire patterns and unusual activity over time.
