# Sprint 1 Report

**Project:** Unified Fire Activity Calendar  
**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**NASA Space Apps Challenge 2026**  
**Date:** 14 November 2026

## Challenge Selection
We selected the **Harmonization of MODIS and VIIRS Hot Spots** challenge because fire managers and researchers cannot directly compare long-term records from two different satellite sensors.

- **MODIS** (Terra/Aqua): ~1 km resolution, continuous data since 2000.
- **VIIRS** (NOAA-20 / NOAA-21): 375 m resolution, higher sensitivity, designed as the continuity mission.

Without harmonization, real fire trends are mixed with sensor differences.

## Project Idea
Build a web application that:
1. Loads active fire hotspot data from NASA FIRMS (MODIS + VIIRS).
2. Harmonizes the records with spatial-temporal matching (1 km distance + ±6 hour window).
3. Presents a unified **burning activity calendar** for any selected region and time period.
4. Highlights anomalies, supports two-period comparison, and shows key statistics (including FRP).

## Goals
- Deliver a working prototype that combines MODIS and VIIRS into one consistent view.
- Let users explore historical patterns and detect unusual activity.
- Keep the interface simple for non-experts while remaining useful for analysis.

## Initial Decisions
- Data source: NASA FIRMS Archive Download + NRT API.
- Stack: Python, Streamlit, Folium / OpenStreetMap.
- Scope: Global capability with practical limits on very large datasets.

## Completed in Sprint 1
- Switched to this challenge and created the repository.
- Defined core features: data loading, matching, interactive map, calendar, and basic statistics.

## Next Steps
- Implement data loading and matching logic.
- Build the first interactive dashboard.
- Test with real FIRMS archive data.
