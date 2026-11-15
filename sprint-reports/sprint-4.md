# Sprint 4 Report

**Project:** Unified Fire Activity Calendar  
**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**NASA Space Apps Challenge 2026**  
**Date:** 15 November 2026

## Results
We delivered a working web application that harmonizes MODIS and VIIRS active fire data:

- Loads NASA FIRMS Archive data (MODIS + VIIRS, 2023–2026).
- Performs spatial-temporal matching (1 km + ±6 hours) and separates Matched / MODIS-only / VIIRS-only detections.
- Interactive map, date range selection, two-period comparison, anomaly detection (busy days), and FRP statistics.
- Supports preset regions and custom areas.
- **Public demo available** via temporary Cloudflare tunnel:  
  https://sally-ultra-anonymous-tobago.trycloudflare.com/

The prototype demonstrates a unified burning activity calendar that combines the two sensors.

## Limitations
- Full global datasets are too large for Streamlit Community Cloud (memory limits).
- Map uses sampling on very large selections, so visual density may not match full statistics.
- Matched counts depend on region, time window, and matching parameters.
- Current public link is a temporary Cloudflare tunnel (not a permanent hosted deployment).
- Relies on pre-processed regional Parquet files for practical performance.

## Lessons Learned
- FIRMS Archive is the practical source for multi-year data.
- Regional pre-processing (Parquet) makes the app usable while keeping full statistics.
- Clear anomaly highlighting and comparison features are more valuable than raw point counts.
- Permanent cloud deployment of large geospatial datasets requires careful data reduction or external storage.

## Future Work
- Optimize further for larger areas.
- Move to a permanent hosting solution with external data storage.
- Add confidence / FRP filters and better export options.
- Improve map sampling transparency.

## Conclusion
The project meets the core goal of the challenge: a web tool that harmonizes MODIS and VIIRS hotspots and helps users examine fire patterns and unusual activity over time. A live demo link is available for judges.
