# FIRMS Archive CSV folder

Place NASA FIRMS Archive CSV downloads here (MODIS and/or VIIRS).

Required columns:

`latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `satellite`

In **Archive CSV** mode the app lists every `*.csv` in this folder, then
**streams** them on Apply / Refresh and keeps only rows inside the selected
region + date range. Multi-GB worldwide dumps are supported without loading
the entire file into memory.

Suggested filenames (optional):

- `archive_modis_terra_aqua.csv`
- `archive_viirs_noaa20.csv`
- `archive_viirs_noaa21.csv`
- `archive_viirs_snpp.csv`

Download archive data from:
https://firms.modaps.eosdis.nasa.gov/download/

`sample_firms.csv` is a tiny fixture for unit tests — safe to keep.
