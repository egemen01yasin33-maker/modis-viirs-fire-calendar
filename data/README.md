# FIRMS Archive CSV folder

Place NASA FIRMS Archive CSV downloads here (MODIS and/or VIIRS), **or** restore
the full set from Google Drive using `archive_manifest.json`:

```bash
python -c "from pathlib import Path; from src.archive_restore import restore_archives; print(restore_archives(Path('data')))"
```

In the UI, when `data/` is empty, use **Restore full archives from Drive**.

Required columns:

`latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `satellite`

In Archive mode the app streams every `*.csv` here and filters by region + dates.
Multi-GB worldwide dumps are supported; keep Drive links as Anyone-with-the-link
so cloud machines can restore without re-uploading.

Download archive data from:
https://firms.modaps.eosdis.nasa.gov/download/

`sample_firms.csv` is a tiny fixture for unit tests — safe to keep.
`archive_manifest.json` lists the full MODIS + VIIRS set (Drive IDs).
