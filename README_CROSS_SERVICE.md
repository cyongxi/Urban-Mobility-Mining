# SI 671: Cross-Service Supplementary Analysis

## Scope
A June 2025 **descriptive** Yellow Taxi / Uber / Lyft pickup comparison. Does not alter RQ1-RQ4 training and uses only pickup timestamps and PULocationID.

## Setup (from repository root)
```bash
conda activate si671-urban
python -m pip install duckdb pandas numpy matplotlib pytest
mkdir -p data/external/hvfhv
curl -fL --retry 3 https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-06.parquet -o data/external/hvfhv/fhvhv_tripdata_2025-06.parquet
```

Existing inputs:
- `data/processed/cleaned_trips/cleaned_2025-06.parquet`
- `data/external/taxi_zone_lookup.csv`

## Run
```bash
python -m pytest tests/test_cross_service_comparison.py -q
python -m src.supplementary.cross_service_comparison
```

Or open `notebooks/10_cross_service_comparison.ipynb` to view summaries and figures, without recomputing when CSV outputs already exist.

## Outputs (`results/cross_service/`)
- `quality_checks.csv`: month-filtered counts and excluded zones by service
- `zone_counts.csv`: one row per service/eligible zone
- `borough_share.csv`: fraction of eligible pickups by area (EWR is NOT one of NYC's five boroughs)
- `temporal_profile.csv`: complete 7x24 grid per service, including averaged hourly demand and normalized typical-week shape
- `summary.csv`: pickup totals, active zones, Manhattan share, EWR share, Top10 concentration, weekend typical-week share, Spearman zone-share similarity
- `analysis_metadata.json`: scope and definitions
- `figures/borough_pickup_share.png`
- `figures/weekday_hour_heatmaps.png`

## Comparability and caveats
- June 2025, same month and TLC pickup zones excluding IDs 264 and 265. Zone 1 (EWR) is retained and labeled separately; it should not be described as one of the five NYC boroughs.
- Yellow uses existing course cleaned records; Uber/Lyft use raw official HVFHV. Both are restricted by pickup time and eligible pickup zones. The QA table makes zone exclusions explicit. Other service-specific data quality issues may remain.
- Normalized temporal profiles use a *typical week* with weekday counts divided by the actual number of calendar days of that weekday. This is **not** the same as trip-share across all calendar days.
- Results are descriptive and month-specific, not causal evidence of displacement, and do not establish representativeness for all transportation services.
- Official data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf
- Official downloadable data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Git
Do not commit raw HVFHV/Yellow trip Parquet data. Commit only this code, notebook, tests, README, and aggregate results under `results/cross_service/`.
