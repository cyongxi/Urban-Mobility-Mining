"""Aggregate 2025 cleaned yellow-taxi trips to reusable OD edge lists.
Run from repo root: python -m src.network.build_od_network
"""
from pathlib import Path
import argparse
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / 'data/processed/cleaned_trips'
ZONE_FILE = ROOT / 'data/external/taxi_zone_lookup.csv'
OUT = ROOT / 'results/network/tables'
PERIODS = ('annual', 'weekday', 'weekend')


def make_view(con, paths):
    # All paths are local files discovered by glob, not supplied by end users.
    quoted = ', '.join("'" + str(p).replace("'", "''") + "'" for p in paths)
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet([{quoted}])")


def aggregate_od(clean_dir=CLEAN_DIR, zone_file=ZONE_FILE, out=OUT):
    clean_dir, zone_file, out = map(Path, (clean_dir, zone_file, out))
    paths = sorted(clean_dir.glob('cleaned_2025-*.parquet'))
    if len(paths) != 12:
        raise FileNotFoundError(f'Expected 12 cleaned_2025-*.parquet files, got {len(paths)} in {clean_dir}')
    zones = pd.read_csv(zone_file)
    required = {'LocationID', 'Borough', 'Zone'}
    if not required.issubset(zones.columns):
        raise ValueError(f'Zone lookup missing columns: {required - set(zones.columns)}')
    zones = zones.loc[~zones['LocationID'].isin([264, 265])].copy()
    zones['LocationID'] = zones['LocationID'].astype(int)
    if zones['LocationID'].duplicated().any():
        raise ValueError('Duplicated LocationID in zone lookup')
    out.mkdir(parents=True, exist_ok=True)
    zones.to_csv(out / 'zone_lookup_used.csv', index=False)
    con = duckdb.connect()
    try:
        con.execute('SET threads = 4')
        make_view(con, paths)
        con.register('valid_zones', zones[['LocationID']])
        # DuckDB ISODOW: Monday=1 ... Sunday=7. Dates restricted to calendar 2025.
        # GROUP BY weekend first; the annual table is derived by sum to avoid rescanning.
        by_type = con.execute('''
            SELECT CAST(t.PULocationID AS INTEGER) AS origin,
                   CAST(t.DOLocationID AS INTEGER) AS destination,
                   (date_part('isodow', t.tpep_pickup_datetime) IN (6,7)) AS is_weekend,
                   COUNT(*)::BIGINT AS trip_count
            FROM trips t
            JOIN valid_zones p ON t.PULocationID = p.LocationID
            JOIN valid_zones d ON t.DOLocationID = d.LocationID
            WHERE t.valid_od IS TRUE
              AND t.tpep_pickup_datetime >= TIMESTAMP '2025-01-01'
              AND t.tpep_pickup_datetime < TIMESTAMP '2026-01-01'
            GROUP BY 1,2,3
        ''').df()
    finally:
        con.close()
    if by_type.empty:
        raise ValueError('No OD-valid trips after filters. Verify cleaned parquet schema.')
    result = {}
    for period in PERIODS:
        if period == 'annual':
            edges = by_type.groupby(['origin', 'destination'], as_index=False)['trip_count'].sum()
        else:
            edges = by_type.loc[by_type['is_weekend'].eq(period == 'weekend'),
                                ['origin', 'destination', 'trip_count']].copy()
        edges = edges.sort_values(['trip_count', 'origin', 'destination'],
                                  ascending=[False, True, True]).reset_index(drop=True)
        inter = edges.loc[edges['origin'] != edges['destination']].copy()
        edges.to_csv(out / f'od_{period}_all.csv', index=False)
        inter.to_csv(out / f'od_{period}_interzone.csv', index=False)
        result[period] = edges
        print(f'{period:8s} trips={int(edges.trip_count.sum()):,}  pairs={len(edges):,}  '
              f'interzone_trips={int(inter.trip_count.sum()):,}')
    # Reconciliation is an invariant, not a rough expected value.
    a, w, e = [int(result[p].trip_count.sum()) for p in PERIODS]
    assert a == w + e, 'Annual counts must equal weekday + weekend'
    print('Saved OD tables to:', out)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', type=Path, default=CLEAN_DIR)
    parser.add_argument('--zone-file', type=Path, default=ZONE_FILE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    aggregate_od(args.clean_dir, args.zone_file, args.out)
