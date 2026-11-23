# SI 671 RQ2 — Sparsity sensitivity add-on

From a clean working tree on `feature/temporal-clustering`, place
`src/temporal/rq2_sparsity_sensitivity.py` into your repository (do not replace existing RQ2 files).

Fix the single deprecated Pandas alias in `src/temporal/build_profiles.py`:

```python
counts = pd.Series(dates.day_of_week).value_counts()
```

The alias change does not alter the numerical result.

Run from the repository root, after the original `build_profiles.py` has generated its tables:

```bash
python -m src.temporal.rq2_sparsity_sensitivity
```

This reuses existing 263x168 average Pickup Profile CSVs, not original trip-level parquet data. It fits separate PCA + K-means for each minimum included Pickup count (1/100/500/1000/5000). Each scenario saves tables to a separate directory, and the combined comparison lives in:

```
results/temporal/sensitivity/rq2_sparsity_sensitivity_summary.csv
```

The existing primary `results/temporal/tables` and `results/temporal/figures` are **never overwritten**. Do not consider K=2 with sizes 259+1 and silhouette .845 an interpretable clustering merely because the score is high. Review size balance, silhouette, stability, coverage, and actual profiles. If a scenario fails its >=5 minimum-cluster-size preference, label it as not yet suitable for substantive 'demand type' conclusions.

For figures from a scenario, use an existing helper:

```bash
python - <<'PY'
from pathlib import Path
from src.temporal.plot_results import write_figures
path = Path('results/temporal/sensitivity/min1000')
write_figures(table_dir=path, fig_dir=path/'figures')
PY
```

For `results/temporal/sensitivity/min1000/`, the normal `plot_results.py` will read the sensitivity RQ1/RQ2 cross-tab if it was produced. The same geometry lookup logic applies for maps.

Before pushing:

```bash
python -m pytest tests/test_rq2_profiles.py -q
git status --short
git add src/temporal/rq2_sparsity_sensitivity.py src/temporal/build_profiles.py results/temporal/sensitivity/
git diff --cached --stat
git commit -m 'Add RQ2 sparse-zone sensitivity analysis and fix deprecated Pandas alias'
git push origin feature/temporal-clustering
```
