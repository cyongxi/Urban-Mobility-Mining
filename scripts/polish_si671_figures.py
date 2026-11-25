"""Create report-ready SI 671 figures from already saved aggregate CSVs.

Usage from repo root:
    python polish_si671_figures.py

Only creates *_report.png images. Never edits training notebooks or source results.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def csv(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f'Missing existing result CSV: {path}')
    return pd.read_csv(path)


def three_panel_heatmaps(data, name_column, value_column, names, output, scale=1.0, cbar_label='Normalized share'):
    fig = plt.figure(figsize=(15.5, 4.3), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    maxima = data[value_column].max() * scale
    if not np.isfinite(maxima) or maxima <= 0:
        raise ValueError('Nonpositive or nonfinite heatmap values')
    last_im = None
    for i, (name, title) in enumerate(names):
        subset = data.loc[data[name_column] == name]
        if len(subset) != 168:
            raise ValueError(f'Expected exactly 168 slots for {name}, got {len(subset)}')
        grid = subset.pivot(index='day_of_week', columns='hour', values=value_column)
        grid = grid.reindex(index=range(7), columns=range(24))
        if grid.isna().any().any():
            raise ValueError(f'Missing weekday-hour slot for {name}')
        ax = axes[i]
        last_im = ax.imshow(grid.to_numpy() * scale, cmap='YlOrRd', vmin=0, vmax=maxima, aspect='auto')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        ax.set_xlabel('Hour (NYC local time)')
        ax.set_yticks(range(7))
        if i == 0:
            ax.set_yticklabels(DAYS)
            ax.set_ylabel('Day of week')
        else:
            ax.set_yticklabels([])
    fig.colorbar(last_im, cax=cax, label=cbar_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)
    print('Created:', output)


def cross_service_heatmap():
    relative = 'results/cross_service'
    data = csv(f'{relative}/temporal_profile.csv')
    three_panel_heatmaps(
        data, 'service', 'normalized_share',
        [('Yellow', 'Yellow Taxi'), ('Uber', 'Uber'), ('Lyft', 'Lyft')],
        ROOT / relative / 'figures/weekday_hour_heatmaps_report.png',
        scale=100,
        cbar_label='Share of normalized typical week (%)'
    )


def rq2_heatmap():
    relative = 'results/temporal'
    data = csv(f'{relative}/tables/cluster_profiles.csv')
    meta = csv(f'{relative}/tables/cluster_summary.csv')
    labels = [('0', 'C0 · Morning-peaking'), ('1', 'C1 · Broad activity'), ('2', 'C2 · Weekend late-night')]
    labels = [(int(cid), f'{title} (n={int(meta.loc[meta.cluster_id == int(cid), "n_zones"].iloc[0])})') for cid, title in labels]
    three_panel_heatmaps(
        data, 'cluster_id', 'mean_normalized_share', labels,
        ROOT / relative / 'figures/rq2_cluster_168d_heatmaps_report.png',
        scale=100,
        cbar_label='Mean zone-normalized share (%)'
    )


def compact_group_label(grouping, label):
    if grouping == 'demand_group':
        return label
    if grouping == 'rq1_community':
        m = re.search(r'Community\s+(\d+).*?(\d+)\s+zones', label)
        if m:
            return f'C{m.group(1)} ({m.group(2)} zones)'
        return 'No OD' if label == 'No OD trips' else label
    if label.startswith('C0 '):
        return 'C0 AM-peak'
    if label.startswith('C1 '):
        return 'C1 broad activity'
    if label.startswith('C2 '):
        return 'C2 late-night'
    if label.startswith('Unclustered'):
        return 'Unclustered'
    return label


def rq4_group_gain():
    relative = 'results/forecasting/final_rq3_rq4'
    df = csv(f'{relative}/rq4_test_bootstrap_by_group.csv')
    selections = [
        ('demand_group', 'Training demand'),
        ('rq1_community', 'RQ1 mobility community'),
        ('rq2_cluster', 'RQ2 temporal profile')
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained', sharey=True)
    for ax, (grouping, title) in zip(axes, selections):
        d = df.loc[(df.grouping == grouping) & (df.group != 'ALL')].copy()
        if d.empty:
            raise ValueError(f'No group data for {grouping}')
        y = d['observed_abs_MAE_reduction'].to_numpy(dtype=float)
        lo = d['ci_lower_2.5pct'].to_numpy(dtype=float)
        hi = d['ci_upper_97.5pct'].to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError('Nonfinite gain/interval')
        x = np.arange(len(d))
        ax.errorbar(x, y, yerr=np.vstack([y-lo, hi-y]), fmt='o', capsize=4, elinewidth=1.6)
        ax.axhline(0, linewidth=1, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels([compact_group_label(grouping, str(z)) for z in d.group], rotation=17, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Test MAE reduction (M1 − M2)')
    fig.suptitle('Where historical OD strength helps (95% daily-block intervals)', fontsize=14)
    output = ROOT / relative / 'figures/rq4_gain_by_group_report.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches='tight')
    plt.close(fig)
    print('Created:', output)


if __name__ == '__main__':
    cross_service_heatmap()
    rq2_heatmap()
    rq4_group_gain()
