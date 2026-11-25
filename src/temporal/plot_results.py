"""Plot final RQ2 tables and optional NYC taxi-zone shapefile.

Run from repo root: python -m src.temporal.plot_results
"""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_profiles import OUTPUT_DIR, ROOT
FIG_DIR = ROOT / "results/temporal/figures"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def write_figures(table_dir=OUTPUT_DIR, fig_dir=FIG_DIR, shape_file=None):
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv(table_dir/"k_selection.csv")
    pca = pd.read_csv(table_dir/"pca_explained_variance.csv")
    pcs = pd.read_csv(table_dir/"pca_zone_scores.csv")
    curves = pd.read_csv(table_dir/"cluster_profiles.csv")
    labels = pd.read_csv(table_dir/"zone_temporal_clusters.csv")
    summary = pd.read_csv(table_dir/"cluster_summary.csv")

    def save(fig, filename):
        fig.savefig(fig_dir/filename, dpi=190, bbox_inches="tight")
        plt.close(fig)
        print("Saved", fig_dir/filename)

    fig, ax = plt.subplots(figsize=(7,4.2))
    ax.plot(k.k, k.silhouette, marker="o", linewidth=2)
    chosen = int(k.loc[k.selected.astype(bool), "k"].iloc[0])
    ax.axvline(chosen, linestyle="--", color="tab:red", label=f"Selected K={chosen}")
    ax.legend();ax.grid(alpha=.25)
    ax.set(title="Silhouette score (retained PCA space)", xlabel="K", ylabel="Silhouette score")
    save(fig,"rq2_silhouette_k.png")

    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.plot(pca.component, pca.cumulative_variance_ratio)
    ax.axhline(.9, color="tab:red", linestyle="--", label="90% target")
    ax.set(title="PCA cumulative explained variance", xlabel="Number of components",
           ylabel="Cumulative variance", ylim=(0,1.02))
    ax.grid(alpha=.25); ax.legend()
    save(fig,"rq2_pca_variance.png")

    fig,ax=plt.subplots(figsize=(7,5.4))
    for c,group in pcs.groupby("cluster_id"):
        ax.scatter(group.PC1,group.PC2,s=30,alpha=.8,label=f"C{int(c)}")
    ax.set(title="RQ2 temporal clusters — first two PCs",xlabel="PC1",ylabel="PC2")
    ax.legend(bbox_to_anchor=(1,1),loc="upper left")
    save(fig,"rq2_pca_scatter.png")

    cids=sorted(curves.cluster_id.unique())
    nrows=math.ceil(len(cids)/2)
    fig,axs=plt.subplots(nrows,2,figsize=(12,max(3.2,3.2*nrows)),squeeze=False)
    limit=curves.mean_normalized_share.max()*100
    mappable=None
    for i,cid in enumerate(cids):
        ax=axs[i//2,i%2]
        sub=curves.loc[curves.cluster_id==cid]
        matrix=sub.pivot(index="day_of_week",columns="hour",values="mean_normalized_share")
        matrix=matrix.reindex(index=range(7),columns=range(24))
        mappable=ax.imshow(matrix.values*100,aspect="auto",vmin=0,vmax=limit,cmap="YlOrRd")
        ax.set(title=f"Cluster {int(cid)} · {int(summary.loc[summary.cluster_id==cid,'n_zones'].iloc[0])} zones")
        ax.set_yticks(range(7),DAYS)
        ax.set_xticks([0,4,8,12,16,20,23])
    for i in range(len(cids),nrows*2):axs[i//2,i%2].axis("off")
    fig.subplots_adjust(hspace=.4, wspace=.25)
    fig.colorbar(mappable,ax=axs.ravel().tolist(),shrink=.8,label="Mean zone-normalized profile share (%)")
    save(fig,"rq2_cluster_168d_heatmaps.png")

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for cid in cids:
        sub=curves.loc[curves.cluster_id==cid]
        for ax,sel in [(ax1,sub.day_of_week<5),(ax2,sub.day_of_week>=5)]:
            g=sub.loc[sel].groupby("hour").mean_normalized_share.mean()*100
            ax.plot(g.index,g.values,label=f"C{int(cid)}")
    ax1.set_title("Weekday hourly shape");ax2.set_title("Weekend hourly shape")
    ax1.set_ylabel("Mean share of normalized weekly profile (%)")
    for ax in (ax1,ax2):ax.set_xlabel("Hour");ax.grid(alpha=.2)
    ax2.legend(bbox_to_anchor=(1,1),loc="upper left")
    save(fig,"rq2_cluster_time_curves.png")

    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(summary.cluster_id.astype(str),summary.n_zones)
    ax.set(title="Cluster membership",xlabel="Cluster",ylabel="Number of zones")
    save(fig,"rq2_cluster_sizes.png")

    cross=table_dir/"community_cluster_crosstab.csv"
    if cross.exists():
        matrix=pd.read_csv(cross).pivot(index="community_id",columns="cluster_id",values="pct_within_community").fillna(0)
        fig,ax=plt.subplots(figsize=(7,4))
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap="Blues",aspect="auto")
        ax.set_xticks(range(len(matrix.columns)),[str(int(i)) for i in matrix.columns])
        ax.set_yticks(range(len(matrix.index)),[str(int(i)) for i in matrix.index])
        ax.set(xlabel="RQ2 temporal cluster",ylabel="RQ1 mobility community",title="RQ1 x RQ2 zone memberships")
        for (i,j),value in np.ndenumerate(matrix.to_numpy()):
            ax.text(j,i,f"{value:.0%}",ha="center",va="center",color="white" if value>.5 else "black")
        fig.colorbar(im,ax=ax,label="Fraction of RQ1 community")
        save(fig,"rq1_rq2_overlap.png")

    shp = Path(shape_file) if shape_file else None
    if shp is None:
        found=sorted((ROOT/"data/external").rglob("taxi_zones.shp"))
        shp=found[0] if found else None
    if shp is None or not shp.exists():
        print("No official taxi_zones.shp — map skipped (other plots created)")
        return
    import geopandas as gpd
    geoms=gpd.read_file(shp)
    key="LocationID" if "LocationID" in geoms.columns else "OBJECTID"
    if key not in geoms.columns:
        raise ValueError("Shapefile lacks LocationID/OBJECTID")
    geoms[key]=pd.to_numeric(geoms[key]).astype(int)
    geoms=geoms.loc[~geoms[key].isin([264,265])]
    mapped=geoms.merge(labels[["zone_id","cluster_id"]],left_on=key,right_on="zone_id",how="left",validate="one_to_one")
    mapped["cluster_id"] = mapped["cluster_id"].astype(float)
    fig,ax=plt.subplots(figsize=(8,8))
    mapped.plot(column="cluster_id",categorical=True,legend=True,ax=ax,cmap="tab10",
                edgecolor="white",linewidth=.25,
                missing_kwds={"color":"lightgrey","label":"Unassigned"})
    ax.set_axis_off();ax.set_title("NYC taxi zones: 2025 temporal clusters")
    save(fig,"rq2_temporal_cluster_map.png")


if __name__ == "__main__":
    write_figures()
