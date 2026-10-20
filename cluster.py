"""K-means clustering of block groups for a chosen set of states.

    python cluster.py --states 36 34 09 --k 5

Input: data/processed/features_standardized.parquet (transformed + imputed + national z-scores)
and feature_metadata.csv (concept group weights).

Scaling (--scale):
  * regional (default): re-z-score every feature within the selected states, with each
    pooled-scale block (the three ECEF coordinates; each compositional share block) sharing
    one pooled scale (as in standardize.py, per feature_metadata `scale_block`). Use this for a regional
    run: under national scaling a tri-state area spans a tiny slice of each feature's range
    (especially distance), so those features would barely influence the clusters.
  * national: use the national z-scores as-is (clusters comparable across regional runs).
Then multiply by the per-column group weights so each concept contributes equal variance.

Outputs in outputs/cluster_<states>_k<k>/:
  assignments.csv   GEOID, cluster (clusters numbered by descending size)
  profile.csv       per-cluster medians of raw features + BG/population counts
  centroids_z.csv   cluster centers in (unweighted) z units, to see what defines each cluster
  map.png           block groups colored by cluster (colorblind-safe), white BG boundaries,
                    land BGs not clustered (excluded rows, absent from ACS) in gray;
                    water-only BGs (ALAND = 0) are left blank so lakes/sounds don't read as land
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import PROCESSED, RAW, ROOT

ECEF = ["ecef_x_km", "ecef_y_km", "ecef_z_km"]
# Okabe-Ito colorblind-safe palette (black dropped: it hides the boundaries). Past 7
# clusters, Paul Tol "muted" colors are appended; those are less reliably distinguishable.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
           "#882255", "#332288", "#999933"]
EXCLUDED_COLOR = "#BDBDBD"  # BGs present in TIGER but not clustered
EDGE_COLOR = "white"
PROFILE_COLS = [
    "population", "pop_density_per_sqkm", "median_hh_income", "median_age",
    "race_nh_white_share", "race_nh_black_share", "race_hispanic_share", "race_nh_asian_share",
    "lang_tract_english_only_share", "renter_share", "units_1_detached_share", "units_20plus_share",
    "zero_vehicle_hh_share", "median_commute_min", "hh_with_children_share",
    "school_quality_index", "violent_crime_rate", "property_crime_rate",
    "eff_property_tax_rate_tract", "occ_mgmt_prof_share", "occ_service_share",
    "occ_production_transp_share",
]


def regional_matrix(z: pd.DataFrame, meta: pd.DataFrame, scale: str,
                    concept_weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Rescale (see --scale) and apply column weights. concept_weights overrides the fitted
    CONCEPT_WEIGHTS: column weight = concept_weights[group] / sqrt(n_cols_in_group)."""
    x = z.copy()
    m = meta.set_index("column").reindex(x.columns)
    if scale == "regional":
        # pooled-scale blocks (ECEF, compositional shares) share one sd, as in standardize.py
        block = m["scale_block"] if "scale_block" in m else pd.Series(
            np.where(m.index.isin(ECEF), "ecef", None), index=m.index)
        x = x - x.mean()
        sd = x.std(ddof=0).replace(0, 1)
        for b, cols in block.dropna().groupby(block.dropna()).groups.items():
            sd[cols] = np.sqrt(x[cols].var(ddof=0).mean())
        x = x / sd
    if concept_weights is None:
        w = m["group_weight"]
    else:
        w = m["group"].map(concept_weights) / np.sqrt(m["n_cols_in_group"])
    return x * w


def load_bg_shapes(states: list[str]) -> gpd.GeoDataFrame:
    """Every TIGER 2023 land block group in `states` (incl. ones never clustered)."""
    frames = []
    for st in states:
        zp = RAW / "geography" / f"tl_2023_{st}_bg.zip"
        with zipfile.ZipFile(zp) as zf:
            shp = next(n for n in zf.namelist() if n.endswith(".shp"))
        frames.append(gpd.read_file(f"zip://{zp}!{shp}", columns=["GEOID", "ALAND"]))
    g = pd.concat(frames, ignore_index=True)
    return g[g["ALAND"] > 0].drop(columns="ALAND").to_crs(5070)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True, help="2-digit state FIPS codes")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--scale", choices=["regional", "national"], default="regional")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = pd.read_parquet(PROCESSED / "features_standardized.parquet")
    meta = pd.read_csv(PROCESSED / "feature_metadata.csv")
    raw = pd.read_parquet(PROCESSED / "features_raw.parquet")

    z = z[z.index.str[:2].isin(args.states)]
    X = regional_matrix(z, meta, args.scale)
    print(f"{len(X):,} block groups x {X.shape[1]} features")

    km = KMeans(n_clusters=args.k, n_init=25, random_state=args.seed).fit(X.values)
    # renumber clusters 1..k by descending size
    order = pd.Series(km.labels_).value_counts().index
    relabel = {old: new for new, old in enumerate(order, start=1)}
    labels = pd.Series([relabel[l] for l in km.labels_], index=X.index, name="cluster")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(20000, len(X)), replace=False)
    sil = silhouette_score(X.values[idx], labels.values[idx])
    print(f"silhouette (20k sample): {sil:.3f}")

    tag = f"cluster_{'-'.join(args.states)}_k{args.k}" + ("" if args.scale == "regional" else "_national")
    out = ROOT / "outputs" / tag
    out.mkdir(parents=True, exist_ok=True)
    labels.to_frame().assign(state=labels.index.str[:2]).to_csv(out / "assignments.csv")

    r = raw.loc[X.index, PROFILE_COLS].join(labels)
    prof = r.groupby("cluster").median().drop(columns="population")
    prof.insert(0, "n_block_groups", labels.value_counts().sort_index())
    prof.insert(1, "population", r.groupby("cluster")["population"].sum())
    prof.insert(2, "pct_of_population", (prof["population"] / prof["population"].sum()).round(3))
    for st in args.states:
        prof[f"share_bgs_in_{st}"] = labels[labels.index.str[:2] == st].value_counts().reindex(prof.index, fill_value=0) / prof["n_block_groups"]
    prof.T.to_csv(out / "profile.csv")

    zc = z.join(labels).groupby("cluster").mean() if args.scale == "national" else \
        (X / meta.set_index("column")["group_weight"].reindex(X.columns)).join(labels).groupby("cluster").mean()
    zc.T.round(2).to_csv(out / "centroids_z.csv")

    if args.k > 7:
        print(f"note: k={args.k} > 7, map colors beyond the 7th are not colorblind-safe")
    colors = {c: PALETTE[(c - 1) % len(PALETTE)] for c in range(1, args.k + 1)}
    g = load_bg_shapes(args.states).merge(labels, left_on="GEOID", right_index=True, how="left")
    g["color"] = g["cluster"].map(colors).fillna(EXCLUDED_COLOR)
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1, 1]})
    g.plot(ax=ax, color=g["color"], edgecolor=EDGE_COLOR, linewidth=0.05)
    handles = [Patch(facecolor=col, label=str(c)) for c, col in colors.items()]
    ax.legend(handles=handles, title="cluster", loc="lower right")
    ax.set_title(f"Block-group clusters (k={args.k}) — states {', '.join(args.states)}")
    # zoom panel: around the densest cluster (5th-95th pct of its centroids, padded 15 km)
    dense = prof["pop_density_per_sqkm"].idxmax()
    c = g[g["cluster"] == dense].geometry.centroid
    pad = 15_000
    x0, x1 = c.x.quantile([.05, .95]); y0, y1 = c.y.quantile([.05, .95])
    gz = g.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad]
    gz.plot(ax=axz, color=gz["color"], edgecolor=EDGE_COLOR, linewidth=0.25)
    axz.set_xlim(x0 - pad, x1 + pad); axz.set_ylim(y0 - pad, y1 + pad)
    axz.set_title(f"Zoom: densest cluster ({dense}) area")
    for a in (ax, axz):
        a.set_axis_off()
    fig.savefig(out / "map.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(prof.T.round(3))


if __name__ == "__main__":
    main()
