"""K-means clustering of block groups for a chosen set of states.

    python cluster.py --states 36 34 09 --k 5
    python cluster.py --states 06 --k 15 --zoom bay_area los_angeles --out examples/california_k15

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

    python cluster.py --states conus --k 15 --zoom top10_metros --out examples/conus_k15

Map zoom panels (--zoom): one panel per named region in ZOOMS (e.g. chicago, bay_area,
los_angeles) or group in ZOOM_GROUPS (top10_metros); default = one panel around the densest
cluster. With more than 2 zooms the overview sits on top and the zooms form a grid below.

Outputs in outputs/cluster_<states>_k<k>/ (or --out):
  assignments.csv   GEOID, cluster (clusters numbered by descending size)
  profile.csv       per-cluster medians of raw features + BG/population counts
  centroids_z.csv   cluster centers in (unweighted) z units, to see what defines each cluster
  map.png           block groups colored by cluster, white BG boundaries, land BGs not
                    clustered (excluded rows, absent from ACS) in gray; water-only BGs
                    (ALAND = 0) are left blank so lakes/sounds don't read as land
  map_by_cluster.png  (k > 7 only) one small panel per cluster, that cluster in dark ink on
                    light gray: no color set keeps more than ~7 categories distinguishable
                    (see PALETTE), so these panels carry cluster identity without color
"""
from __future__ import annotations

import argparse
import zipfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import PROCESSED, RAW, ROOT, STATE_FIPS
from features.geography import _download

CB_COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"

ECEF = ["ecef_x_km", "ecef_y_km", "ecef_z_km"]
# Clusters 1-7: Okabe-Ito colorblind-safe palette (black dropped: it hides the boundaries).
# Clusters 8-15: picked greedily from Paul Tol's and Kelly's sets to maximize the minimum
# OKLab distance (x100) to all earlier colors and the gray, under normal vision and simulated
# protan/deutan/tritan. Closest pair among all 15 + gray: 8.6 normal, 5.2-5.4 color-blind
# (below the ~15 / 8 readability targets; no 15-color set reaches them), hence
# map_by_cluster.png for k > 7.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
           "#663333", "#AA3377", "#666633", "#225555", "#99DDFF", "#604E97", "#DDCC77",
           "#EE6677"]
EXCLUDED_COLOR = "#BDBDBD"  # BGs present in TIGER but not clustered
EDGE_COLOR = "white"
HIGHLIGHT_COLOR, BACKGROUND_COLOR = "#1A1A1A", "#E4E4E4"  # map_by_cluster.png
# zoom panels: name -> (label, lon_min, lat_min, lon_max, lat_max)
ZOOMS = {
    "nyc": ("New York City", -74.27, 40.49, -73.68, 40.92),
    "chicago": ("Chicago", -88.05, 41.62, -87.52, 42.08),
    "bay_area": ("San Francisco Bay Area", -122.65, 37.20, -121.75, 38.05),
    "los_angeles": ("Los Angeles", -118.75, 33.65, -117.65, 34.35),
    "houston": ("Houston", -95.85, 29.45, -94.95, 30.15),
    "dfw": ("Dallas-Fort Worth", -97.55, 32.50, -96.50, 33.20),
    "atlanta": ("Atlanta", -84.75, 33.45, -84.05, 34.10),
    "washington_dc": ("Washington, DC", -77.45, 38.65, -76.75, 39.15),
    "philadelphia": ("Philadelphia", -75.45, 39.75, -74.90, 40.20),
    "miami": ("Miami", -80.50, 25.55, -80.05, 26.35),
    "phoenix": ("Phoenix", -112.45, 33.20, -111.60, 33.75),
}
# --zoom shorthands; metros ranked by 2023 MSA population
ZOOM_GROUPS = {
    "top10_metros": ["nyc", "los_angeles", "chicago", "dfw", "houston", "atlanta",
                     "washington_dc", "philadelphia", "miami", "phoenix"],
}
# --states shorthand: 48 contiguous states + DC
CONUS = [s for s in STATE_FIPS if s not in ("02", "15")]
LARGE_MAP_BGS = 60_000  # above this many BGs the overview panel drops BG outlines (unreadable)
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


@lru_cache(maxsize=1)
def _cb_counties() -> gpd.GeoDataFrame:
    zp = _download(CB_COUNTY_URL, RAW / "geography" / "cb_2023_us_county_500k.zip")
    return gpd.read_file(f"zip://{zp}", columns=["STATEFP"]).to_crs(5070)


@lru_cache(maxsize=None)
def land_mask(states: tuple[str, ...]):
    """Shoreline-clipped land area of `states` (Census cartographic county boundaries, 1:500k)."""
    c = _cb_counties()
    return c[c["STATEFP"].isin(states)].union_all()


def load_bg_shapes(states: list[str]) -> gpd.GeoDataFrame:
    """Every TIGER 2023 land block group in `states` (incl. ones never clustered), clipped to
    the shoreline: TIGER/Line BG polygons include their water area (the Farallon Islands BG
    covers 248 km2 of ocean for 0.4 km2 of land), which drew BGs across bays and straits.
    Only BGs with water (AWATER > 0) are clipped, one state at a time, which keeps a
    national run tractable."""
    frames = []
    for st in states:
        zp = RAW / "geography" / f"tl_2023_{st}_bg.zip"
        with zipfile.ZipFile(zp) as zf:
            shp = next(n for n in zf.namelist() if n.endswith(".shp"))
        g = gpd.read_file(f"zip://{zp}!{shp}", columns=["GEOID", "ALAND", "AWATER"])
        g = g[g["ALAND"] > 0].to_crs(5070)
        wet = g[g["AWATER"] > 0]
        clipped = gpd.clip(wet, land_mask((st,)), keep_geom_type=True)
        clipped = clipped[~clipped.geometry.is_empty]
        # tiny islands the generalized 1:500k shoreline misses keep their TIGER shape
        keep = g[(g["AWATER"] == 0) | ~g["GEOID"].isin(clipped["GEOID"])]
        frames += [keep, clipped]
    return pd.concat(frames, ignore_index=True).drop(columns=["ALAND", "AWATER"])


def zoom_extent(name: str) -> tuple[str, float, float, float, float]:
    """ZOOMS entry -> (label, x0, x1, y0, y1) in EPSG:5070."""
    label, lon0, lat0, lon1, lat1 = ZOOMS[name]
    p = gpd.GeoSeries(gpd.points_from_xy([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1]),
                      crs=4326).to_crs(5070)
    return label, p.x.min(), p.x.max(), p.y.min(), p.y.max()


def densest_extent(g: gpd.GeoDataFrame, cluster: int, pad: float = 15_000):
    """Around the given cluster: 5th-95th pct of its BG centroids, padded."""
    c = g[g["cluster"] == cluster].geometry.centroid
    x0, x1 = c.x.quantile([.05, .95]); y0, y1 = c.y.quantile([.05, .95])
    return f"densest cluster ({cluster}) area", x0 - pad, x1 + pad, y0 - pad, y1 + pad


def plot_map(g: gpd.GeoDataFrame, colors: dict, title: str, zooms: list, path,
             zoom_cols: int = 5) -> None:
    """Overview + zoom panels: one row for up to 2 zooms, else the overview on top and the
    zooms in a grid of `zoom_cols` columns below it."""
    if len(zooms) <= 2:
        fig, axes = plt.subplots(1, 1 + len(zooms), figsize=(10 * (1 + len(zooms)), 10))
        ax, zaxes = axes[0], list(axes[1:])
    else:
        x0, y0, x1, y1 = g.total_bounds
        width, zrows = 5 * zoom_cols, -(-len(zooms) // zoom_cols)
        main_h = width * (y1 - y0) / (x1 - x0)
        fig = plt.figure(figsize=(width, main_h + 1.5 + 5.3 * zrows))
        gs = fig.add_gridspec(1 + zrows, zoom_cols, height_ratios=[main_h + 1.5] + [5.3] * zrows)
        ax = fig.add_subplot(gs[0, :])
        zaxes = [fig.add_subplot(gs[1 + i // zoom_cols, i % zoom_cols]) for i in range(len(zooms))]
    axes = [ax, *zaxes]
    lw = 0 if len(g) > LARGE_MAP_BGS else 0.05
    g.plot(ax=ax, color=g["color"], edgecolor=EDGE_COLOR, linewidth=lw)
    handles = [Patch(facecolor=col, label=str(c)) for c, col in colors.items()]
    ax.legend(handles=handles, title="cluster", loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=min(len(handles), 8), frameon=False)
    ax.set_title(title)
    for axz, (label, x0, x1, y0, y1) in zip(zaxes, zooms):
        gz = g.cx[x0:x1, y0:y1]
        gz.plot(ax=axz, color=gz["color"], edgecolor=EDGE_COLOR, linewidth=0.25)
        axz.set_xlim(x0, x1); axz.set_ylim(y0, y1)
        axz.set_title(f"Zoom: {label}")
    for a in axes:
        a.set_axis_off()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_cluster(g: gpd.GeoDataFrame, colors: dict, pop_share: pd.Series, states: list[str],
                    path, ncols: int = 5) -> None:
    """Small multiples: one panel per cluster, that cluster dark on the light-gray land area."""
    k = len(colors)
    nrows = -(-k // ncols)
    x0, y0, x1, y1 = g.total_bounds
    aspect = (y1 - y0) / (x1 - x0)  # panel height follows the region's shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, (4 * aspect + 0.4) * nrows))
    land = gpd.GeoSeries([land_mask(tuple(states))], crs=g.crs)
    for ax, (c, col) in zip(axes.flat, colors.items()):
        land.plot(ax=ax, color=BACKGROUND_COLOR, linewidth=0)
        g[g["cluster"] == c].plot(ax=ax, color=HIGHLIGHT_COLOR, linewidth=0)
        ax.set_title(f"Cluster {c} · {pop_share[c]:.1%} of population", loc="left", x=0.09,
                     fontsize=11)
        # swatch = this cluster's color on map.png
        ax.add_patch(Rectangle((0, 1.01), 0.07, 0.05, transform=ax.transAxes, facecolor=col,
                               edgecolor="none", clip_on=False))
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True,
                    help="2-digit state FIPS codes, or 'conus' (48 contiguous states + DC)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--scale", choices=["regional", "national"], default="regional")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", nargs="+", choices=[*ZOOMS, *ZOOM_GROUPS], default=None,
                    help="zoom panels to draw (default: around the densest cluster)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/cluster_<states>_k<k>)")
    args = ap.parse_args()
    region = "contiguous US" if args.states == ["conus"] else None
    if region:
        args.states = CONUS
    region = region or f"states {', '.join(args.states)}"
    if args.zoom:
        args.zoom = [z for name in args.zoom for z in ZOOM_GROUPS.get(name, [name])]

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

    states_tag = "conus" if args.states == CONUS else "-".join(args.states)
    tag = f"cluster_{states_tag}_k{args.k}" + ("" if args.scale == "regional" else "_national")
    out = Path(args.out) if args.out else ROOT / "outputs" / tag
    out.mkdir(parents=True, exist_ok=True)
    labels.to_frame().assign(state=labels.index.str[:2]).to_csv(out / "assignments.csv")

    r = raw.loc[X.index, PROFILE_COLS].join(labels)
    prof = r.groupby("cluster").median().drop(columns="population")
    prof.insert(0, "n_block_groups", labels.value_counts().sort_index())
    prof.insert(1, "population", r.groupby("cluster")["population"].sum())
    prof.insert(2, "pct_of_population", (prof["population"] / prof["population"].sum()).round(3))
    for st in args.states if len(args.states) <= 8 else []:  # per-state shares only for small runs
        prof[f"share_bgs_in_{st}"] = labels[labels.index.str[:2] == st].value_counts().reindex(prof.index, fill_value=0) / prof["n_block_groups"]
    prof.T.to_csv(out / "profile.csv")

    zc = z.join(labels).groupby("cluster").mean() if args.scale == "national" else \
        (X / meta.set_index("column")["group_weight"].reindex(X.columns)).join(labels).groupby("cluster").mean()
    zc.T.round(2).to_csv(out / "centroids_z.csv")

    if args.k > len(PALETTE):
        print(f"note: k={args.k} > {len(PALETTE)}, map colors repeat; use map_by_cluster.png")
    elif args.k > 7:
        print(f"note: k={args.k} > 7, map colors are not all distinguishable; see map_by_cluster.png")
    colors = {c: PALETTE[(c - 1) % len(PALETTE)] for c in range(1, args.k + 1)}
    g = load_bg_shapes(args.states).merge(labels, left_on="GEOID", right_index=True, how="left")
    g["color"] = g["cluster"].map(colors).fillna(EXCLUDED_COLOR)
    if args.zoom:
        zooms = [zoom_extent(z) for z in args.zoom]
    else:
        zooms = [densest_extent(g, prof["pop_density_per_sqkm"].idxmax())]
    plot_map(g, colors, f"Block-group clusters (k={args.k}) — {region}", zooms, out / "map.png")
    if args.k > 7:
        plot_by_cluster(g, colors, prof["pct_of_population"], args.states,
                        out / "map_by_cluster.png")
    print(f"wrote {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(prof.T.round(3))


if __name__ == "__main__":
    main()
