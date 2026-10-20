"""Final pipeline step: merge interim BG features and build the clustering-ready matrix.

Run:  python standardize.py            (reads data/interim/*.parquet, writes data/processed/*)
Use:  from standardize import transform; Z = transform(raw_df)   # same transform on new data

Inputs  (index = 12-char 2020 BG GEOID, 239,741 rows, raw):
    data/interim/{acs_demographics, geography, education, crime, occupation}.parquet
    (building_footprints.parquet is Delaware-only -> NOT merged into the national matrix.)

Outputs (data/processed/):
    features_raw.parquet            every interim column, left-joined on the ACS universe, untouched
    features_standardized.parquet   selected features, transformed + imputed + scaled, no NaN
    features_weighted.parquet       standardized x per-column group weight
    missing_mask.parquet            bool, True where the value was imputed
    feature_metadata.csv            one row per selected feature (group, transform, bounds, ...)
    scaler_params.json              everything `transform()` needs (+ imputation_medians.parquet)
    imputation_medians.parquet      tract/county/state medians (fit rows) referenced by the JSON
    excluded_block_groups.csv       GEOIDs dropped from the standardized outputs + reason

Design rules
------------
1. SELECTION.  `FEATURES` below maps every candidate column to one concept group and a
   transform; set "include": False to drop one. The concepts map to 15 groups: distance,
   race, density, language, age, income, dependents, home_type, tenure, vehicles, commute,
   education, crime, property_tax, occupation.
   * Distance uses ECEF x/y/z (km): Euclidean ECEF distance = chord length, monotone in
     great-circle distance and uniform for AK/HI/CONUS. lat/lon/x_m/y_m are NOT used.
   * Excluded as metadata / duplicates / denominators (kept in features_raw only):
     coord_is_fallback, lat, lon, x_m, y_m, aland_sqkm, awater_sqkm, population,
     seda_avg_2009_2019 (== school_quality_index), seda_avg_2009_2019_se, crime_geo_level,
     crime_source, employed_civ_16plus (occupation denominator).
   * Off by default (toggle in FEATURES): seda_avg_2022_2025 (no data for 6 whole states, so
     ~12% would be imputed from state/national medians - imputation would manufacture
     state-level structure); adult_ba_plus_share (adult attainment, not school quality).
   * hud_school_proficiency_index IS included in `education`: SEDA is one value per school
     district, so every BG of e.g. a big-city district is identical; HUD's attendance-zone
     index (r ~0.5 with SEDA) supplies the within-district variation. Its cross-state
     incomparability (state tests) is tempered by sharing the group's weight with SEDA,
     which is nationally comparable. Its ~2% gaps (AK/ME/KS/DE) are imputed.
2. ROW FILTER.  Standardized outputs drop BGs with population < POP_THRESHOLD (=50) and BGs
   with zero households (group-quarters only: prisons, dorms, bases), since every household
   feature (income, tenure, vehicles, hh size...) is undefined there and would be 100%
   imputed. Why 50 and not 1: in pop 1-49 BGs, 88-100% lack median income, ~10-17% of all
   values are NaN (vs 0.5% for pop>=500) and ratio features explode (avg_hh_size up to
   279); above ~50 these rates fall off steeply. Only ~310 populated BGs lie in 1-49, so
   the cost is negligible. features_raw keeps every row.
3. TRANSFORMS (chosen from empirical skewness on fit rows, see feature_metadata.csv):
   * log1p: pop_density (skew 8.9 -> -0.7), median_hh_income (1.3 -> -0.2; ACS top-code
     250,001 left as is).
   * winsorize (0.5 / 99.5 pct of fit rows) for ratio features with tiny-denominator
     blow-ups: avg_hh_size, mean_vehicles_per_hh, eff_property_tax_rate_tract.
   * winsorize then log1p: age_dependency_ratio (2.3 -> 0.8), median_commute_min
     (1.1 -> -0.2), property_crime_rate (64k commercial-enclave outliers), and
     violent_crime_rate with a 2.0 lower percentile (1.1% of BGs report exactly 0 violent
     crimes - mostly tiny city agencies; with log1p those zeros would sit ~6 sd below the
     median, so they are floored at the 2nd pct ~16/100k: skew -1.4 -> -0.4).
     Bounds are stored in raw units; winsorizing before or after a monotone log is
     equivalent.
   * none: bounded 0-1 shares (race, units, language, occupation, renter, children,
     zero-vehicle, LEP),
     median_age, school indices, ECEF. Shares are left unwinsorized: their tails (e.g.
     AIAN share on reservations) are real signal and are bounded.
4. IMPUTATION (after transform/winsorizing, before scaling): hierarchical geographic median
   of the transformed feature among non-missing fit BGs: tract (GEOID[:11]) -> county
   ([:5]) -> state ([:2]) -> national. Compositional blocks (race, units, language,
   occupation) are
   imputed all-or-nothing per row: if any share is missing the whole block is replaced by
   the medians (from complete-block donors) at the finest level where donors exist, then
   rescaled to sum to 1. missing_mask flags every imputed cell.
5. SCALING: z-score (population sd, ddof=0) per feature, EXCEPT pooled-scale blocks: each
   column is centered on its own mean but all columns of the block are divided by ONE
   shared scalar s = sqrt(mean of the block's column variances) (pooled sd), so the block
   still carries total variance = its column count, like that many z-scored features.
   * ecef_x/y/z_km: keeps Euclidean distances proportional to true chord distance
     (per-axis z-scoring would stretch the N-S axis relative to E-W).
   * compositional share blocks (race, units, language, occupation; POOL_SHARE_BLOCKS):
     keeps one percentage point worth the same in every category. Per-column z-scoring
     divided rare categories by tiny sds (NHPI share sd ~0.005), so 1 pp of NHPI counted
     ~70x 1 pp of NH white and a few BGs reached z ~50-67, dominating squared distances
     and giving a 0.03%-of-population group the same variance budget as the largest group.
     Within a block, variance now follows each category's actual spread.
6. GROUP WEIGHTS (features_weighted only): column weight = CONCEPT_WEIGHTS[group] /
   sqrt(n_cols_in_group), so each concept contributes equal total variance (= weight^2)
   whatever its column count. Edit CONCEPT_WEIGHTS to up/down-weight a concept.
   Note: compositional blocks are linearly dependent (shares sum to 1), so each block has
   one fewer effective dimension than columns; this is accepted (dropping a reference
   category would make the result depend on which one is dropped).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTERIM, PROCESSED  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
INTERIM_FILES = ["acs_demographics", "geography", "education", "crime", "occupation"]  # ACS first = universe
POP_THRESHOLD = 50            # BGs with population < this are excluded from standardized outputs
EXCLUDE_NO_HOUSEHOLDS = True  # also exclude BGs with 0 occupied households (group quarters only)
HOUSEHOLD_FLAG_COL = "hh_with_children_share"  # NaN <=> 0 households (B11005_001 == 0)
DEFAULT_WINSOR = (0.5, 99.5)  # percentiles, fit on included rows

ECEF_COLS = ["ecef_x_km", "ecef_y_km", "ecef_z_km"]

# transform: "none" | "log1p" | "winsor" | "winsor_log1p";  winsor: (lo_pct, hi_pct) override
FEATURES: dict[str, dict] = {
    # distance
    "ecef_x_km": {"group": "distance", "transform": "none", "include": True},
    "ecef_y_km": {"group": "distance", "transform": "none", "include": True},
    "ecef_z_km": {"group": "distance", "transform": "none", "include": True},
    # racial composition (block)
    "race_nh_white_share": {"group": "race", "transform": "none", "include": True},
    "race_nh_black_share": {"group": "race", "transform": "none", "include": True},
    "race_nh_aian_share": {"group": "race", "transform": "none", "include": True},
    "race_nh_asian_share": {"group": "race", "transform": "none", "include": True},
    "race_nh_nhpi_share": {"group": "race", "transform": "none", "include": True},
    "race_nh_other_multi_share": {"group": "race", "transform": "none", "include": True},
    "race_hispanic_share": {"group": "race", "transform": "none", "include": True},
    # population density
    "pop_density_per_sqkm": {"group": "density", "transform": "log1p", "include": True},
    # languages spoken (tract block + BG LEP)
    "lang_tract_english_only_share": {"group": "language", "transform": "none", "include": True},
    "lang_tract_spanish_share": {"group": "language", "transform": "none", "include": True},
    "lang_tract_other_ie_share": {"group": "language", "transform": "none", "include": True},
    "lang_tract_asian_pi_share": {"group": "language", "transform": "none", "include": True},
    "lang_tract_other_share": {"group": "language", "transform": "none", "include": True},
    "lep_hh_share": {"group": "language", "transform": "none", "include": True},
    # median age
    "median_age": {"group": "age", "transform": "none", "include": True},
    # median income
    "median_hh_income": {"group": "income", "transform": "log1p", "include": True},
    # dependents per household
    "hh_with_children_share": {"group": "dependents", "transform": "none", "include": True},
    "avg_hh_size": {"group": "dependents", "transform": "winsor", "include": True},
    "age_dependency_ratio": {"group": "dependents", "transform": "winsor_log1p", "include": True},
    # home type composition (block)
    "units_1_detached_share": {"group": "home_type", "transform": "none", "include": True},
    "units_1_attached_share": {"group": "home_type", "transform": "none", "include": True},
    "units_2_4_share": {"group": "home_type", "transform": "none", "include": True},
    "units_5_19_share": {"group": "home_type", "transform": "none", "include": True},
    "units_20plus_share": {"group": "home_type", "transform": "none", "include": True},
    "units_mobile_other_share": {"group": "home_type", "transform": "none", "include": True},
    # rent vs buy
    "renter_share": {"group": "tenure", "transform": "none", "include": True},
    # cars owned
    "zero_vehicle_hh_share": {"group": "vehicles", "transform": "none", "include": True},
    "mean_vehicles_per_hh": {"group": "vehicles", "transform": "winsor", "include": True},
    # commute time
    "median_commute_min": {"group": "commute", "transform": "winsor_log1p", "include": True},
    # education quality
    "school_quality_index": {"group": "education", "transform": "none", "include": True},
    "hud_school_proficiency_index": {"group": "education", "transform": "none", "include": True},
    "seda_avg_2022_2025": {"group": "education", "transform": "none", "include": False},
    "adult_ba_plus_share": {"group": "education", "transform": "none", "include": False},
    # crime rate
    "violent_crime_rate": {"group": "crime", "transform": "winsor_log1p", "include": True,
                           "winsor": (2.0, 99.5)},
    "property_crime_rate": {"group": "crime", "transform": "winsor_log1p", "include": True},
    # effective property tax rate
    "eff_property_tax_rate_tract": {"group": "property_tax", "transform": "winsor", "include": True},
    # occupation composition (block; employed civilians 16+, by where they live)
    "occ_mgmt_prof_share": {"group": "occupation", "transform": "none", "include": True},
    "occ_service_share": {"group": "occupation", "transform": "none", "include": True},
    "occ_sales_office_share": {"group": "occupation", "transform": "none", "include": True},
    "occ_construction_maint_share": {"group": "occupation", "transform": "none", "include": True},
    "occ_production_transp_share": {"group": "occupation", "transform": "none", "include": True},
}

# Compositional share blocks: imputed all-or-nothing and renormalized to sum to 1.
SHARE_BLOCKS: dict[str, list[str]] = {
    "race": [c for c in FEATURES if c.startswith("race_")],
    "units": [c for c in FEATURES if c.startswith("units_")],
    "language": [c for c in FEATURES if c.startswith("lang_tract_")],
    "occupation": [c for c in FEATURES if c.startswith("occ_")],
}
POOL_SHARE_BLOCKS = True  # one pooled sd per share block (see SCALING); False = per-column z

# Multiplier per concept group (features_weighted = z * CONCEPT_WEIGHTS[g] / sqrt(n_g)).
CONCEPT_WEIGHTS: dict[str, float] = {
    "distance": 1.0, "race": 1.0, "density": 1.0, "language": 1.0, "age": 1.0,
    "income": 1.0, "dependents": 1.0, "home_type": 1.0, "tenure": 1.0, "vehicles": 1.0,
    "commute": 1.0, "education": 1.0, "crime": 1.0, "property_tax": 1.0, "occupation": 1.0,
}

# Documentation only: never selected (kept in features_raw).
EXCLUDED_COLUMNS = ["coord_is_fallback", "lat", "lon", "x_m", "y_m", "aland_sqkm",
                    "awater_sqkm", "population", "seda_avg_2009_2019",
                    "seda_avg_2009_2019_se", "crime_geo_level", "crime_source",
                    "employed_civ_16plus"]

GEO_LEVELS = [("tract", 11), ("county", 5), ("state", 2)]
PARAMS_PATH = PROCESSED / "scaler_params.json"
MEDIANS_PATH = PROCESSED / "imputation_medians.parquet"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def selected_features(features: dict = FEATURES) -> list[str]:
    return [c for c, f in features.items() if f["include"]]


def active_blocks(cols: list[str]) -> dict[str, list[str]]:
    out = {}
    for b, members in SHARE_BLOCKS.items():
        m = [c for c in members if c in cols]
        if m:
            out[b] = m
    return out


def load_raw(interim_dir: Path = INTERIM) -> pd.DataFrame:
    """Left-join all national interim files on the ACS universe; values untouched."""
    base = pd.read_parquet(interim_dir / f"{INTERIM_FILES[0]}.parquet")
    for name in INTERIM_FILES[1:]:
        df = pd.read_parquet(interim_dir / f"{name}.parquet")
        dup = base.columns.intersection(df.columns)
        assert dup.empty, f"duplicate columns from {name}: {list(dup)}"
        base = base.join(df, how="left")
    base.index.name = "GEOID"
    assert base.index.is_unique and (base.index.str.len() == 12).all()
    return base


def row_filter(raw: pd.DataFrame, pop_threshold: int = POP_THRESHOLD,
               exclude_no_hh: bool = EXCLUDE_NO_HOUSEHOLDS) -> pd.Series:
    """Return a Series of exclusion reasons (index = excluded GEOIDs)."""
    pop = raw["population"]
    reason = pd.Series(pd.NA, index=raw.index, dtype="object")
    if exclude_no_hh and HOUSEHOLD_FLAG_COL in raw:
        reason[raw[HOUSEHOLD_FLAG_COL].isna()] = "no_households"
    reason[pop < pop_threshold] = f"population_lt_{pop_threshold}"
    reason[pop.isna() | (pop == 0)] = "population_zero"
    return reason.dropna()


def _skew(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.skew()) if len(s) > 2 else float("nan")


def apply_transforms(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Winsorize (raw-unit bounds) then log1p, per params['columns'][c]['transform']."""
    out = pd.DataFrame(index=df.index)
    for c, p in params["columns"].items():
        s = df[c].astype(float)
        if p["transform"] in ("winsor", "winsor_log1p"):
            s = s.clip(p["winsor_lo"], p["winsor_hi"])
        if p["transform"] in ("log1p", "winsor_log1p"):
            s = np.log1p(s.clip(lower=0))
        out[c] = s
    return out


def compute_medians(t: pd.DataFrame, blocks: dict[str, list[str]]) -> pd.DataFrame:
    """Hierarchical medians of transformed values. Rows keyed by geo code (len 11/5/2),
    plus a 'national' row. Block columns use only complete-block donor rows."""
    donors = t.copy()
    for cols in blocks.values():
        incomplete = donors[cols].isna().any(axis=1)
        donors.loc[incomplete, cols] = np.nan
    parts = []
    for level, n in GEO_LEVELS:
        m = donors.groupby(donors.index.str[:n]).median()
        m.insert(0, "level", level)
        parts.append(m)
    nat = donors.median().to_frame("national").T
    nat.insert(0, "level", "national")
    out = pd.concat(parts + [nat])
    out.index.name = "key"
    return out


def impute(t: pd.DataFrame, medians: pd.DataFrame, blocks: dict[str, list[str]]):
    """Hierarchical median imputation. Returns (imputed frame, boolean mask)."""
    t = t.copy()
    for cols in blocks.values():  # all-or-nothing blocks
        incomplete = t[cols].isna().any(axis=1)
        t.loc[incomplete, cols] = np.nan
    mask = t.isna()
    by_level = {lvl: g.drop(columns="level") for lvl, g in medians.groupby("level")}
    nat = by_level["national"].iloc[0]
    in_block = {c for cols in blocks.values() for c in cols}

    for c in [c for c in t.columns if c not in in_block]:
        miss = t[c].isna()
        if not miss.any():
            continue
        fill = pd.Series(np.nan, index=t.index[miss])
        for level, n in GEO_LEVELS:
            need = fill.isna()
            if not need.any():
                break
            keys = fill.index[need].str[:n]
            fill[need] = by_level[level][c].reindex(keys).to_numpy()
        fill = fill.fillna(nat[c])
        t.loc[miss, c] = fill

    for cols in blocks.values():
        miss = t[cols[0]].isna()
        if not miss.any():
            continue
        fill = pd.DataFrame(np.nan, index=t.index[miss], columns=cols)
        for level, n in GEO_LEVELS:
            need = fill.isna().any(axis=1)
            if not need.any():
                break
            vals = by_level[level][cols].reindex(fill.index[need].str[:n])
            ok = vals.notna().all(axis=1).to_numpy()
            idx = fill.index[need][ok]
            fill.loc[idx] = vals.to_numpy()[ok]
        need = fill.isna().any(axis=1)
        fill.loc[need] = nat[cols].to_numpy()
        fill = fill.div(fill.sum(axis=1), axis=0)  # medians don't sum to 1 -> renormalize
        t.loc[miss, cols] = fill.to_numpy()
    return t, mask


def scale(t: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = params["columns"]
    mean = pd.Series({c: cols[c]["scale_mean"] for c in t.columns})
    std = pd.Series({c: cols[c]["scale_std"] for c in t.columns})
    return (t - mean) / std


def weight(z: pd.DataFrame, params: dict) -> pd.DataFrame:
    w = pd.Series({c: params["columns"][c]["group_weight"] for c in z.columns})
    return z * w


# --------------------------------------------------------------------------------------
# Fit / transform
# --------------------------------------------------------------------------------------
def fit(raw_kept: pd.DataFrame, features: dict = FEATURES,
        concept_weights: dict = CONCEPT_WEIGHTS):
    """Fit all parameters on the included rows. Returns (params, medians, imputed, mask)."""
    cols = selected_features(features)
    blocks = active_blocks(cols)
    params: dict = {"columns": {}, "selected_columns": cols, "share_blocks": blocks,
                    "ecef_cols": [c for c in ECEF_COLS if c in cols],
                    "geo_levels": GEO_LEVELS, "pop_threshold": POP_THRESHOLD,
                    "exclude_no_households": EXCLUDE_NO_HOUSEHOLDS,
                    "household_flag_col": HOUSEHOLD_FLAG_COL,
                    "concept_weights": concept_weights,
                    "medians_path": MEDIANS_PATH.name}
    for c in cols:
        f = features[c]
        s = raw_kept[c].astype(float)
        p = {"group": f["group"], "transform": f["transform"],
             "winsor_lo": None, "winsor_hi": None,
             "raw_mean": float(s.mean()), "raw_std": float(s.std(ddof=0)),
             "raw_median": float(s.median()), "raw_skew": _skew(s)}
        if f["transform"] in ("winsor", "winsor_log1p"):
            lo_p, hi_p = f.get("winsor", DEFAULT_WINSOR)
            p["winsor_pct"] = [lo_p, hi_p]
            p["winsor_lo"], p["winsor_hi"] = (float(v) for v in s.quantile([lo_p / 100, hi_p / 100]))
        params["columns"][c] = p

    t = apply_transforms(raw_kept, params)
    for c in cols:
        params["columns"][c]["transformed_skew"] = _skew(t[c])
    medians = compute_medians(t, blocks)
    imputed, mask = impute(t, medians, blocks)

    ecef = params["ecef_cols"]
    pooled_blocks = {"ecef": ecef} if ecef else {}
    if POOL_SHARE_BLOCKS:
        pooled_blocks.update(blocks)
    params["pooled_scale_blocks"] = pooled_blocks
    pooled_sd = {b: float(np.sqrt(imputed[bc].var(ddof=0).mean())) for b, bc in pooled_blocks.items()}
    params["pooled_scale"] = pooled_sd
    params["ecef_shared_scale"] = pooled_sd.get("ecef")
    block_of = {c: b for b, bc in pooled_blocks.items() for c in bc}
    n_by_group = pd.Series({c: features[c]["group"] for c in cols}).value_counts()
    for c in cols:
        p = params["columns"][c]
        g = p["group"]
        p["scale_block"] = block_of.get(c)
        p["scale_mean"] = float(imputed[c].mean())
        p["scale_std"] = pooled_sd[block_of[c]] if c in block_of else float(imputed[c].std(ddof=0))
        p["pct_imputed"] = float(100 * mask[c].mean())
        p["n_cols_in_group"] = int(n_by_group[g])
        p["concept_weight"] = float(concept_weights.get(g, 1.0))
        p["group_weight"] = p["concept_weight"] / float(np.sqrt(n_by_group[g]))
    return params, medians, imputed, mask


def load_params(path: Path = PARAMS_PATH) -> tuple[dict, pd.DataFrame]:
    params = json.loads(Path(path).read_text())
    params["geo_levels"] = [tuple(x) for x in params["geo_levels"]]
    medians = pd.read_parquet(Path(path).parent / params["medians_path"])
    return params, medians


def transform(df: pd.DataFrame, params: dict | None = None, medians: pd.DataFrame | None = None,
              apply_row_filter: bool = True, weighted: bool = False,
              return_mask: bool = False):
    """Apply the fitted transform (from scaler_params.json by default) to a raw frame with the
    features_raw column layout (index = GEOID). Missing values are imputed with the FIT-data
    tract/county/state/national medians. Returns the standardized (or weighted) matrix."""
    if params is None or medians is None:
        params, medians = load_params()
    if apply_row_filter and "population" in df:
        excl = row_filter(df, params["pop_threshold"], params["exclude_no_households"])
        df = df.drop(index=excl.index)
    t = apply_transforms(df, params)
    imputed, mask = impute(t, medians, params["share_blocks"])
    z = scale(imputed, params)
    if weighted:
        z = weight(z, params)
    return (z, mask) if return_mask else z


# --------------------------------------------------------------------------------------
# Validation + main
# --------------------------------------------------------------------------------------
def validate(z: pd.DataFrame, imputed: pd.DataFrame, params: dict, r_thresh: float = 0.95):
    arr = z.to_numpy()
    assert np.isfinite(arr).all(), "NaN/inf in standardized output"
    pooled = params["pooled_scale_blocks"]
    in_pooled = {c for bc in pooled.values() for c in bc}
    other = [c for c in z.columns if c not in in_pooled]
    assert np.allclose(z.mean(), 0, atol=1e-8), "means not ~0"
    assert np.allclose(z[other].std(ddof=0), 1, atol=1e-8), "sds not ~1"
    for b, bc in pooled.items():
        tv = float(z[bc].var(ddof=0).sum())
        assert abs(tv - len(bc)) < 1e-6, f"block {b} total variance {tv}"
        print(f"pooled scale {b}: {params['pooled_scale'][b]:.4g}; per-column sd:",
              z[bc].std(ddof=0).round(3).to_dict())
    for b, cols in params["share_blocks"].items():
        s = imputed[cols].sum(axis=1)
        print(f"block {b}: share sums min={s.min():.6f} max={s.max():.6f}")
        assert np.allclose(s, 1, atol=1e-6), f"block {b} does not sum to 1"
    corr = z.corr()
    iu = np.triu_indices_from(corr, k=1)
    pairs = [(corr.index[i], corr.columns[j], corr.iat[i, j]) for i, j in zip(*iu)
             if abs(corr.iat[i, j]) > r_thresh]
    pairs.sort(key=lambda x: -abs(x[2]))
    print(f"pairs with |r| > {r_thresh}: {len(pairs)}")
    for a, b, r in pairs:
        print(f"  {a} ~ {b}: r={r:+.3f}")
    print("validation OK")
    return pairs


def main() -> None:
    raw = load_raw()
    raw.to_parquet(PROCESSED / "features_raw.parquet")
    print(f"features_raw: {raw.shape}")

    excl = row_filter(raw)
    pd.DataFrame({"reason": excl, "population": raw.loc[excl.index, "population"]}) \
        .rename_axis("GEOID").to_csv(PROCESSED / "excluded_block_groups.csv")
    kept = raw.drop(index=excl.index)
    print(f"kept {len(kept):,} / {len(raw):,}; excluded {len(excl):,}:",
          excl.value_counts().to_dict())

    params, medians, imputed, mask = fit(kept)
    z = scale(imputed, params)
    zw = weight(z, params)

    # sanity: the public transform() reproduces the fitted output
    z2 = transform(raw, params, medians)
    assert z2.index.equals(z.index) and np.allclose(z2.to_numpy(), z.to_numpy())

    z.to_parquet(PROCESSED / "features_standardized.parquet")
    zw.to_parquet(PROCESSED / "features_weighted.parquet")
    mask.to_parquet(PROCESSED / "missing_mask.parquet")
    medians.to_parquet(MEDIANS_PATH)
    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    meta = pd.DataFrame([{"column": c, **{k: p.get(k) for k in (
        "group", "transform", "winsor_lo", "winsor_hi", "pct_imputed", "raw_mean", "raw_std",
        "raw_median", "raw_skew", "transformed_skew", "scale_block", "scale_mean", "scale_std",
        "n_cols_in_group", "concept_weight", "group_weight")}}
        for c, p in params["columns"].items()])
    meta.to_csv(PROCESSED / "feature_metadata.csv", index=False)

    validate(z, imputed, params)
    pd.set_option("display.width", 200)
    print(meta[["column", "group", "transform", "pct_imputed", "raw_skew",
                "transformed_skew", "group_weight"]].round(3).to_string(index=False))
    print(f"standardized: {z.shape}")


if __name__ == "__main__":
    main()
