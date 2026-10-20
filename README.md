# Census Block Group Clustering Algorithm

Groups 2020 Census Block Groups (BGs) into neighborhood types with k-means. A national
feature matrix (50 states + DC, index = 12-char `GEOID`) describes neighborhood character
(location, demographics, housing, schools, crime, taxes, occupations); it is standardized,
weighted so every concept counts equally, and clustered for any set of states.
`kmeans_formulation.tex` states the exact objective and distance.

![NY-NJ-CT example, k=5](outputs/cluster_36-34-09_k5/map.png)

## Layout

```
common.py                 shared config/helpers (paths, ACS fetch, GEOID contract)
features/<name>.py        one module per source; build() -> raw DataFrame, __main__ writes interim
standardize.py            merge + select + transform + impute + scale (final step; importable transform())
run_pipeline.py           runs every feature module (if needed) then standardize
cluster.py                k-means for chosen states -> outputs/cluster_<states>_k<k>/
kmeans_formulation.tex    mathematical statement of the full method
outputs/cluster_36-34-09_k5/   New York metro example (below)
data/raw/<name>/          cached downloads (reruns are offline)
data/interim/<name>.parquet   raw per-module features (239,741 BGs)
data/processed/           outputs below
```

`data/processed/`: `features_raw.parquet` (all interim columns, all rows, untouched),
`features_standardized.parquet` (clustering matrix), `features_weighted.parquet` (x group
weights), `missing_mask.parquet` (True = imputed), `feature_metadata.csv`,
`scaler_params.json` + `imputation_medians.parquet` (to re-apply the transform),
`excluded_block_groups.csv`. `data/` is not in the repository (~1.6 GB); the pipeline
downloads every source from its public URL. SEDA's terms forbid redistributing its raw files.

## Run

```
pip install -r requirements.txt
```

```
python run_pipeline.py                  # reuse existing interim files, then standardize (~30 s)
python run_pipeline.py --force          # rebuild everything (downloads; slow)
python run_pipeline.py --force crime    # rebuild one module
```
Set `CENSUS_API_KEY` for ACS rebuilds. To apply the fitted transform to new/edited data:
`from standardize import transform; Z = transform(raw_df)` (same columns as `features_raw`).

Clustering (details in the `cluster.py` docstring):
```
python cluster.py --states 36 34 09 --k 5        # state FIPS codes; ~15 s
```
Writes `assignments.csv`, `profile.csv` (per-cluster medians), `centroids_z.csv` and `map.png`
(colorblind-safe palette; BGs not clustered in gray) to `outputs/cluster_<states>_k<k>/`.
By default features are re-standardized within the selected states (`--scale regional`).

## Example: New York metro (NY, NJ, CT), k = 5

`outputs/cluster_36-34-09_k5/` - 24,680 block groups, silhouette 0.151. Medians per cluster:

| Cluster | Share of pop. | Density (/km²) | Median HH income | Renters | Single-family detached | Median commute (min) | Character |
|---|---|---|---|---|---|---|---|
| 1 | 36% | 1,055 | $138,600 | 10% | 89% | 28 | Affluent suburbs |
| 2 | 19% | 3,446 | $62,900 | 63% | 26% | 21 | Small cities and older inner suburbs |
| 3 | 16% | 183 | $76,300 | 18% | 76% | 21 | Upstate towns and rural areas |
| 4 | 16% | 14,714 | $107,200 | 54% | 4% | 38 | Middle- and upper-income NYC neighborhoods |
| 5 | 13% | 24,618 | $54,400 | 89% | 0% | 43 | Dense, lower-income, mostly renter NYC neighborhoods |

The right panel of the map zooms on New York City.

## Features (default selection)

| Concept | Columns | Source | Native geography | Caveat |
|---|---|---|---|---|
| Physical distance | `ecef_x_km`, `ecef_y_km`, `ecef_z_km` | CenPop2020 BG pop-weighted centers (TIGER internal point fallback) | BG | Fallback point for ~0.6% of BGs |
| Racial composition | 7 `race_*_share` (NH white/black/AIAN/Asian/NHPI/other+multi, Hispanic) | ACS 2019-23 B03002 | BG | Small groups (AIAN, NHPI) are rare; pooled block scale keeps their z bounded (see Scaling) |
| Population density | `pop_density_per_sqkm` | ACS B01003 / TIGER ALAND | BG | |
| Languages spoken | 5 `lang_tract_*_share`, `lep_hh_share` | ACS C16001 (tract), C16002 (BG) | tract / BG | Language mix is tract-level, broadcast to BGs |
| Median age | `median_age` | ACS B01002 | BG | |
| Median income | `median_hh_income` | ACS B19013 | BG | Top-coded at $250,001; ~6% missing (imputed) |
| Dependents / household | `hh_with_children_share`, `avg_hh_size`, `age_dependency_ratio` | ACS B11005, B25010, B01001 | BG | Ratios blow up in tiny BGs (winsorized) |
| Home type | 6 `units_*_share` | ACS B25024 | BG | Denominator includes vacant units |
| Rent vs buy | `renter_share` | ACS B25003 | BG | |
| Cars owned | `zero_vehicle_hh_share`, `mean_vehicles_per_hh` | ACS B25044 | BG | "5+" counted as 5 |
| Commute time | `median_commute_min` | ACS B08303 (binned, interpolated) | BG | Excludes work-from-home |
| Education quality | `school_quality_index` (SEDA), `hud_school_proficiency_index` | SEDA 6.0 / 2025.2; HUD AFFH (2015-16) | school district / attendance zone | See limitations |
| Crime rate | `violent_crime_rate`, `property_crime_rate` | FBI CDE 2025 (+ CHR 2022 fallback) | city / county remainder | See limitations |
| Property tax | `eff_property_tax_rate_tract` | ACS B25090/B25082 | tract | Not published at BG |
| Occupation composition | 5 `occ_*_share` (mgmt/business/science/arts, service, sales/office, construction/maintenance, production/transportation) | ACS C24010 | BG | Residents' occupations, not local jobs; noisy in small BGs |

In `features_raw` but off by default (toggle in `standardize.FEATURES`): `seda_avg_2022_2025`
(no data for AK, AR, CO, ME, NY, OR), `adult_ba_plus_share` (attainment, not school quality).
Never selected: metadata/duplicates/denominators (`lat`, `lon`, `x_m`, `y_m`,
`coord_is_fallback`, `aland_sqkm`, `awater_sqkm`, `population`, `seda_avg_2009_2019`(+`_se`),
`crime_geo_level`, `crime_source`, `employed_civ_16plus`).

**Optional enrichment:** `data/interim/building_footprints.parquet` (Microsoft footprints:
building count/density/coverage/mean size) exists for Delaware only and is not merged. A
national run (`python features/building_footprints.py all`) takes ~2.5 h and downloads ~4.8 GB.

## Standardization (details in the `standardize.py` docstring)

1. **Rows:** drop BGs with population < 50 or zero households (2,912 of 239,741; listed in
   `excluded_block_groups.csv`). Below 50 people, 88-100% lack income and ratios explode.
2. **Transforms:** `log1p` for density and income; winsorize at the 0.5/99.5 percentiles for
   `avg_hh_size`, `mean_vehicles_per_hh`, property tax; winsorize + `log1p` for age dependency,
   commute and crime (violent crime floored at the 2nd percentile because zero-crime
   small agencies would otherwise become extreme outliers). Shares and ECEF are untouched.
3. **Imputation:** median of the transformed value in the same tract, else county, else
   state, else nation. Race, units, language and occupation shares are imputed together and
   rescaled to sum to 1.
4. **Scaling:** z-score per feature, except pooled-scale blocks whose columns share one sd
   (`scale_block` in `feature_metadata.csv`): ECEF x/y/z (862 km) so distances stay
   geometrically true, and each compositional share block (race, home type, language mix,
   occupation; `POOL_SHARE_BLOCKS`) so one percentage point counts the same in every
   category. Per-column z-scoring had divided rare shares by tiny sds (NHPI: 1 pp = 2.1 sd,
   max z ~67 nationally), letting a few BGs dominate distances. `cluster.py`'s regional
   re-scaling pools the same blocks.
5. **Weights** (`features_weighted`): column weight = `CONCEPT_WEIGHTS[group] / sqrt(n_cols)`,
   so every concept contributes equal variance; edit `CONCEPT_WEIGHTS` to emphasize one.

## Known limitations

- **Coarse geography:** property tax and language mix are tract-level; crime is
  jurisdiction-level (every BG in a city or county remainder shares a value); SEDA is
  school-district-level (all BGs of a large district are identical). Clusters can partly
  reflect those boundaries.
- **Crime:** incomplete agency reporting (FL/ND limited, NIBRS transition), state-police
  offenses omitted, commercial enclaves have extreme property rates (winsorized).
- **SEDA terms:** non-commercial use only, cite SEDA, do not redistribute raw files.
  Scores reflect achievement (SES-confounded), not school value-added.
- **HUD school index:** 2015-16 data, state-relative proficiency (not comparable across states),
  missing for AK/ME/KS/DE (imputed).
- **Universe:** 40 TIGER BGs (Suffolk Co., NY + one Ulster Co. BG) are absent from ACS 2023
  and from every output.
- **Footprints:** only run for Delaware (see above).
