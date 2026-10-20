"""Education-quality features for 2020-vintage Census Block Groups (50 states + DC).

Columns (all raw, unstandardized; NaN where unavailable)
-------------------------------------------------------
school_quality_index
    PRIMARY feature. Mean test score of the geographic school district containing the
    BG's population-weighted centroid, SEDA cohort-standardized ("cs") scale, pooled over
    grades 3-8, math + reading, school years 2008-09..2018-19, empirical-Bayes shrunk
    (`cs_mn_avg_eb`). Units = grade levels relative to the national average student
    (0 = national average; +1 = one grade level ahead). HIGHER = BETTER.
    Source: SEDA 6.0 geographic-district file; for districts absent from it (DC, HI and a
    handful of others) the identical-period SEDA 2025.2 administrative-district estimate
    is used (same scale/period). Equals `seda_avg_2009_2019`.
seda_avg_2009_2019
    Same value as school_quality_index (kept as the explicit source column).
seda_avg_2009_2019_se
    Standard error of the EB estimate above (grade levels) - useful as a reliability weight.
seda_avg_2022_2025
    Post-pandemic counterpart: SEDA 2025.2 administrative-district `cs_mn_avg_eb`,
    pooled 2021-22..2024-25 (grades 3-8, math+RLA). Same CS scale (both versions anchor
    to the national cohorts in 4th grade in 2009/11/13/15), so the national median is
    about -0.19, which reflects post-pandemic score declines. HIGHER = BETTER. Matched to the
    geographic district by NCES LEAID (sedaadmin == sedalea). About 88% of BGs are covered. SEDA has
    no post-2022 data for AK, AR, CO, ME, NY or OR, and ND reports few districts.
hud_school_proficiency_index
    HUD AFFH School Proficiency Index, version AFFHT0006 (GreatSchools 2015-16 4th-grade
    reading/math proficiency, CCD 2015-16 enrollment, Maponics 2016 attendance zones).
    Enrollment-weighted proficiency of up to 3 nearby/zoned elementary schools,
    national percentile rank 0-100. HIGHER = BETTER. Published on 2010 block groups;
    mapped to 2020 BGs by point-in-polygon of the 2020 BG population centroid in the
    2010 BG polygon. Captures within-district (attendance-zone) variation, but state
    proficiency rates are NOT comparable across states (different tests/cut scores),
    so cross-state comparisons with this column are unreliable. About 98% of BGs are covered, but there
    are almost no values for AK, ME, KS or DE.
adult_ba_plus_share
    ACS 2019-2023 5-yr (B15003): share of population 25+ with a bachelor's degree or
    higher. This is ADULT EDUCATIONAL ATTAINMENT, NOT school quality.

Sources (downloaded and cached under data/raw/education/)
--------------------------------------------------------
* SEDA 6.0 (Reardon et al., 2026), Stanford Digital Repository purl.stanford.edu/xh833nn4025:
  seda_geodist_pool_cs_6.0.csv, SEDA-2019-District-Shapefile.zip (SEDA's own geographic
  district polygons, TIGER/EDGE 2018-19 unified + elementary districts).
* SEDA 2025.2 (purl.stanford.edu/np279jm6134): seda_admindist_pool_cs_2025.2.csv.
  SEDA Data Use Agreement: non-commercial use, cite SEDA, do not redistribute raw files.
* HUD eGIS open data "School Proficiency Index" (item 11a058178b9d471292ae2571e84d9ca8,
  layer modified 2023-07, VERSION=AFFHT0006), shapefile via ArcGIS Hub download API.
  The newer AFFHT0007 (Aug 2024) tables were not obtainable (HUD Exchange page 404s,
  huduser docs return AccessDenied).
* Census CenPop2020 block-group population-weighted centroids (CenPop2020_Mean_BG.txt).
* TIGER/Line 2023 BG polygons for Connecticut only (to re-key CT centroids to the
  planning-region county codes used by ACS 2022+; BG shapes are unchanged).
* ACS 2019-2023 5-yr B15003 via common.acs_fetch (also defines the BG universe).

Method
------
1. BG universe = all ACS 2023 BGs in 50 states + DC (2020 vintage GEOIDs, CT planning
   regions).
2. Each BG is represented by its 2020 population-weighted centroid (CenPop2020). BGs with
   zero population / no centroid fall back to the TIGER internal point (CT) or NaN.
3. Centroids are spatially joined (within) to SEDA geographic district polygons
   (EPSG:5070); centroids falling in no polygon (coastal/water edge) get the nearest
   district within 5 km.
4. District scores attached by LEAID; HUD index by point-in-2010-BG polygon (nearest
   within 1 km as fallback).

Caveats
-------
* District averages hide within-district (school/attendance zone) variation - large
  urban districts get one value for every BG.
* SEDA cs scores reflect student achievement, which is strongly confounded with family
  SES; they are not a value-added measure. (SEDA's learning-rate `cs_mn_lrn_eb` is closer
  to school effectiveness but is not included.)
* In states with separate elementary/secondary districts, the geographic district is the
  elementary (grades 3-8 relevant) district.
* HUD index: stale (2015-16), 2010 geography, state-relative proficiency, missing for
  states without GreatSchools data.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, STATE_FIPS, acs_fetch, safe_div, save_interim  # noqa: E402

DIR = RAW / "education"
DIR.mkdir(parents=True, exist_ok=True)
CRS = "EPSG:5070"

STACKS = "https://stacks.stanford.edu/file"
URLS = {
    "seda_geodist_pool_cs_6.0.csv": f"{STACKS}/xh833nn4025/seda_geodist_pool_cs_6.0.csv",
    "SEDA-2019-District-Shapefile.zip": f"{STACKS}/xh833nn4025/SEDA-2019-District-Shapefile.zip",
    "seda_admindist_pool_cs_2025.2.csv": f"{STACKS}/np279jm6134/seda_admindist_pool_cs_2025.2.csv",
    "CenPop2020_Mean_BG.txt": "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt",
    "tl_2023_09_bg.zip": "https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_09_bg.zip",
    "hud/School_Proficiency_Index.zip": (
        "https://hub.arcgis.com/api/v3/datasets/11a058178b9d471292ae2571e84d9ca8_22/"
        "downloads/data?format=shp&spatialRefId=4326&where=1%3D1"),
}


def _fetch(name: str) -> Path:
    path = DIR / name
    if path.exists() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    with requests.get(URLS[name], stream=True, timeout=600,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    return path


def _zip_shp(path: Path) -> str:
    shp = next(n for n in zipfile.ZipFile(path).namelist() if n.endswith(".shp"))
    return f"zip://{path.as_posix()}!{shp}"


# ----------------------------------------------------------------------------- inputs
def _adult_attainment() -> pd.DataFrame:
    v = ["B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
    acs = acs_fetch(v, geo="block group", cache_name="education_b15003_bg")
    out = pd.DataFrame(index=acs.index)
    out["adult_ba_plus_share"] = safe_div(acs[v[1:]].sum(axis=1, min_count=1), acs[v[0]])
    out.index.name = "GEOID"
    return out


def _bg_points(universe: pd.Index) -> gpd.GeoDataFrame:
    """Population-weighted 2020 BG centroids keyed to ACS-2023 GEOIDs."""
    cp = pd.read_csv(_fetch("CenPop2020_Mean_BG.txt"), dtype=str, encoding="utf-8-sig")
    cp["GEOID"] = cp.STATEFP + cp.COUNTYFP + cp.TRACTCE + cp.BLKGRPCE
    cp = cp[cp.STATEFP.isin(STATE_FIPS)]
    pts = gpd.GeoDataFrame(
        {"GEOID": cp.GEOID.values},
        geometry=gpd.points_from_xy(cp.LONGITUDE.astype(float), cp.LATITUDE.astype(float)),
        crs="EPSG:4269").to_crs(CRS)

    # Connecticut: ACS 2022+ uses planning-region county codes -> re-key spatially.
    ct_bg = gpd.read_file(_zip_shp(_fetch("tl_2023_09_bg.zip")),
                          columns=["GEOID", "INTPTLAT", "INTPTLON"]).to_crs(CRS)
    ct_old = pts[pts.GEOID.str[:2] == "09"]
    ct_new = gpd.sjoin(ct_old, ct_bg[["GEOID", "geometry"]], predicate="within",
                       rsuffix="new").drop_duplicates("GEOID_left")
    ct_pts = ct_new[["GEOID_new", "geometry"]].rename(columns={"GEOID_new": "GEOID"})
    # any CT BG not hit by a centroid -> TIGER internal point
    miss = ct_bg[~ct_bg.GEOID.isin(ct_pts.GEOID)]
    if len(miss):
        ip = gpd.GeoDataFrame({"GEOID": miss.GEOID.values}, geometry=gpd.points_from_xy(
            miss.INTPTLON.astype(float), miss.INTPTLAT.astype(float)), crs="EPSG:4269").to_crs(CRS)
        ct_pts = pd.concat([ct_pts, ip])
    pts = pd.concat([pts[pts.GEOID.str[:2] != "09"], ct_pts], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=CRS).drop_duplicates("GEOID")
    n_missing = (~universe.isin(pts.GEOID)).sum()
    print(f"BG centroids: {len(pts):,}; universe BGs without centroid: {n_missing}")
    return pts[pts.GEOID.isin(universe)].reset_index(drop=True)


def _join_polys(pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, key: str,
                max_dist: float) -> pd.Series:
    """GEOID -> polygon key; within, then nearest (<= max_dist m) for unmatched."""
    j = gpd.sjoin(pts, polys[[key, "geometry"]], predicate="within", how="left")
    j = j.drop_duplicates("GEOID")
    s = j.set_index("GEOID")[key]
    miss = s[s.isna()].index
    if len(miss):
        mp = pts[pts.GEOID.isin(miss)]
        nn = gpd.sjoin_nearest(mp, polys[[key, "geometry"]], how="left",
                               max_distance=max_dist).drop_duplicates("GEOID")
        s.loc[nn.GEOID.values] = nn[key].values
        print(f"  {key}: {len(miss):,} centroids outside polygons; "
              f"{nn[key].notna().sum():,} filled by nearest <= {max_dist / 1000:g} km")
    return s


def _seda(pts: gpd.GeoDataFrame) -> pd.DataFrame:
    dist = gpd.read_file(_zip_shp(_fetch("SEDA-2019-District-Shapefile.zip")),
                         columns=["GEOID"]).to_crs(CRS)
    dist["leaid"] = pd.to_numeric(dist.GEOID, errors="coerce")
    lea = _join_polys(pts, dist, "leaid", 5000)

    use = ["cs_mn_avg_eb", "cs_mn_avg_eb_se"]
    g = pd.read_csv(_fetch("seda_geodist_pool_cs_6.0.csv"), low_memory=False,
                    usecols=["sedalea", "subgroup"] + use)
    g = g[g.subgroup == "all"].set_index("sedalea")[use]
    a = pd.read_csv(_fetch("seda_admindist_pool_cs_2025.2.csv"), low_memory=False,
                    usecols=["sedaadmin", "subgroup", "pooled_years"] + use)
    a = a[a.subgroup == "all"]
    a_old = a[a.pooled_years == "2009-2019"].set_index("sedaadmin")[use]
    a_new = a[a.pooled_years == "2022-2025"].set_index("sedaadmin")[use]

    out = pd.DataFrame(index=lea.index)
    out["seda_leaid"] = lea
    old = g.reindex(lea.values)
    fill = a_old.reindex(lea.values)
    src_geo = old["cs_mn_avg_eb"].notna().values
    old = old.fillna(fill)  # same period/scale; covers DC, HI, a few others
    out["seda_avg_2009_2019"] = old["cs_mn_avg_eb"].values
    out["seda_avg_2009_2019_se"] = old["cs_mn_avg_eb_se"].values
    out["seda_avg_2022_2025"] = a_new["cs_mn_avg_eb"].reindex(lea.values).values
    print(f"  SEDA 2009-19: {src_geo.mean():.1%} BGs from geodist file, "
          f"{(out.seda_avg_2009_2019.notna().values & ~src_geo).mean():.2%} filled from admindist")
    return out


def _hud(pts: gpd.GeoDataFrame) -> pd.Series:
    h = gpd.read_file(_zip_shp(_fetch("hud/School_Proficiency_Index.zip")),
                      columns=["GEOID", "SCHL_IDX", "VERSION"])
    print(f"  HUD versions: {h.VERSION.value_counts().to_dict()}")
    h = h.rename(columns={"GEOID": "GEOID10"}).to_crs(CRS)
    g10 = _join_polys(pts, h, "GEOID10", 1000)
    val = h.drop_duplicates("GEOID10").set_index("GEOID10")["SCHL_IDX"]
    val = pd.to_numeric(val, errors="coerce")
    val = val.where(val >= 0)  # guard against negative sentinels
    s = pd.Series(val.reindex(g10.values).values, index=g10.index,
                  name="hud_school_proficiency_index")
    same = (g10.index == g10.values).mean()
    print(f"  HUD: {same:.1%} of 2020 BGs sit in a 2010 BG with the identical GEOID")
    return s


# ----------------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    att = _adult_attainment()
    universe = att.index
    pts = _bg_points(universe)
    seda = _seda(pts)
    hud = _hud(pts)

    df = pd.DataFrame(index=universe)
    df.index.name = "GEOID"
    df["school_quality_index"] = seda["seda_avg_2009_2019"].reindex(universe)
    for c in ["seda_avg_2009_2019", "seda_avg_2009_2019_se", "seda_avg_2022_2025"]:
        df[c] = seda[c].reindex(universe)
    df["hud_school_proficiency_index"] = hud.reindex(universe)
    df["adult_ba_plus_share"] = att["adult_ba_plus_share"]
    return df.astype("float64")


if __name__ == "__main__":
    out = build()
    save_interim(out, "education")
    print(out.describe().T)
    print("non-null share:\n", out.notna().mean().round(4))
    st = out.groupby(out.index.str[:2])["school_quality_index"].apply(lambda s: s.notna().mean())
    print("worst-5 states (school_quality_index coverage):\n", st.sort_values().head(5))
