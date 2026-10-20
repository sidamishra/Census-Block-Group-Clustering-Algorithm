"""ACS 2019-2023 5-year demographic / housing features per 2020 Census Block Group.

All values raw and unstandardized; shares are 0-1 fractions; NaN where the ACS
suppresses the estimate or the denominator is 0. Variable codes verified against
https://api.census.gov/data/2023/acs/acs5/groups/<TABLE>.json.

Columns (BG-level unless noted)
-------------------------------
Race / ethnicity (B03002, denominator B03002_001 total pop)
  race_nh_white_share        B03002_003
  race_nh_black_share        B03002_004
  race_nh_aian_share         B03002_005
  race_nh_asian_share        B03002_006
  race_nh_nhpi_share         B03002_007
  race_nh_other_multi_share  B03002_008 + B03002_009 (NH some other race + NH two or more)
  race_hispanic_share        B03002_012
Age / income
  median_age                 B01002_001
  median_hh_income           B19013_001 (2023 $; ACS top-codes at 250,001)
Household composition
  hh_with_children_share     B11005_002 / B11005_001 (households with 1+ person under 18;
                             includes non-own/related children - "own children" not at BG)
  avg_hh_size                B25010_001 (published at BG)
  age_dependency_ratio       (under 18 + 65+) / (18-64) from B01001 (both sexes)
Units in structure (B25024, denominator B25024_001 all housing units incl. vacant)
  units_1_detached_share     _002
  units_1_attached_share     _003
  units_2_4_share            _004 + _005
  units_5_19_share           _006 + _007
  units_20plus_share         _008 + _009
  units_mobile_other_share   _010 + _011 (mobile home + boat/RV/van)
Tenure
  renter_share               B25003_003 / B25003_001 (occupied units)
Vehicles (B25044, owner+renter combined)
  zero_vehicle_hh_share      (_003 + _010) / _001
  mean_vehicles_per_hh       sum(k * count_k) / _001, "5 or more" counted as 5 (lower bound)
Commute (B08303, workers 16+ who did not work from home)
  median_commute_min         linear interpolation within bins; edges 0,5,...,45,60,90,
                             top bin "90+" treated as 90-120
  (mean commute NOT produced: aggregate travel time B08136 / B08013 is null at BG.)
Language
  lang_tract_english_only_share   C16001_002 / C16001_001  (TRACT level, broadcast to BGs)
  lang_tract_spanish_share        C16001_003
  lang_tract_other_ie_share       C16001_006+_009+_012+_015 (French/Haitian/Cajun, German/W.Germanic,
                                  Slavic, other Indo-European)
  lang_tract_asian_pi_share       C16001_018+_021+_024+_027+_030 (Korean, Chinese, Vietnamese,
                                  Tagalog, other Asian & PI)
  lang_tract_other_share          C16001_033+_036 (Arabic, other & unspecified)
  lep_hh_share               BG: (C16002_004+_007+_010+_013) / C16002_001 limited-English households
Property tax (TRACT level, broadcast to BGs)
  eff_property_tax_rate_tract  B25090_001 aggregate real estate taxes / B25082_001 aggregate
                             value (owner-occupied units). Where aggregates are suppressed,
                             fallback = B25103_001 median taxes / B25077_001 median value,
                             only if neither is top-coded (10,001 / 2,000,001). Ratios >0.10
                             set to NaN. Caveat: B25103, B25090, B25082 are all null at BG in
                             the 2023 5-yr API, hence tract level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import acs_fetch, safe_div, save_interim  # noqa: E402


def _v(table: str, nums) -> list[str]:
    return [f"{table}_{n:03d}E" for n in nums]


# --- variable lists -----------------------------------------------------------------
RACE = _v("B03002", [1, 3, 4, 5, 6, 7, 8, 9, 12])
AGE_U18 = _v("B01001", [3, 4, 5, 6, 27, 28, 29, 30])
AGE_65P = _v("B01001", [20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49])
AGE = ["B01001_001E"] + AGE_U18 + AGE_65P
MISC = ["B01002_001E", "B19013_001E", "B11005_001E", "B11005_002E", "B25010_001E",
        "B25003_001E", "B25003_003E"]
UNITS = _v("B25024", range(1, 12))
VEH = _v("B25044", [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15])
COMMUTE = _v("B08303", range(1, 14))
LEP = _v("C16002", [1, 4, 7, 10, 13])
BG_VARS = RACE + AGE + MISC + UNITS + VEH + COMMUTE + LEP

LANG = _v("C16001", [1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
TAX = ["B25090_001E", "B25082_001E", "B25103_001E", "B25077_001E"]
TRACT_VARS = LANG + TAX

COMMUTE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90, 120], dtype=float)
TAX_RATE_MAX = 0.10


def _sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    # min_count=1 so an all-NaN row stays NaN instead of becoming 0
    return df[cols].sum(axis=1, min_count=len(cols))


def median_from_bins(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Grouped-data median via linear interpolation. counts: (n, k), edges: (k+1,)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=1)
    half = total / 2.0
    cum = np.cumsum(counts, axis=1)
    # first bin whose cumulative count reaches half
    idx = (cum < half[:, None]).sum(axis=1)
    idx = np.clip(idx, 0, counts.shape[1] - 1)
    rows = np.arange(len(counts))
    below = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    in_bin = counts[rows, idx]
    lo, hi = edges[idx], edges[idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(in_bin > 0, (half - below) / in_bin, 0.0)
    med = lo + frac * (hi - lo)
    med[~(total > 0) | np.isnan(counts).any(axis=1)] = np.nan
    return med


def _tract_features() -> pd.DataFrame:
    t = acs_fetch(TRACT_VARS, geo="tract", cache_name="acs_demographics_tract")
    out = pd.DataFrame(index=t.index)
    pop5 = t["C16001_001E"]
    out["lang_tract_english_only_share"] = safe_div(t["C16001_002E"], pop5)
    out["lang_tract_spanish_share"] = safe_div(t["C16001_003E"], pop5)
    out["lang_tract_other_ie_share"] = safe_div(_sum(t, _v("C16001", [6, 9, 12, 15])), pop5)
    out["lang_tract_asian_pi_share"] = safe_div(_sum(t, _v("C16001", [18, 21, 24, 27, 30])), pop5)
    out["lang_tract_other_share"] = safe_div(_sum(t, _v("C16001", [33, 36])), pop5)

    agg = safe_div(t["B25090_001E"], t["B25082_001E"])
    med_ok = (t["B25103_001E"] < 10001) & (t["B25077_001E"] < 2000001)
    med = safe_div(t["B25103_001E"], t["B25077_001E"]).where(med_ok)
    rate = agg.fillna(med)
    n_fallback = int((agg.isna() & med.notna()).sum())
    bad = rate > TAX_RATE_MAX
    print(f"[tax] tracts: {len(rate)}, aggregate-ratio: {int(agg.notna().sum())}, "
          f"median-ratio fallback: {n_fallback}, >{TAX_RATE_MAX:.0%} clipped to NaN: {int(bad.sum())}")
    out["eff_property_tax_rate_tract"] = rate.mask(bad)
    return out


def build() -> pd.DataFrame:
    d = acs_fetch(BG_VARS, geo="block group", cache_name="acs_demographics_bg")
    out = pd.DataFrame(index=d.index)
    out.index.name = "GEOID"

    # 1. race / ethnicity
    pop = d["B03002_001E"]
    for name, col in [("nh_white", "B03002_003E"), ("nh_black", "B03002_004E"),
                      ("nh_aian", "B03002_005E"), ("nh_asian", "B03002_006E"),
                      ("nh_nhpi", "B03002_007E")]:
        out[f"race_{name}_share"] = safe_div(d[col], pop)
    out["race_nh_other_multi_share"] = safe_div(_sum(d, ["B03002_008E", "B03002_009E"]), pop)
    out["race_hispanic_share"] = safe_div(d["B03002_012E"], pop)

    # 2-3. age, income
    out["median_age"] = d["B01002_001E"]
    out["median_hh_income"] = d["B19013_001E"]

    # 4. dependents
    out["hh_with_children_share"] = safe_div(d["B11005_002E"], d["B11005_001E"])
    out["avg_hh_size"] = d["B25010_001E"]
    u18, o65 = _sum(d, AGE_U18), _sum(d, AGE_65P)
    working = d["B01001_001E"] - u18 - o65
    out["age_dependency_ratio"] = safe_div(u18 + o65, working)

    # 5. units in structure
    hu = d["B25024_001E"]
    for name, nums in [("1_detached", [2]), ("1_attached", [3]), ("2_4", [4, 5]),
                       ("5_19", [6, 7]), ("20plus", [8, 9]), ("mobile_other", [10, 11])]:
        out[f"units_{name}_share"] = safe_div(_sum(d, _v("B25024", nums)), hu)

    # 6. tenure
    out["renter_share"] = safe_div(d["B25003_003E"], d["B25003_001E"])

    # 7. vehicles
    hh = d["B25044_001E"]
    out["zero_vehicle_hh_share"] = safe_div(_sum(d, ["B25044_003E", "B25044_010E"]), hh)
    veh = sum(k * (d[f"B25044_{3 + k:03d}E"] + d[f"B25044_{10 + k:03d}E"]) for k in range(1, 6))
    out["mean_vehicles_per_hh"] = safe_div(veh, hh)

    # 8. commute
    bins = d[_v("B08303", range(2, 14))].to_numpy()
    out["median_commute_min"] = median_from_bins(bins, COMMUTE_EDGES)

    # 10b. limited-English households (BG)
    out["lep_hh_share"] = safe_div(_sum(d, _v("C16002", [4, 7, 10, 13])), d["C16002_001E"])

    # 9 + 10a. tract-level features broadcast to BGs
    tr = _tract_features()
    tr_join = tr.reindex(out.index.str[:11])
    tr_join.index = out.index
    out = out.join(tr_join)

    out = out.astype(float).replace([np.inf, -np.inf], np.nan)
    return out


if __name__ == "__main__":
    df = build()
    save_interim(df, "acs_demographics")
