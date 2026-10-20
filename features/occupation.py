"""ACS 2019-2023 5-year occupation composition per 2020 Census Block Group.

Shares of the civilian employed population 16+ (C24010_001) in each of the five ACS
major occupation groups, male + female combined. The five shares sum to 1 (a
compositional block, like race / units). NaN where the BG has no employed residents
or the ACS suppresses the estimate. Variable codes verified against
https://api.census.gov/data/2023/acs/acs5/groups/C24010.json (female = male + 36).

Columns
-------
  occ_mgmt_prof_share         C24010_003 + _039  Management, business, science, and arts
  occ_service_share           C24010_019 + _055  Service
  occ_sales_office_share      C24010_027 + _063  Sales and office
  occ_construction_maint_share C24010_030 + _066 Natural resources, construction, maintenance
  occ_production_transp_share C24010_034 + _070  Production, transportation, material moving
  employed_civ_16plus         C24010_001 (denominator; metadata, not a clustering feature)

Caveat: measures where residents work (by occupation), not the jobs located in the BG.
BG estimates rest on few respondents, so small-BG shares are noisy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import acs_fetch, safe_div, save_interim  # noqa: E402

FEMALE_OFFSET = 36
GROUPS = {  # name -> male major-group line
    "mgmt_prof": 3,
    "service": 19,
    "sales_office": 27,
    "construction_maint": 30,
    "production_transp": 34,
}
VARS = ["C24010_001E"] + [f"C24010_{n + k:03d}E" for n in GROUPS.values()
                          for k in (0, FEMALE_OFFSET)]


def build() -> pd.DataFrame:
    d = acs_fetch(VARS, geo="block group", cache_name="occupation_c24010_bg")
    out = pd.DataFrame(index=d.index)
    out.index.name = "GEOID"
    emp = d["C24010_001E"]
    for name, n in GROUPS.items():
        cnt = d[f"C24010_{n:03d}E"] + d[f"C24010_{n + FEMALE_OFFSET:03d}E"]
        out[f"occ_{name}_share"] = safe_div(cnt, emp)
    out["employed_civ_16plus"] = emp
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    df = build()
    shares = df.filter(like="occ_")
    s = shares.sum(axis=1)[shares.notna().all(axis=1)]
    print(f"BGs: {len(df):,}; with shares: {len(s):,}; share sums min={s.min():.6f} max={s.max():.6f}")
    print(shares.describe().T.round(3))
    save_interim(df, "occupation")
