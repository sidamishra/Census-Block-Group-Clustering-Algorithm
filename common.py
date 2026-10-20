"""Shared config + helpers for the block-group clustering feature pipeline.

Contract every feature module follows:
  * Unit of analysis: 2020-vintage Census Block Group, keyed by 12-char string `GEOID`
    (state 2 + county 3 + tract 6 + block group 1). Coverage: 50 states + DC (no PR).
  * ACS vintage: 2019-2023 5-year (ACS_YEAR = 2023).
  * Each module `features/<name>.py` exposes `build() -> pd.DataFrame` (index = GEOID,
    numeric columns only, raw/unstandardized, NaN where unavailable) and, when run as a
    script, writes it to `data/interim/<name>.parquet`.
  * Raw downloads are cached under `data/raw/<name>/` so reruns are offline.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
for _d in (RAW, INTERIM, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

ACS_YEAR = 2023
ACS_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

# 50 states + DC
STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]


def _get(url: str, params: dict, tries: int = 5) -> requests.Response:
    err: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r
            err = requests.HTTPError(f"{r.status_code}: {r.text[:200]}", response=r)
        except requests.RequestException as e:
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"GET {url} failed after {tries} tries") from err


def acs_fetch(variables: list[str], geo: str = "block group", cache_name: str | None = None,
              dataset_url: str = ACS_URL) -> pd.DataFrame:
    """Fetch ACS variables for every state at `geo` ("block group" | "tract" | "county").

    Returns a DataFrame indexed by GEOID (12 chars for block group, 11 for tract,
    5 for county) with numeric columns named exactly as the variables requested.
    ACS negative sentinels (-666666666 etc.) become NaN. Cached to
    data/raw/acs/<cache_name>.parquet when cache_name is given.
    """
    cache = RAW / "acs" / f"{cache_name}.parquet" if cache_name else None
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)

    frames = []
    chunks = [variables[i:i + 45] for i in range(0, len(variables), 45)]  # API limit 50
    for st in STATE_FIPS:
        parts = []
        for chunk in chunks:
            params = {"get": ",".join(chunk), "for": f"{geo}:*", "key": CENSUS_API_KEY}
            if geo == "block group":
                params["in"] = f"state:{st} county:* tract:*"
            elif geo == "tract":
                params["in"] = f"state:{st} county:*"
            elif geo == "county":
                params["in"] = f"state:{st}"
            js = _get(dataset_url, params).json()
            df = pd.DataFrame(js[1:], columns=js[0])
            geo_cols = [c for c in ("state", "county", "tract", "block group") if c in df.columns]
            df["GEOID"] = df[geo_cols].astype(str).agg("".join, axis=1)
            parts.append(df.set_index("GEOID")[chunk])
        frames.append(pd.concat(parts, axis=1))
    out = pd.concat(frames)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.where(out > -555555555)  # ACS annotation sentinels -> NaN
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        out.to_parquet(tmp)
        os.replace(tmp, cache)  # atomic: concurrent writers can't leave a torn file
    return out


def safe_div(num, den):
    """Elementwise num/den with 0 or NaN denominators -> NaN."""
    den = pd.Series(den) if not isinstance(den, pd.Series) else den
    return num / den.where(den > 0)


def save_interim(df: pd.DataFrame, name: str) -> Path:
    assert df.index.name == "GEOID", "index must be named GEOID"
    path = INTERIM / f"{name}.parquet"
    df.to_parquet(path)
    print(f"wrote {path} shape={df.shape}")
    return path
