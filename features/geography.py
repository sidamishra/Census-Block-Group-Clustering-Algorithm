"""Geography features: block-group location (for distance) + land area + population density.

Unit: 2020-vintage Census Block Group, index = 12-char GEOID (50 states + DC, no PR).

Columns (all raw, unstandardized)
---------------------------------
lat, lon               Population-weighted center (2020 Census Centers of Population, BG
                       level). Fallback for BGs without a CenPop row / with zero 2020
                       population: TIGER/Line 2023 BG internal point (INTPTLAT/INTPTLON).
coord_is_fallback      1.0 if lat/lon came from the TIGER internal point, else 0.0.
ecef_x_km, ecef_y_km, ecef_z_km
                       Earth-Centered Earth-Fixed coordinates (WGS84 ellipsoid, km).
                       ** USE THESE FOR DISTANCE IN CLUSTERING. ** Euclidean distance in
                       ECEF is the straight-line chord through the Earth, which equals the
                       great-circle distance to within 0.1% up to ~700 km and is
                       monotone in it beyond that. Works uniformly for AK, HI, CONUS.
x_m, y_m               EPSG:5070 (NAD83 / CONUS Albers equal-area) easting/northing, meters.
                       Good for CONUS (distance error a few % at most). Provided for
                       convenience/plotting; AK and HI are far outside the projection's
                       design area so local distances there are distorted -> prefer ECEF.
aland_sqkm             BG land area (TIGER/Line 2023 ALAND / 1e6).
awater_sqkm            BG water area (TIGER/Line 2023 AWATER / 1e6).
population             ACS 2019-2023 5-year total population (B01003_001E).
pop_density_per_sqkm   population / aland_sqkm (NaN if aland == 0).

Sources
-------
* https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt
* https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_<ss>_bg.zip (per state)
* ACS 2023 5-year API, B01003_001E via common.acs_fetch.
Raw files cached under data/raw/geography/ (TIGER zips are kept because
features/building_footprints.py reuses the BG polygons).

Caveats
-------
* CenPop centers use 2020 decennial population; ACS population is 2019-2023. Same 2020
  BG geography, so GEOIDs align; the ACS BG universe is the TIGER 2023 BG universe.
* Connecticut: TIGER 2023 and ACS 2023 both use CT planning-region county codes (09110..
  09190) while CenPop2020 uses the old 8 CT counties (09001..09015). CT CenPop rows are
  re-keyed onto the new GEOIDs by tract+BG (the tract/BG codes were preserved), see
  `_ct_crosswalk`. Any CT BG still unmatched falls back to the TIGER internal point.
* Row universe = ACS 2023 BGs (239,741). TIGER 2023 has 40 extra BGs (Suffolk Co. NY
  36103 tracts 1224xx/1225xx/1456xx-1462xx/2012xx, and Ulster 36111954401) that ACS 2023
  does not report; they are dropped (printed at build time).
* Water-only BGs (aland == 0, e.g. GEOIDs ending in "0") get NaN density.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, STATE_FIPS, acs_fetch, safe_div, save_interim  # noqa: E402

GEO_RAW = RAW / "geography"
CENPOP_URL = "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt"
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_{st}_bg.zip"


def _download(url: str, dest: Path, tries: int = 5) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for _ in range(tries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"download failed {url}: {last}")


def tiger_bg_zip(st: str) -> Path:
    """Path to the (cached) TIGER/Line 2023 BG shapefile zip for state FIPS `st`."""
    return _download(TIGER_URL.format(st=st), GEO_RAW / f"tl_2023_{st}_bg.zip")


def load_tiger_bg(st: str, geometry: bool = True):
    """Read one state's TIGER 2023 BGs (GeoDataFrame if geometry else DataFrame)."""
    import pyogrio
    path = f"zip://{tiger_bg_zip(st).as_posix()}"
    return pyogrio.read_dataframe(path, read_geometry=geometry)


def tiger_attributes() -> pd.DataFrame:
    """National TIGER BG attribute table (GEOID, ALAND, AWATER, INTPTLAT, INTPTLON)."""
    cache = GEO_RAW / "tiger2023_bg_attrs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(tiger_bg_zip, STATE_FIPS))
    frames = []
    for st in STATE_FIPS:
        df = load_tiger_bg(st, geometry=False)
        frames.append(df[["GEOID", "ALAND", "AWATER", "INTPTLAT", "INTPTLON"]])
    out = pd.concat(frames, ignore_index=True)
    out["GEOID"] = out["GEOID"].astype(str)
    for c in ("ALAND", "AWATER", "INTPTLAT", "INTPTLON"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.set_index("GEOID")
    out.to_parquet(cache)
    return out


def cenpop() -> pd.DataFrame:
    path = _download(CENPOP_URL, GEO_RAW / "CenPop2020_Mean_BG.txt")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df["GEOID"] = df.STATEFP + df.COUNTYFP + df.TRACTCE + df.BLKGRPCE
    out = pd.DataFrame({
        "GEOID": df.GEOID,
        "cp_pop": pd.to_numeric(df.POPULATION),
        "cp_lat": pd.to_numeric(df.LATITUDE),
        "cp_lon": pd.to_numeric(df.LONGITUDE),
    })
    return out.set_index("GEOID")


def _ct_crosswalk(cp: pd.DataFrame, universe: pd.Index) -> pd.DataFrame:
    """Re-key CT CenPop rows (old counties) onto 2022+ planning-region GEOIDs.

    Tract+BG codes (last 7 chars) are unique within CT and unchanged by the county
    redefinition, so we match on GEOID[5:].
    """
    is_ct = cp.index.str.startswith("09")
    ct = cp[is_ct].copy()
    new_ct = pd.Series(universe[universe.str.startswith("09")])
    if ct.index.isin(new_ct).all():  # already new codes
        return cp
    keys = new_ct.str[5:]
    ok = ~keys.duplicated(keep=False)  # only remap tract+BG keys unique within CT
    key_to_new = dict(zip(keys[ok], new_ct[ok]))
    ct.index = ct.index.str[5:].map(key_to_new)
    ct = ct[ct.index.notna()]
    ct.index.name = "GEOID"
    return pd.concat([cp[~is_ct], ct])


def ecef_km(lat_deg, lon_deg):
    a = 6378.137
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = n * np.cos(lat) * np.cos(lon)
    y = n * np.cos(lat) * np.sin(lon)
    z = n * (1 - e2) * np.sin(lat)
    return x, y, z


def build() -> pd.DataFrame:
    tiger = tiger_attributes()
    tiger = tiger[tiger.index.str[:2].isin(STATE_FIPS)]
    cp = _ct_crosswalk(cenpop(), tiger.index)
    acs = acs_fetch(["B01003_001E"], cache_name="b01003_bg")

    # Universe = ACS 2023 BGs (the contract's universe). TIGER-only BGs are reported.
    idx = acs.index
    df = pd.DataFrame(index=pd.Index(idx, name="GEOID"))
    df = df.join(tiger).join(cp)

    use_cp = df.cp_lat.notna() & df.cp_lon.notna() & (df.cp_pop.fillna(0) > 0)
    df["lat"] = np.where(use_cp, df.cp_lat, df.INTPTLAT)
    df["lon"] = np.where(use_cp, df.cp_lon, df.INTPTLON)
    df["coord_is_fallback"] = np.where(df.lat.notna(), (~use_cp).astype(float), np.nan)

    df["ecef_x_km"], df["ecef_y_km"], df["ecef_z_km"] = ecef_km(df.lat.values, df.lon.values)

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(df.lon.values, df.lat.values)
    df["x_m"] = np.where(df.lat.notna(), x, np.nan)
    df["y_m"] = np.where(df.lat.notna(), y, np.nan)

    df["aland_sqkm"] = df.ALAND / 1e6
    df["awater_sqkm"] = df.AWATER / 1e6
    df["population"] = acs["B01003_001E"].reindex(df.index)
    df["pop_density_per_sqkm"] = safe_div(df.population, df.aland_sqkm)

    cols = ["lat", "lon", "coord_is_fallback", "ecef_x_km", "ecef_y_km", "ecef_z_km",
            "x_m", "y_m", "aland_sqkm", "awater_sqkm", "population", "pop_density_per_sqkm"]
    out = df[cols].astype(float).sort_index()

    # --- validation report ---
    tiger_only = tiger.index.difference(acs.index)
    print(f"rows={len(out)}  ACS BGs={len(acs)}  TIGER BGs={len(tiger)}")
    print(f"in ACS not TIGER: {len(acs.index.difference(tiger.index))}  "
          f"in TIGER not ACS (dropped): {len(tiger_only)} {sorted(set(tiger_only.str[:11]))}")
    print(f"missing coords: {out.lat.isna().sum()}  fallback coords: "
          f"{int(out.coord_is_fallback.sum())}  CenPop rows unmatched: "
          f"{(~cp.index.isin(out.index)).sum()}")
    return out


if __name__ == "__main__":
    save_interim(build(), "geography")
