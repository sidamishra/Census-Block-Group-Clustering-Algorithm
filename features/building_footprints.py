"""Building-footprint features per block group (optional enrichment; independent of geography.py's output).

Unit: 2020-vintage Census Block Group, index = 12-char GEOID. Only BGs in the states
processed are present (all BGs of a processed state appear, with 0 buildings if none).

Columns (raw, unstandardized)
-----------------------------
bldg_count                 Number of building footprints whose centroid falls in the BG.
bldg_density_per_sqkm      bldg_count / BG land area (NaN if ALAND == 0).
bldg_footprint_share       Total footprint area / BG land area (NaN if ALAND == 0). Can
                           slightly exceed what's physically possible on piers/shoreline
                           since footprints over water count but water area does not.
bldg_mean_area_sqm         Mean footprint area, m^2 (NaN if no buildings).
bldg_centroid_lat, bldg_centroid_lon
                           Building-count-weighted mean of footprint centroids (NaN if none).

Sources
-------
* Microsoft US Building Footprints v2 (https://github.com/microsoft/USBuildingFootprints),
  per-state zipped GeoJSON from
  https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/<State>.geojson.zip
  (imagery mostly 2014-2020; ~129M buildings nationally).
* BG polygons + ALAND: TIGER/Line 2023 BG (via features.geography helpers, cached under
  data/raw/geography/).

Method / caveats
----------------
* Each state's zip is streamed with pyogrio/Arrow in batches (bounded memory), areas and
  centroids are computed in EPSG:6933 (WGS84 global equal-area, valid for AK/HI too),
  buildings are assigned to the BG containing their centroid (STRtree point-in-polygon),
  then the raw zip is deleted (set KEEP_RAW=True to keep). Per-state results are cached in
  data/raw/building_footprints/bg_<fips>.parquet so a national run is resumable.
* Buildings whose centroid falls outside every BG of the state (e.g. offshore, across a
  state line due to coordinate precision) are dropped and counted in the log.
* Footprints are ML-derived; small sheds/garages are inconsistently detected, and very
  new construction is missing.

Usage
-----
    python features/building_footprints.py            # DEFAULT_STATES (Delaware demo)
    python features/building_footprints.py 10 11 24   # specific state FIPS
    python features/building_footprints.py all        # all 50 + DC (tens of GB, hours)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, STATE_FIPS, save_interim  # noqa: E402
from features.geography import _download, load_tiger_bg  # noqa: E402

FP_RAW = RAW / "building_footprints"
DEFAULT_STATES = ["10"]  # Delaware demo
KEEP_RAW = False
BATCH = 200_000
URL = "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/{name}.geojson.zip"
AREA_CRS = "EPSG:6933"

STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "DistrictofColumbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
    "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana",
    "31": "Nebraska", "32": "Nevada", "33": "NewHampshire", "34": "NewJersey",
    "35": "NewMexico", "36": "NewYork", "37": "NorthCarolina", "38": "NorthDakota",
    "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "RhodeIsland",
    "45": "SouthCarolina", "46": "SouthDakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "WestVirginia",
    "55": "Wisconsin", "56": "Wyoming",
}


def process_state(st: str) -> pd.DataFrame:
    """Download, stream and aggregate one state's footprints to BGs; delete raw after."""
    import pyogrio
    import shapely
    from pyproj import Transformer

    cache = FP_RAW / f"bg_{st}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    t0 = time.time()
    zpath = _download(URL.format(name=STATE_NAMES[st]), FP_RAW / f"{STATE_NAMES[st]}.geojson.zip")
    zip_mb = zpath.stat().st_size / 1e6
    t_dl = time.time() - t0

    bg = load_tiger_bg(st, geometry=True).to_crs(AREA_CRS)
    bg = bg[["GEOID", "ALAND", "geometry"]].reset_index(drop=True)
    tree = shapely.STRtree(bg.geometry.values)
    nbg = len(bg)
    cnt = np.zeros(nbg)
    area = np.zeros(nbg)
    sx = np.zeros(nbg)
    sy = np.zeros(nbg)
    to_ea = Transformer.from_crs("EPSG:4326", AREA_CRS, always_xy=True)
    n_total = n_unassigned = 0

    with pyogrio.open_arrow(f"zip://{zpath.as_posix()}", batch_size=BATCH,
                            use_pyarrow=True) as (meta, reader):
        geom_col = meta["geometry_name"] or "wkb_geometry"
        for batch in reader:
            geoms = shapely.from_wkb(batch.column(geom_col).to_numpy(zero_copy_only=False))
            geoms = shapely.transform(geoms, lambda xy: np.column_stack(
                to_ea.transform(xy[:, 0], xy[:, 1])))
            a = shapely.area(geoms)
            cen = shapely.centroid(geoms)
            hit_pt, hit_bg = tree.query(cen, predicate="within")
            # a centroid on a shared edge can match 2 BGs -> keep first
            _, first = np.unique(hit_pt, return_index=True)
            hit_pt, hit_bg = hit_pt[first], hit_bg[first]
            cx, cy = shapely.get_x(cen[hit_pt]), shapely.get_y(cen[hit_pt])
            np.add.at(cnt, hit_bg, 1)
            np.add.at(area, hit_bg, a[hit_pt])
            np.add.at(sx, hit_bg, cx)
            np.add.at(sy, hit_bg, cy)
            n_total += len(geoms)
            n_unassigned += len(geoms) - len(hit_pt)

    to_ll = Transformer.from_crs(AREA_CRS, "EPSG:4326", always_xy=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / cnt, sy / cnt
    lon, lat = to_ll.transform(mx, my)
    has = cnt > 0
    aland_km = bg.ALAND.astype(float).values / 1e6
    aland_m = np.where(aland_km > 0, aland_km * 1e6, np.nan)
    out = pd.DataFrame({
        "bldg_count": cnt,
        "bldg_density_per_sqkm": cnt / np.where(aland_km > 0, aland_km, np.nan),
        "bldg_footprint_share": area / aland_m,
        "bldg_mean_area_sqm": np.where(has, area / np.where(has, cnt, 1), np.nan),
        "bldg_centroid_lat": np.where(has, lat, np.nan),
        "bldg_centroid_lon": np.where(has, lon, np.nan),
    }, index=pd.Index(bg.GEOID.astype(str).values, name="GEOID"))
    out.to_parquet(cache)

    if not KEEP_RAW:
        zpath.unlink(missing_ok=True)
    print(f"[{st} {STATE_NAMES[st]}] zip={zip_mb:.1f}MB buildings={n_total:,} "
          f"unassigned={n_unassigned:,} BGs={nbg} download={t_dl:.1f}s "
          f"total={time.time() - t0:.1f}s")
    return out


def build(states: list[str] | None = None) -> pd.DataFrame:
    states = states or DEFAULT_STATES
    return pd.concat([process_state(st) for st in states]).sort_index()


if __name__ == "__main__":
    args = sys.argv[1:]
    sts = STATE_FIPS if args == ["all"] else (args or DEFAULT_STATES)
    save_interim(build(sts), "building_footprints")
