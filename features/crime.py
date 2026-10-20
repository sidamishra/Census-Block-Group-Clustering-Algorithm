"""Crime-rate features per 2020-vintage Census Block Group (50 states + DC).

Columns (all per 100,000 residents unless noted; raw, unstandardized)
----------------------------------------------------------------------
violent_crime_rate   FBI violent crime (murder/nonnegligent manslaughter, rape, robbery,
                     aggravated assault) offenses known to law enforcement per 100k.
property_crime_rate  FBI property crime (burglary, larceny-theft, motor vehicle theft) per 100k.
crime_geo_level      Geography whose rate the BG inherits:
                       1 = city/town/village/township police agency (Census incorporated
                           place, or county subdivision in MCD states) containing the BG's
                           2020 population-weighted centroid
                       2 = county "remainder": sheriff / county-PD offenses (or a
                           consolidated metro agency such as LVMPD, Charlotte-Mecklenburg,
                           Honolulu, Nashville Metro, Athens-Clarke, Jacksonville SO) divided by
                           the county ACS population living outside level-1 jurisdictions.
                           Requires remainder pop >= 1,000 and a "patrol" test (property rate
                           >= 50/100k; NJ/PA-style non-patrol sheriffs fail it)
                       3 = county aggregate: all FBI offenses attributable to the county
                           (reporting cities' offenses apportioned by BG pop + patrol-agency
                           offenses) / population they cover (>= 1,000); used when level 2 is
                           unavailable. Tends to give rural BGs town-like rates.
                       4 = county, County Health Rankings fallback (no usable FBI data)
                     NaN = no FBI data and no CHR value (~0.3% of BGs; mostly IN, MS, WV, AK)
crime_source         1 = FBI Crime Data Explorer, data year 2025
                     2 = County Health Rankings 2022 release (violent crime, UCR 2014 & 2016);
                         property_crime_rate for these BGs is imputed as CHR violent rate x the
                         state's FBI property/violent ratio (pop-weighted over FBI BGs)

Sources
-------
* FBI Crime Data Explorer (CDE) bulk "Offenses Known to Law Enforcement", data year 2025
  (CIUS-style tables, downloaded keyless through CDE's public S3 signed-URL endpoint):
    - Table 8  by State by City (city/town/township agencies, with FBI-estimated population)
    - Table 10 by State by Metropolitan and Nonmetropolitan County (sheriffs + county PDs;
      offenses only - no population, so the denominator is derived from ACS)
  Table 9 (universities) and Table 11 (state police, tribal, other) are NOT used: they
  cannot be placed geographically below the state level.
* Census 2023 cartographic boundaries (cb_2023_us_place_500k, cb_2023_us_cousub_500k,
  cb_2023_us_county_500k) and 2020 population-weighted BG centroids (CenPop2020_Mean_BG).
* ACS 2019-2023 B01003 BG population (via common.acs_fetch) for county/remainder denominators.
* County Health Rankings 2022 analytic data (the last CHR release carrying "Violent crime";
  the 2023+ releases, incl. the project's chr2024.csv, dropped the measure).

Method
------
1. BG pop-weighted centroids are point-in-polygon joined to incorporated places (CDPs
   excluded - they have no police) and to functioning county subdivisions (CCDs/unorganized
   territories excluded).
2. Table 8 agencies are matched by state + normalized name (suffix-aware: "X Village",
   "X Town", "X Township, Y County"); places first, then county subdivisions; ties broken by
   county qualifier and then by closeness of FBI population to the ACS population of the
   candidate. Matches whose FBI pop differs >3x (agencies >= 2,000 pop) or >10x (smaller)
   from the candidate's centroid-based ACS pop are rejected (regional/multi-municipal PDs,
   places containing no BG centroid). A short ALIASES table handles name exceptions.
   BGs inside a matched jurisdiction get that agency's rate (offenses / FBI population).
   Place match takes precedence over subdivision (e.g. NY village inside a town).
3. Remaining BGs get their county's sheriff/county-PD offenses / (county ACS pop minus pop of
   level-1 BGs) [level 2], else the county aggregate [level 3], else CHR [level 4].
4. Connecticut: ACS 2023 uses planning regions (09110-09190); CHR uses the old 8 counties.
   FBI data are agency-level so no crosswalk is needed for them (CT county aggregates are by
   planning region). For CHR, CT BGs are mapped to the old county through their unchanged
   tract+BG code in the 2020 centroid file (only pop-0 water tracts collide).

Caveats
-------
* Agency coverage is incomplete: Table 8/10 list only agencies that submitted data for the
  year. Florida and North Dakota were flagged "limited data" for 2025; Alaska, CT, MA, RI, DC
  have no county agencies at all. Non-reporting cities inside a county inflate the remainder
  denominator (diluting level-2 rates).
* State police (Table 11) patrol much rural territory in e.g. PA, NY, NJ, VT, NH, CT, MA, DE;
  their offenses are omitted, so level-2/3 rural rates there are under/over-stated.
* NIBRS transition: the FBI went NIBRS-only in 2021; agencies still converting may report
  partial years or be missing, and NIBRS->SRS conversions can undercount (hierarchy rule
  differences). Rates are offenses known, not victimization; small-population agencies
  produce volatile rates (commercial enclaves such as Rosemont IL or Commerce CA exceed
  30,000 property crimes/100k because offenses are counted where they occur).
* Validation: county pop-weighted FBI violent rate vs CHR 2022 (UCR 2014/16): Pearson ~0.69,
  Spearman ~0.67; FBI 2025 levels run ~20-40% lower than CHR in most states (genuine decline
  plus NIBRS/coverage undercount), much lower in CT/WV/GA/FL.
* Resolution is jurisdictional (city or county remainder), not truly BG-level: every BG in a
  jurisdiction shares one value.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, STATE_FIPS, acs_fetch, save_interim  # noqa: E402

CRIME_RAW = RAW / "crime"
CRIME_RAW.mkdir(parents=True, exist_ok=True)
CDE_SIGN = "https://cde.ucr.cjis.gov/LATEST/s3/signedurl"
CB = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_{}_500k.zip"
CENPOP = "https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt"
CHR_URL = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2022.csv"
OFFENSE_COLS = ["violent", "murder", "rape", "robbery", "agg_assault",
                "property", "burglary", "larceny", "mvt", "arson"]


# ----------------------------------------------------------------------------- downloads
def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def _cde_file(key: str, dest: Path) -> Path:
    if not dest.exists():
        js = requests.get(CDE_SIGN, params={"key": key}, timeout=60).json()
        if key not in js:
            raise RuntimeError(f"CDE signed URL not available for {key}")
        _download(js[key], dest)
    return dest


# ----------------------------------------------------------------------------- names
_SUFFIXES = ["charter township", "township", "town", "village", "city", "borough",
             "plantation", "municipality", "metropolitan government", "unified government",
             "consolidated government", "urban county", "city and borough", "metro"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"\d+$", "", s).strip()                 # footnote markers
    s = s.replace("&", " and ")
    s = re.sub(r"[.'`,()]", "", s)
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bste\b", "sainte", s)
    s = re.sub(r"\bmt\b", "mount", s)
    s = re.sub(r"\bft\b", "fort", s)
    s = re.sub(r"\btwp\b", "township", s)
    s = re.sub(r"\bcharter township\b", "township", s)
    s = re.sub(r"\bbalance\b", "", s)                 # consolidated-city "(balance)"
    return re.sub(r"\s+", " ", s).strip()


def _base(s: str) -> str:
    """Normalized name with trailing governmental suffixes removed (iteratively, so
    'weymouth town city' -> 'weymouth', 'boise city city' -> 'boise')."""
    changed = True
    while changed:
        changed = False
        for suf in sorted(_SUFFIXES, key=len, reverse=True):
            if s.endswith(" " + suf) and len(s) > len(suf) + 2:
                s = s[: -len(suf) - 1].strip()
                changed = True
                break
    return s


# Agencies whose names cannot be matched by rule. Values: ("key", normalized candidate
# name) or ("county", FIPS) for consolidated/metro agencies that police an entire county
# except separately-reporting cities -> treated like a county agency (level-2 remainder).
ALIASES = {
    ("Nevada", "las vegas metropolitan police department"): ("county", "32003"),
    ("North Carolina", "charlotte mecklenburg"): ("county", "37119"),
    ("Hawaii", "honolulu"): ("county", "15003"),
    ("Tennessee", "metropolitan nashville police department"): ("county", "47037"),
    ("Georgia", "athens clarke county"): ("county", "13059"),
    ("Florida", "jacksonville sheriffs office"): ("county", "12031"),
    ("Montana", "anaconda deer lodge"): ("county", "30023"),
    ("Montana", "butte silver bow"): ("county", "30093"),
    ("New Mexico", "los alamos"): ("county", "35028"),
    ("California", "ventura"): ("key", "san buenaventura ventura"),
    ("New Jersey", "camden county police department"): ("key", "camden"),
    ("Florida", "lake worth"): ("key", "lake worth beach"),
    ("New York", "webster town and village"): ("key", "webster town"),
    ("New York", "camillus town and village"): ("key", "camillus town"),
    ("New York", "new hartford town and village"): ("key", "new hartford town"),
    ("New York", "dewitt town"): ("key", "de witt town"),
    ("New Jersey", "orange city"): ("key", "city of orange township"),
    ("California", "paso robles"): ("key", "el paso de robles paso robles"),
    ("Massachusetts", "north attleboro"): ("key", "north attleborough"),
    ("Massachusetts", "middleboro"): ("key", "middleborough"),
}


# ----------------------------------------------------------------------------- loaders
def _read_cde_table(path: Path, skip: int, cols: list[str]) -> pd.DataFrame:
    df = pd.read_excel(path, header=None, skiprows=skip)
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    df["state"] = df["state"].ffill()
    df = df[df["state"].astype(str).str.len() < 40]           # drop footnote rows
    df["state"] = df["state"].astype(str).str.replace(r"\d+$", "", regex=True).str.strip()
    df = df[df.iloc[:, 1].notna()]
    for c in OFFENSE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _load_cities() -> pd.DataFrame:
    p = _cde_file("_all/Table8.xlsx", CRIME_RAW / "cde_Table8.xlsx")
    df = _read_cde_table(p, 4, ["state", "agency", "fbi_pop"] + OFFENSE_COLS)
    df["fbi_pop"] = pd.to_numeric(df["fbi_pop"], errors="coerce")
    return df


def _load_counties() -> pd.DataFrame:
    p = _cde_file("_all/Table10.xlsx", CRIME_RAW / "cde_Table10.xlsx")
    return _read_cde_table(p, 5, ["state", "metro", "agency"] + OFFENSE_COLS)


def _bg_points(bg_index: pd.Index) -> gpd.GeoDataFrame:
    p = _download(CENPOP, CRIME_RAW / "CenPop2020_Mean_BG.txt")
    c = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    c["GEOID"] = c.STATEFP + c.COUNTYFP + c.TRACTCE + c.BLKGRPCE
    c["old_county"] = c.STATEFP + c.COUNTYFP
    # Connecticut: 2020 file uses old counties; ACS 2023 uses planning regions with the
    # same tract+BG codes. Remap by state + tract + BG (only water tracts collide).
    ct_new = pd.Series(bg_index[bg_index.str[:2] == "09"])
    ct_new = ct_new[~ct_new.str[5:].duplicated(keep=False)]
    ct_map = dict(zip("09" + ct_new.str[5:], ct_new))
    is_ct = c.STATEFP == "09"
    key = "09" + c.loc[is_ct, "TRACTCE"] + c.loc[is_ct, "BLKGRPCE"]
    c.loc[is_ct, "GEOID"] = key.map(ct_map)
    c = c.dropna(subset=["GEOID"]).drop_duplicates("GEOID").set_index("GEOID")
    c = c.reindex(bg_index)
    pts = gpd.GeoDataFrame(
        c[["old_county"]],
        geometry=gpd.points_from_xy(pd.to_numeric(c.LONGITUDE), pd.to_numeric(c.LATITUDE)),
        crs="EPSG:4269")
    return pts


def _boundaries():
    place = gpd.read_file(_download(CB.format("place"), CRIME_RAW / "cb_2023_us_place_500k.zip"))
    place = place[place.LSAD != "57"]                           # drop CDPs
    cous = gpd.read_file(_download(CB.format("cousub"), CRIME_RAW / "cb_2023_us_cousub_500k.zip"))
    cous = cous[~cous.LSAD.isin(["22", "23", "00"])
                & ~cous.NAME.str.contains("Unorganized|not defined", case=False)]
    county = gpd.read_file(_download(CB.format("county"), CRIME_RAW / "cb_2023_us_county_500k.zip"),
                           ignore_geometry=True)
    return place, cous, county


def _load_chr() -> pd.Series:
    p = _download(CHR_URL, CRIME_RAW / "chr2022_analytic.csv")
    ch = pd.read_csv(p, dtype=str, skiprows=[1])
    ch = ch[ch["County FIPS Code"] != "000"]
    return pd.to_numeric(ch.set_index("5-digit FIPS Code")["Violent crime raw value"],
                         errors="coerce").dropna()


# ----------------------------------------------------------------------------- matching
def _match_cities(cities: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """cand: rows of jurisdictions with columns jid, kind (place/cousub), STATE_NAME,
    full (normalized NAMELSAD), base (normalized NAME), county_name, acs_pop."""
    lookup: dict[tuple, set] = {}
    for col in ("full", "base", "base2"):
        for k, ix in cand.groupby(["STATE_NAME", col]).indices.items():
            lookup.setdefault(k, set()).update(ix.tolist())
    out = []
    for i, r in cities.iterrows():
        raw = str(r.agency).strip()
        cty = None
        if "," in raw:
            raw, cty = raw.split(",", 1)
            cty = _norm(cty).replace(" county", "").replace(" parish", "")
        n = _norm(raw)
        st = r.state
        alias = ALIASES.get((st, n))
        if alias and alias[0] == "county":
            out.append((i, "K" + alias[1], np.nan))       # handled as a county agency
            continue
        keys = [alias[1]] if alias else [n, _base(n)]
        ix: set = set()
        for k in keys:
            ix |= lookup.get((st, k), set())
        if not ix:
            out.append((i, None, np.nan))
            continue
        hits = cand.iloc[sorted(ix)]
        if cty is not None:
            h2 = hits[hits.county_name == cty]
            hits = h2 if len(h2) else hits
        # suffix-aware preference: "X Township"/"X Town" -> cousub; "X Village"/"X City" ...
        last = (alias[1] if alias else n).split(" ")[-1]
        if last in ("township", "town", "village", "city", "borough"):
            h2 = hits[hits.full.str.endswith(" " + last)]
            if last in ("township", "town"):
                h3 = h2[h2.kind == "cousub"]
                h2 = h3 if len(h3) else h2
            hits = h2 if len(h2) else hits
        # closest population, then place before county subdivision
        lr = np.abs(np.log(hits.acs_pop.clip(lower=1) / max(r.fbi_pop, 1)))
        hits = hits.assign(_lr=lr.round(1), _k=(hits.kind != "place").astype(int)) \
            .sort_values(["_lr", "_k"])
        best = hits.iloc[0]
        out.append((i, best.jid, (best.acs_pop + 1) / (r.fbi_pop + 1)))
    m = pd.DataFrame(out, columns=["i", "jid", "pop_ratio"]).set_index("i")
    return cities.join(m)


# ----------------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    pop = acs_fetch(["B01003_001E"], cache_name="b01003_bg")["B01003_001E"]
    pop = pop[pop.index.str[:2].isin(STATE_FIPS)]
    bg = pd.DataFrame({"pop": pop.fillna(0)})
    bg["county"] = bg.index.str[:5]

    pts = _bg_points(bg.index)
    place, cous, county = _boundaries()

    # BG centroid -> jurisdiction polygons
    j1 = gpd.sjoin(pts[["geometry"]], place[["GEOID", "geometry"]].rename(columns={"GEOID": "JID"}).to_crs(pts.crs),
                   how="left", predicate="within")
    j1 = j1[~j1.index.duplicated()]
    j2 = gpd.sjoin(pts[["geometry"]], cous[["GEOID", "geometry"]].rename(columns={"GEOID": "JID"}).to_crs(pts.crs),
                   how="left", predicate="within")
    j2 = j2[~j2.index.duplicated()]
    bg["place"] = "P" + j1["JID"].reindex(bg.index)
    bg["cousub"] = "C" + j2["JID"].reindex(bg.index)
    bg["old_county"] = pts["old_county"]

    # candidate jurisdictions with ACS pop (sum of BGs whose centroid falls inside)
    cn = county.set_index("GEOID")
    cn_name = cn.NAME.map(_norm)

    def _cand(gdf, kind, col):
        d = pd.DataFrame({
            "jid": ("P" if kind == "place" else "C") + gdf.GEOID.values,
            "kind": kind, "STATE_NAME": gdf.STATE_NAME.values,
            "full": gdf.NAMELSAD.map(_norm).values, "base": gdf.NAME.map(_norm).values,
            "county_name": (cn_name.reindex(gdf.GEOID.str[:5]).values if kind == "cousub"
                            else None)})
        d["acs_pop"] = d.jid.map(bg.groupby(col)["pop"].sum()).fillna(0).values
        return d

    cand = pd.concat([_cand(place, "place", "place"), _cand(cous, "cousub", "cousub")],
                     ignore_index=True)
    cand["base2"] = cand.full.map(_base)
    # consolidated cities ("nashville davidson metropolitan government" -> "nashville")
    extra = cand[cand.base2.str.contains(" ") & cand.full.str.contains(
        "metropolitan government|unified government|consolidated government|urban county|"
        "city and borough|municipality")].copy()
    extra["base"] = extra["base2"] = extra.base2.str.split(" ").str[0]
    cand = pd.concat([cand, extra], ignore_index=True)

    cities = _load_cities()
    cities = cities[cities.fbi_pop > 0]
    cities = _match_cities(cities, cand)
    is_k = cities.jid.fillna("").str.startswith("K")
    # reject implausible population mismatches (only meaningful for non-trivial jurisdictions)
    big = cities.fbi_pop >= 2000
    bad = cities.jid.notna() & ~is_k & (
        (big & ((cities.pop_ratio > 3) | (cities.pop_ratio < 1 / 3)))
        | (~big & ((cities.pop_ratio > 10) | (cities.pop_ratio < 0.1))))
    cities.loc[bad, "jid"] = None
    # several agencies can match one jurisdiction (rare) -> pool them
    agg = cities[cities.jid.notna() & ~is_k].groupby("jid")[["fbi_pop", "violent", "property"]].sum()
    agg["v"] = agg.violent / agg.fbi_pop * 1e5
    agg["p"] = agg.property / agg.fbi_pop * 1e5

    bg["v"] = np.nan
    bg["p"] = np.nan
    bg["level"] = np.nan
    bg["src"] = np.nan
    bg["agency_jid"] = None
    for col in ("cousub", "place"):              # place last -> takes precedence
        hit = bg[col].isin(agg.index)
        bg.loc[hit, "agency_jid"] = bg.loc[hit, col]
    lv1 = bg.agency_jid.notna()
    bg.loc[lv1, "v"] = bg.loc[lv1, "agency_jid"].map(agg.v)
    bg.loc[lv1, "p"] = bg.loc[lv1, "agency_jid"].map(agg.p)
    bg.loc[lv1, ["level", "src"]] = [1, 1]

    # ---- county agencies (Table 10)
    ca = _load_counties()
    nm = ca.agency.map(_norm)
    nm = nm.str.replace(r"\s*(county )?police department$", "", regex=True)
    nm = nm.str.replace(r"\s+(county|parish|borough)$", "", regex=True)
    nm = nm.replace({"unified": "salt lake", "hartsville trousdale": "trousdale"})
    ca["cname"] = nm.str.replace(" ", "")                         # "la salle" == "lasalle"
    cty = county[~county.NAMELSAD.str.endswith(" city")].copy()   # VA/MD/MO indep. cities
    cty["cname"] = cty.NAME.map(_norm).str.replace(" ", "")
    cty = cty.drop_duplicates(["STATE_NAME", "cname"])
    ca = ca.merge(cty[["STATE_NAME", "cname", "GEOID"]], left_on=["state", "cname"],
                  right_on=["STATE_NAME", "cname"], how="left")
    for (st_, an), (kind, fips) in ALIASES.items():
        if kind == "county":
            ca.loc[(ca.state == st_) & (ca.agency.map(_norm) == an), "GEOID"] = fips
    unmatched_ca = ca[ca.GEOID.isna()]
    # consolidated metro agencies from Table 8 (e.g. LVMPD) act as the county agency
    metro = cities[is_k].assign(GEOID=cities.jid[is_k].str[1:])
    ca = pd.concat([ca, metro[["state", "agency", "GEOID", "violent", "property"]]],
                   ignore_index=True)
    sher = ca.dropna(subset=["GEOID"]).groupby("GEOID")[["violent", "property"]].sum()

    cpop = bg.groupby("county")["pop"].sum()
    lv1pop = bg[lv1].groupby("county")["pop"].sum().reindex(cpop.index).fillna(0)
    rem = cpop - lv1pop
    # city offenses/pop attributed to county by where the city's BGs sit (pop-weighted)
    bgl = bg[lv1].copy()
    bgl["share"] = bgl["pop"] / bgl.groupby("agency_jid")["pop"].transform("sum")
    for c in ("violent", "property", "fbi_pop"):
        bgl[c] = bgl.agency_jid.map(agg[c]) * bgl.share
    city_c = bgl.groupby("county")[["violent", "property", "fbi_pop"]].sum()

    cdf = pd.DataFrame({"pop": cpop, "rem": rem}).join(sher).join(city_c, rsuffix="_city")
    cdf = cdf.fillna({"violent_city": 0, "property_city": 0, "fbi_pop": 0})
    has_sher = cdf.violent.notna() | cdf.property.notna()
    # a county agency with ~no property crime on its remainder population is not a general
    # patrol agency (NJ/PA sheriffs, Allegheny Co. PD...) -> use the county aggregate instead
    use2 = has_sher & (cdf.rem >= 1000) & (cdf.property / cdf.rem * 1e5 >= 50)
    cdf["v2"] = np.where(use2, cdf.violent / cdf.rem * 1e5, np.nan)
    cdf["p2"] = np.where(use2, cdf.property / cdf.rem * 1e5, np.nan)
    # county aggregate; a non-patrol county agency (failed test above) is left out
    patrol = has_sher & ((cdf.rem < 1000) | (cdf.property / cdf.rem * 1e5 >= 50))
    cov_pop = cdf.fbi_pop + np.where(patrol, cdf.rem.clip(lower=0), 0)
    tot_v = cdf.violent_city + cdf.violent.where(patrol, 0).fillna(0)
    tot_p = cdf.property_city + cdf.property.where(patrol, 0).fillna(0)
    ok3 = (cov_pop >= 1000) & ((cdf.fbi_pop > 0) | patrol)
    cdf["v3"] = np.where(ok3, tot_v / cov_pop.where(cov_pop > 0) * 1e5, np.nan)
    cdf["p3"] = np.where(ok3, tot_p / cov_pop.where(cov_pop > 0) * 1e5, np.nan)

    rest = bg.level.isna()
    c2 = bg.county.map(cdf.v2).notna() & rest
    bg.loc[c2, "v"] = bg.loc[c2, "county"].map(cdf.v2)
    bg.loc[c2, "p"] = bg.loc[c2, "county"].map(cdf.p2)
    bg.loc[c2, ["level", "src"]] = [2, 1]
    rest = bg.level.isna()
    c3 = bg.county.map(cdf.v3).notna() & rest
    bg.loc[c3, "v"] = bg.loc[c3, "county"].map(cdf.v3)
    bg.loc[c3, "p"] = bg.loc[c3, "county"].map(cdf.p3)
    bg.loc[c3, ["level", "src"]] = [3, 1]

    # ---- CHR fallback (old CT counties via 2020 county code)
    chr_v = _load_chr()
    rest = bg.level.isna()
    key = np.where(bg.index.str[:2] == "09", bg.old_county, bg.county)
    chr_bg = pd.Series(key, index=bg.index).map(chr_v)
    c4 = rest & chr_bg.notna()
    fbi = bg[bg.src == 1]
    w = fbi["pop"]
    st_ratio = ((fbi.p * w).groupby(fbi.index.str[:2]).sum()
                / (fbi.v * w).groupby(fbi.index.str[:2]).sum())
    nat_ratio = (fbi.p * w).sum() / (fbi.v * w).sum()
    bg.loc[c4, "v"] = chr_bg[c4]
    bg.loc[c4, "p"] = chr_bg[c4] * bg.index[c4].str[:2].map(st_ratio).fillna(nat_ratio).values
    bg.loc[c4, ["level", "src"]] = [4, 2]

    # diagnostics for the run log
    global DIAG
    DIAG = {"cities": cities, "unmatched_county_agencies": unmatched_ca, "county": cdf,
            "bg": bg, "chr": chr_v}

    out = pd.DataFrame({
        "violent_crime_rate": bg.v.astype(float),
        "property_crime_rate": bg.p.astype(float),
        "crime_geo_level": bg.level.astype(float),
        "crime_source": bg.src.astype(float),
    }, index=bg.index)
    out.index.name = "GEOID"
    return out


DIAG: dict = {}


def _report(df: pd.DataFrame) -> None:
    bg, cities, chr_v = DIAG["bg"], DIAG["cities"], DIAG["chr"]
    print(f"BGs: {len(df):,}  non-null violent: {df.violent_crime_rate.notna().mean():.2%}")
    lv = df.crime_geo_level.value_counts(dropna=False).sort_index()
    pw = bg.groupby(bg.level, dropna=False)["pop"].sum() / bg["pop"].sum()
    print("geo level share of BGs / pop:\n", pd.DataFrame({"bgs": lv / len(df), "pop": pw}))
    print("medians:\n", df.groupby("crime_geo_level")[["violent_crime_rate",
                                                       "property_crime_rate"]].median())
    print("overall medians:", df[["violent_crime_rate", "property_crime_rate"]].median().to_dict())
    m = cities.jid.notna()
    print(f"Table 8 agencies: {len(cities):,}, matched {m.mean():.1%}, "
          f"pop matched {cities.fbi_pop[m].sum() / cities.fbi_pop.sum():.1%}")
    print("largest unmatched cities:\n",
          cities[~m].sort_values("fbi_pop", ascending=False)[["state", "agency", "fbi_pop"]]
          .head(25).to_string())
    print("unmatched county agencies:", DIAG["unmatched_county_agencies"][["state", "agency"]]
          .to_string())
    # county validation vs CHR (pop-weighted mean of BG rates)
    b = bg[bg.src == 1].copy()
    b["key"] = np.where(b.index.str[:2] == "09", b.old_county, b.county)
    b["vw"] = b.v * b["pop"]
    cty = b.groupby("key")[["vw", "pop"]].sum()
    cty = (cty.vw / cty["pop"]).rename("fbi").to_frame().join(chr_v.rename("chr"), how="inner")
    cty = cty.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"county FBI vs CHR2022 violent (n={len(cty)}): pearson={cty.corr().iloc[0, 1]:.3f} "
          f"spearman={cty.corr('spearman').iloc[0, 1]:.3f} "
          f"log-pearson={np.log1p(cty).corr().iloc[0, 1]:.3f}")
    st = cty.groupby(cty.index.str[:2]).apply(lambda d: d.corr("spearman").iloc[0, 1]
                                              if len(d) > 5 else np.nan)
    print("lowest state spearman:", st.sort_values().head(8).round(2).to_dict())
    rat = cty.groupby(cty.index.str[:2]).median()
    print("state median fbi/chr ratio:", (rat.fbi / rat.chr).sort_values().round(2).to_dict())


if __name__ == "__main__":
    df = build()
    _report(df)
    save_interim(df, "crime")
