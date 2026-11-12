"""Flood and exposure model on real ground data (Copernicus elevation + Sentinel-2 land cover).

For each city a 120 x 120 grid of real elevation and land cover (100 m cells) replaces the schematic terrain:

* River cities. Permanent water from the land-cover map is the river. Every land cell gets its height
  above the nearest river cell. A cell floods when the flood stage, reduced by 0.35 m per km of distance
  from the river, is above that height and the cell is connected to the river through flooded cells.
* Urban cities. Rain above the drainage capacity ponds on built-up ground, deeper where the ground is
  lower than its surroundings.

Zones for the planning model come from the land cover: built-up ground and cropland on each side of the
river, the river frontage near the gauge or barrage, and for urban cities built-up ground grouped by how
low it lies and how far it is from the centre. Asset values and population densities per cell remain
illustrative, so damage figures are indicative only. This is a simple fill model, not a hydraulic one.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

from . import geodata
from .flood import PEOPLE_DEPTH_THRESHOLD_M, FloodResult, Terrain, damage_fraction, river_stage
from .geodata import LC
from .preset import Preset, load_preset

CELL_KM = 2 * geodata.HALF_KM / geodata.N
STAGE_DECAY_M_PER_KM = 0.35
MODEL_VERSION = "real-v2-100m"
CELL_SCALE = (CELL_KM / 0.3) ** 2          # zone values below are quoted per 300 m x 300 m of ground

# zone id -> (display name, land use for the damage curve, asset PKR m and people per 300 m x 300 m). Illustrative.
RIVERINE_ZONES = {
    "town":      ("Built-up area, city side", "residential", 30, 200),
    "village_s": ("Built-up area, far bank", "residential", 12, 150),
    "headworks": ("River frontage at the barrage or crossing", "infrastructure", 60, 5),
    "village_n": ("Farmland and villages, city side", "agriculture", 4, 25),
    "farmland":  ("Farmland, far bank", "agriculture", 3, 10),
}
URBAN_ZONES = {
    "informal":    ("Lowest-lying neighbourhoods", "residential", 5, 600),
    "underpass":   ("Low-lying roads and underpasses", "infrastructure", 20, 30),
    "oldcity":     ("City centre", "infrastructure", 40, 150),
    "residential": ("Inner neighbourhoods", "residential", 15, 350),
    "industrial":  ("Outer districts", "infrastructure", 30, 40),
}


def _sides(water: np.ndarray, built: np.ndarray) -> np.ndarray:
    """True for land cells on the same side of the river as most of the built-up ground."""
    n = water.shape[0]
    rr, cc = np.mgrid[0:n, 0:n]
    pts = np.column_stack([rr[water], cc[water]]).astype(float)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    normal = vt[1] if len(vt) > 1 else np.array([0.0, 1.0])            # across-river direction
    side = ((rr - centre[0]) * normal[0] + (cc - centre[1]) * normal[1]) >= 0
    return side if built[side].sum() >= built[~side].sum() else ~side


def build_site(preset: Preset, city_id: str, kind: str) -> tuple[Preset, Terrain, dict]:
    """Return a copy of the preset with real zones, the real terrain, and site metadata."""
    site = geodata.load_site(city_id)
    n = site["n"]
    raw = ndimage.median_filter(np.array(site["elevation_m"], dtype=float), size=5)             # remove pixel-size voids
    elev = ndimage.gaussian_filter(raw, 0.2 / CELL_KM)                                          # smooth over about 200 m
    lc = np.array(site["landcover"])
    builtf = np.array(site["built_fraction"])
    waterf = np.array(site["water_fraction"])
    water = waterf >= 0.5
    built = (builtf >= 0.5) & ~water
    rr, cc = np.mgrid[0:n, 0:n]
    dist_centre_km = np.hypot(rr - (n - 1) / 2, cc - (n - 1) / 2) * CELL_KM
    masks: dict[str, np.ndarray] = {}

    if kind == "riverine":
        if water.mean() < 0.005:                    # river not visible in the land-cover grid: use the lowest ground
            water = elev <= np.quantile(elev, 0.04)
            built &= ~water
        city_side = _sides(water, built)
        dist_river = ndimage.distance_transform_edt(~water) * CELL_KM
        frontage = ~water & (dist_river <= 0.45) & (dist_centre_km <= 2.5)
        land = ~water & ~frontage
        masks["headworks"] = frontage
        masks["town"] = land & built & city_side
        masks["village_s"] = land & built & ~city_side
        masks["village_n"] = land & ~built & city_side
        masks["farmland"] = land & ~built & ~city_side
        defs = RIVERINE_ZONES
    else:
        water = water | (lc == LC["water"])
        built &= ~water
        rank = np.zeros((n, n))
        if built.any():
            order = elev[built].argsort().argsort() / max(1, built.sum() - 1)        # 0 = lowest built-up cell
            rank[built] = order
        masks["informal"] = built & (rank < 0.15)
        masks["underpass"] = built & (rank >= 0.15) & (rank < 0.25)
        rest = built & (rank >= 0.25)
        masks["oldcity"] = rest & (dist_centre_km <= 2.5)
        masks["residential"] = rest & (dist_centre_km > 2.5) & (dist_centre_km <= 5.0)
        masks["industrial"] = rest & (dist_centre_km > 5.0)
        defs = URBAN_ZONES

    zones = []
    for z in preset.zones:                           # keep the preset's zone order and drainage parameters
        name, landuse, asset, people = defs[z["id"]]
        zones.append(dict(z, name=name, landuse=landuse, asset_pkr_m_per_cell=asset * CELL_SCALE, people_per_cell=people * CELL_SCALE))
    terrain = Terrain(elev=elev, channel=water, bed=np.zeros(n), river_col=np.zeros(n), zone_masks={z["id"]: masks[z["id"]] for z in zones})
    meta = {"bbox": site["bbox"], "n": n, "sources": site["sources"], "kind": kind,
            "built_share": float(built.mean()), "water_share": float(water.mean())}
    return replace(preset, zones=zones), terrain, meta


def run_flood_real(preset: Preset, terrain: Terrain, scenario: dict, kind: str) -> FloodResult:
    elev, water = terrain.elev, terrain.channel
    n = elev.shape[0]
    rain = scenario["rain_72h_mm"]
    depth = np.zeros((n, n))
    stage = 0.0
    if kind == "riverine":
        stage = river_stage(preset, scenario["peak_inflow_cusecs"])
        bank = float(preset.raw["terrain"]["channel_depth_m"])                  # stage needed before the river leaves its banks
        dist, (ir, ic) = ndimage.distance_transform_edt(~water, return_indices=True)
        hand = np.clip(elev - elev[ir, ic], 0.0, None)                          # height above the nearest river cell
        level = (stage - bank) - STAGE_DECAY_M_PER_KM * dist * CELL_KM
        wet = ((level - hand) > 0) | water
        labels, _ = ndimage.label(wet)
        ids = np.unique(labels[water])
        connected = np.isin(labels, ids[ids > 0])
        depth = np.where(connected & ~water, np.clip(level - hand, 0.0, None), 0.0)
    # Rain ponding: deeper where the ground is lower than its surroundings.
    rel = elev - ndimage.uniform_filter(elev, size=int(round(2.1 / CELL_KM)), mode="nearest")
    lowness = np.clip(1.0 - rel / 1.5, 0.0, 2.5)          # no ponding on ground well above its surroundings
    for z in preset.zones:
        excess_m = max(0.0, rain - z["drain_capacity_mm"]) / 1000.0
        depth = depth + terrain.zone_masks[z["id"]] * excess_m * z["pluvial_factor"] * lowness
    depth = np.where(water, 0.0, depth)

    zd, zp, zm, za, zpop = {}, {}, {}, {}, {}
    for z in preset.zones:
        m = terrain.zone_masks[z["id"]]
        d = depth[m]
        zd[z["id"]] = float((damage_fraction(d, z["landuse"]) * z["asset_pkr_m_per_cell"]).sum())
        zp[z["id"]] = float((d > PEOPLE_DEPTH_THRESHOLD_M).sum() * z["people_per_cell"])
        zm[z["id"]] = float(d.mean()) if m.any() else 0.0
        za[z["id"]] = float(m.sum() * z["asset_pkr_m_per_cell"])
        zpop[z["id"]] = float(m.sum() * z["people_per_cell"])
    return FloodResult(scenario=scenario, depth=depth, stage_m=stage, zone_damage=zd, zone_people=zp, zone_mean_depth=zm,
                       zone_assets=za, zone_population=zpop)


_SITES: dict = {}


def city_model(assessment: dict):
    """(preset with real zones, terrain, flood result, metadata) for one city assessment."""
    cid, kind = assessment["city_id"], assessment["kind"]
    if cid not in _SITES:
        _SITES[cid] = build_site(load_preset(assessment["preset"]), cid, kind)
    preset, terrain, meta = _SITES[cid]
    return preset, terrain, run_flood_real(preset, terrain, assessment["scenario"], kind), meta
