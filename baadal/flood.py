"""Classical stage: synthetic terrain, simple flood-depth model and per-zone damage.

This is deliberately simple (a connected "bathtub" fill from the river plus a
local rainfall-ponding term). It is NOT a hydrodynamic model. It exists so that
the optimisation stage is driven by a forecast-dependent damage estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .preset import Preset

# Depth (m) -> damage fraction. Shapes loosely follow the JRC global flood
# depth-damage functions for Asia (Huizinga, de Moel & Szewczyk, 2017).
# Values are approximate and illustrative, not calibrated for Pakistan.
DEPTH_DAMAGE = {
    "residential":    ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
                       [0.0, 0.33, 0.49, 0.62, 0.72, 0.87, 0.93, 0.98, 1.0]),
    "infrastructure": ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
                       [0.0, 0.38, 0.54, 0.66, 0.76, 0.88, 0.94, 0.98, 1.0]),
    "agriculture":    ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
                       [0.0, 0.30, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.0]),
}

PEOPLE_DEPTH_THRESHOLD_M = 0.3  # people counted as "exposed" above this depth


@dataclass
class Terrain:
    elev: np.ndarray          # (rows, cols) ground elevation, m
    channel: np.ndarray       # bool mask of river-channel cells
    bed: np.ndarray           # (rows,) channel-bed elevation per row, m
    river_col: np.ndarray     # (rows,) river centre column per row
    zone_masks: dict[str, np.ndarray]


@dataclass
class FloodResult:
    scenario: dict
    depth: np.ndarray                 # total flood depth, m (0 in channel)
    stage_m: float                    # river stage above bed, m
    zone_damage: dict[str, float]     # PKR million
    zone_people: dict[str, float]     # people exposed (> 0.3 m)
    zone_mean_depth: dict[str, float]
    zone_assets: dict[str, float]     # total asset value in the zone, PKR million
    zone_population: dict[str, float]

    @property
    def total_damage(self) -> float:
        return float(sum(self.zone_damage.values()))

    @property
    def total_people(self) -> float:
        return float(sum(self.zone_people.values()))


def build_terrain(preset: Preset) -> Terrain:
    t = preset.raw["terrain"]
    rows, cols = t["rows"], t["cols"]
    rng = np.random.default_rng(t["seed"])
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]

    river_col = t["river_col"] + t["meander_amp_cells"] * np.sin(2 * np.pi * np.arange(rows) / t["meander_period_cells"])
    bank = t["base_elev_m"] - t["downstream_slope_m_per_cell"] * np.arange(rows)
    bed = bank - t["channel_depth_m"]

    dist = np.abs(c - river_col[:, None])
    noise = ndimage.gaussian_filter(rng.normal(size=(rows, cols)), t["noise_smooth_cells"])
    noise = noise / (np.abs(noise).max() + 1e-12) * t["noise_amp_m"]
    elev = bank[:, None] + t["lateral_slope_m_per_cell"] * dist + noise + 0 * r

    zone_masks = {}
    for z in preset.zones:
        r0, r1, c0, c1 = z["rect"]
        m = np.zeros((rows, cols), dtype=bool)
        m[r0:r1 + 1, c0:c1 + 1] = True
        zone_masks[z["id"]] = m
        elev = elev + m * z.get("mound_m", 0.0)

    channel = dist <= t["channel_halfwidth_cells"] + 0.5
    elev = np.where(channel, bed[:, None], elev)
    for zid in zone_masks:
        zone_masks[zid] &= ~channel
    return Terrain(elev=elev, channel=channel, bed=bed, river_col=river_col, zone_masks=zone_masks)


def river_stage(preset: Preset, inflow_cusecs: float) -> float:
    rc = preset.raw["rating_curve"]
    return float(rc["a_m"] * (inflow_cusecs / rc["q_ref_cusecs"]) ** rc["b"])


def damage_fraction(depth: np.ndarray, landuse: str) -> np.ndarray:
    xs, ys = DEPTH_DAMAGE[landuse]
    return np.interp(depth, xs, ys, left=0.0, right=1.0)


def run_flood(preset: Preset, terrain: Terrain, scenario: dict) -> FloodResult:
    h = river_stage(preset, scenario["peak_inflow_cusecs"])
    wse = (terrain.bed + h)[:, None] * np.ones_like(terrain.elev)

    # Riverine: cells below the local water surface that are hydraulically connected to the channel.
    below = (terrain.elev < wse) | terrain.channel
    labels, _ = ndimage.label(below)
    connected_ids = np.unique(labels[terrain.channel])
    connected = np.isin(labels, connected_ids[connected_ids > 0])
    riverine = np.where(connected & ~terrain.channel, wse - terrain.elev, 0.0)
    riverine = np.clip(riverine, 0.0, None)

    # Pluvial: rainfall above local drainage capacity ponds in each zone (urban areas concentrate it).
    depth = riverine.copy()
    rain = scenario["rain_72h_mm"]
    for z in preset.zones:
        excess_m = max(0.0, rain - z["drain_capacity_mm"]) / 1000.0
        depth = depth + terrain.zone_masks[z["id"]] * excess_m * z["pluvial_factor"]
    depth = np.where(terrain.channel, 0.0, depth)

    zone_damage, zone_people, zone_depth, zone_assets, zone_pop = {}, {}, {}, {}, {}
    for z in preset.zones:
        m = terrain.zone_masks[z["id"]]
        d = depth[m]
        zone_damage[z["id"]] = float((damage_fraction(d, z["landuse"]) * z["asset_pkr_m_per_cell"]).sum())
        zone_people[z["id"]] = float((d > PEOPLE_DEPTH_THRESHOLD_M).sum() * z["people_per_cell"])
        zone_depth[z["id"]] = float(d.mean())
        zone_assets[z["id"]] = float(m.sum() * z["asset_pkr_m_per_cell"])
        zone_pop[z["id"]] = float(m.sum() * z["people_per_cell"])
    return FloodResult(scenario=scenario, depth=depth, stage_m=h, zone_damage=zone_damage,
                       zone_people=zone_people, zone_mean_depth=zone_depth,
                       zone_assets=zone_assets, zone_population=zone_pop)
