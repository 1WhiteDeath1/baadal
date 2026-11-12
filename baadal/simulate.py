"""Simulate a response plan: the flood map and damage with the chosen measures applied.

The optimisation model describes each measure by how much damage it removes (or adds) per zone.
To show a plan on the map, the remaining damage of each zone under the plan is converted back into
water depth: every cell of the zone is shifted by one uniform depth change, chosen so that the
depth-damage curves reproduce the plan's remaining damage. The "after" map is therefore consistent
with the planning model, but it is not a new hydraulic simulation.
"""

from __future__ import annotations

import numpy as np

from .flood import FloodResult, Terrain, damage_fraction
from .model import PlanningProblem
from .preset import Preset


def _zone_damage(depths: np.ndarray, shift: float, landuse: str, asset: float) -> float:
    return float((damage_fraction(np.clip(depths + shift, 0.0, None), landuse) * asset).sum())


def apply_plan(preset: Preset, terrain: Terrain, flood: FloodResult, problem: PlanningProblem, x) -> dict:
    ev = problem.true_eval(x)
    depth_after = flood.depth.copy()
    shifts = {}
    for z in preset.zones:
        zid = z["id"]
        mask = terrain.zone_masks[zid]
        d = flood.depth[mask]
        target = ev["remaining_damage"][zid]
        lo, hi = -float(d.max()) - 0.01, 6.0
        if target <= 1e-9:
            shift = lo
        elif target >= _zone_damage(d, hi, z["landuse"], z["asset_pkr_m_per_cell"]):
            shift = hi
        else:
            for _ in range(50):                      # bisection: damage grows with depth
                mid = 0.5 * (lo + hi)
                if _zone_damage(d, mid, z["landuse"], z["asset_pkr_m_per_cell"]) < target:
                    lo = mid
                else:
                    hi = mid
            shift = 0.5 * (lo + hi)
        shifts[zid] = shift
        depth_after[mask] = np.clip(d + shift, 0.0, None)
    depth_after = np.where(terrain.channel, 0.0, depth_after)
    return {
        "depth_after": depth_after,
        "zone_depth_change_m": shifts,
        "eval": ev,
        "damage_before": flood.total_damage,
        "damage_after": float(sum(ev["remaining_damage"].values())),
        "people_before": flood.total_people,
        "people_after": float(sum(ev["remaining_people"].values())),
        "chosen": problem.describe(x),
    }
