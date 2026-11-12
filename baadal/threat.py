"""Threat detection: turn collected weather / river data into a threat level and a flood scenario.

Riverine cities: the driver is the forecast peak river discharge, judged against that location's own
2000-2023 history (2-, 5- and 10-year floods from GloFAS annual maxima).
Urban cities: the driver is the largest 72-hour rainfall total in the 7-day window.

A threat is declared when the best-estimate forecast reaches the "high" level, or when the ensemble
gives at least a 30% chance of reaching it. Planning then uses a reasonable worst case: the 75th
percentile of the ensemble (never below the best estimate).

The terrain behind the flood map is a synthetic stand-in for each city type, so real discharge is
transferred to it by severity (return period), not by absolute volume.
"""

from __future__ import annotations

import numpy as np

from . import qae
from .weather import CITY_BY_ID, _roll3_max

LEVELS = ["normal", "watch", "high", "very_high", "exceptional"]
LEVEL_LABEL = {"normal": "Normal", "watch": "Watch", "high": "High flood", "very_high": "Very high flood",
               "exceptional": "Exceptionally high flood"}
LEVEL_COLOR = {"normal": "#16a34a", "watch": "#ca8a04", "high": "#ea580c", "very_high": "#dc2626", "exceptional": "#7f1d1d"}
ALERT_LEVEL = "high"
ALERT_PROBABILITY = 0.30
DESIGN_PERCENTILE = 75

# Illustrative 72-hour rainfall thresholds for urban flooding (mm). Not official PMD categories.
URBAN_THRESHOLDS = {"watch": 50.0, "high": 90.0, "very_high": 150.0, "exceptional": 230.0}
# Severity transfer onto the synthetic terrain presets (model inflow in cusecs).
RIVERINE_MODEL_INFLOW = [250_000, 450_000, 620_000, 780_000, 950_000]     # at 0.5*Q2, Q2, Q5, Q10, record
URBAN_RAIN_ANCHORS = [0.0, 90.0, 150.0, 230.0, 350.0]
URBAN_MODEL_INFLOW = [1_500, 6_000, 9_000, 12_000, 15_000]
PRESET_FOR_KIND = {"riverine": "riverine_punjab", "urban": "urban_karachi"}


def _level(value: float, thr: dict) -> str:
    lvl = "normal"
    for name in ("watch", "high", "very_high", "exceptional"):
        if value >= thr[name]:
            lvl = name
    return lvl


def assess_city(city_id: str, data: dict, ref: dict, snapshot_meta: dict) -> dict:
    city = CITY_BY_ID[city_id]
    rain_best = _roll3_max(data["rain_daily_mm"])
    rain_members = np.array(data["rain_members_72h_mm"] or [rain_best], dtype=float)
    rain_design = float(max(rain_best, np.percentile(rain_members, DESIGN_PERCENTILE)))

    if city["kind"] == "riverine":
        r = ref["cities"][city_id]
        thr = {"watch": 0.75 * r["q2_m3s"], "high": r["q2_m3s"], "very_high": r["q5_m3s"], "exceptional": r["q10_m3s"]}
        best = float(max(v for v in data["discharge_daily_m3s"] if v is not None))
        members = np.array(data["discharge_members_peak_m3s"] or [best], dtype=float)
        design = float(max(best, np.percentile(members, DESIGN_PERCENTILE)))
        record = max(r["qmax_m3s"], 1.25 * r["q10_m3s"])
        inflow = float(np.interp(design, [0.5 * r["q2_m3s"], r["q2_m3s"], r["q5_m3s"], r["q10_m3s"], record], RIVERINE_MODEL_INFLOW))
        driver, unit, member_source = "Peak river discharge", "m3/s", data["discharge_member_source"]
        context = f"2-yr flood {r['q2_m3s']:,.0f}, 10-yr flood {r['q10_m3s']:,.0f}, 2000-2023 record {r['qmax_m3s']:,.0f} m3/s"
    else:
        thr = dict(URBAN_THRESHOLDS)
        best, members, design = rain_best, rain_members, rain_design
        inflow = float(np.interp(design, URBAN_RAIN_ANCHORS, URBAN_MODEL_INFLOW))
        driver, unit, member_source = "72-hour rainfall", "mm", data["rain_member_source"]
        context = "illustrative urban thresholds: 90 mm high, 150 mm very high, 230 mm exceptional (72 h)"

    p_alert = float((members >= thr[ALERT_LEVEL]).mean())
    level = _level(best, thr)
    design_level = _level(design, thr)
    threat = LEVELS.index(level) >= LEVELS.index(ALERT_LEVEL) or p_alert >= ALERT_PROBABILITY
    as_of = snapshot_meta["as_of"]
    return {
        "city_id": city_id, "city": city["name"], "kind": city["kind"], "river": city["river"], "province": city["province"],
        "lat": city["lat"], "lon": city["lon"],
        "driver": driver, "unit": unit, "best": best, "design": design, "members": members.tolist(),
        "member_source": member_source, "thresholds": thr, "context": context,
        "level": level, "design_level": design_level, "p_alert_exact": p_alert, "threat": bool(threat),
        "rain_72h_best_mm": rain_best, "rain_72h_design_mm": rain_design,
        "days": data["days"], "rain_daily_mm": data["rain_daily_mm"], "discharge_daily_m3s": data["discharge_daily_m3s"],
        "preset": PRESET_FOR_KIND[city["kind"]],
        "scenario": {"id": f"{city_id}_{as_of}", "name": f"{city['name']}: {LEVEL_LABEL[design_level]} ({snapshot_meta['mode']}, {as_of})",
                     "peak_inflow_cusecs": inflow, "rain_72h_mm": rain_design},
    }


def scan(snapshot: dict, ref: dict) -> list[dict]:
    """Assess every city in a snapshot, most threatened first."""
    out = [assess_city(cid, data, ref, snapshot) for cid, data in snapshot["cities"].items()]
    out.sort(key=lambda a: (LEVELS.index(a["level"]), a["p_alert_exact"], a["best"] / a["thresholds"][ALERT_LEVEL]), reverse=True)
    return out


def quantum_risk(assessment: dict, level: str = ALERT_LEVEL, seed: int = 1234, **kw) -> dict:
    """Estimate P(driver >= threshold of `level`) with quantum amplitude estimation (Qiskit + Aer)."""
    d = qae.discretise(assessment["members"], assessment["thresholds"][level])
    res = qae.estimate(d, seed=seed, **kw)
    res.update({"level": level, "threshold": assessment["thresholds"][level], "bin_probs": d.probs.tolist(),
                "bin_width": d.width, "k_threshold": d.k_threshold, "n_members": len(assessment["members"])})
    return res
