"""End-to-end case studies on real weather data, and a benchmark of the quantum risk estimator.

For each (event, city) pair the whole chain is run and recorded:
collected data -> threat level -> QAE risk estimate (vs exact) -> flood model on real terrain ->
response plan (exact, greedy, QAOA) -> simulated outcome.

    python -m baadal.case_studies
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import planner, qae, weather
from .classical import brute_force, greedy
from .experiments import strip_private
from .qaoa import simple_noise_model
from .simulate import apply_plan
from .threat import ALERT_LEVEL, LEVEL_LABEL, scan

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BUDGET = 220

CASES = [
    ("2010-07-28_kp_superflood", "nowshera"),
    ("2010-07-28_kp_superflood", "sukkur"),
    ("2014-09-04_chenab", "trimmu"),
    ("2014-09-04_chenab", "rawalpindi"),
    ("2022-07-23_karachi", "karachi"),
    ("2022-08-17_sindh_rains", "larkana"),
    ("2025-08-25_punjab", "marala"),
    ("2025-08-25_punjab", "lahore"),
]


def run_case(snapshot: dict, assessment: dict, budget: float = BUDGET, log=print) -> dict:
    a = assessment
    risk = planner.risk(a["members"], a["thresholds"][ALERT_LEVEL])
    preset, terrain, flood, prob = planner.problem_for(a)
    out = {
        "event": (snapshot.get("event") or {}).get("name", "live"), "mode": snapshot["mode"], "as_of": snapshot["as_of"],
        "city": a["city"], "kind": a["kind"], "driver": a["driver"], "unit": a["unit"],
        "best": a["best"], "design": a["design"], "alert_threshold": a["thresholds"][ALERT_LEVEL],
        "level": a["level"], "level_label": LEVEL_LABEL[a["level"]], "threat": a["threat"],
        "rain_72h_mm": a["rain_72h_best_mm"], "member_source": a["member_source"], "n_members": len(a["members"]),
        "risk": {k: risk[k] for k in ("estimate", "exact", "abs_error", "queries", "std_error_ideal",
                                      "classical_mc_std_error_same_queries", "qubits", "max_cx", "max_depth")},
        "model_inflow_cusecs": a["scenario"]["peak_inflow_cusecs"],
        "flooded_share_of_area": float((flood.depth > 0.05).mean()),
        "damage_no_action_pkr_m": flood.total_damage, "people_exposed": flood.total_people,
        "budget_pkr_m": budget, "plan": None,
    }
    if prob is not None:
        ex = brute_force(prob, budget)
        gr = greedy(prob, budget)
        q = planner.quantum_plan(a, budget)
        x = q["x"] if q and q["x"] is not None else ex["x"]
        sim = apply_plan(preset, terrain, flood, prob, x)
        score = q["score"] if q else None
        out["plan"] = {
            "exact": ex["chosen"], "exact_score": ex["quad_score"],
            "greedy_matches": bool(abs(gr["quad_score"] - ex["quad_score"]) < 1e-6),
            "qaoa": prob.describe(x), "qaoa_score": score,
            "qaoa_matches": bool(score is not None and abs(score - ex["quad_score"]) < 1e-6),
            "qaoa_ratio": (score / ex["quad_score"]) if score is not None else None,
            "p_opt": q["p_opt"], "p_feasible": q["p_feasible"], "p_find_opt_in_shots": q["p_find"],
            "qaoa_seconds": q["seconds"], "circuit": {"qubits": q["qubits"], "transpiled_cx": q["cx"], "transpiled_depth": q["depth"], "p": q["layers"]},
            "damage_after_pkr_m": sim["damage_after"], "people_after": sim["people_after"],
        }
    p = out["plan"]
    log(f"  {out['city']:18s} {out['level']:11s} risk QAE {risk['estimate']:.3f} / exact {risk['exact']:.3f} | damage {flood.total_damage:6.0f}"
        + (f" -> {p['damage_after_pkr_m']:6.0f} | QAOA {'= exact' if p['qaoa_matches'] else format(p['qaoa_ratio'], '.1%')} "
           f"P(opt) {p['p_opt']:.4f}" if p else " | no plan needed"))
    return out


def run_case_studies(log=print) -> list[dict]:
    ref = weather.load_reference()
    out = []
    for event_id, city_id in CASES:
        try:
            snap = weather.load_replay(event_id)
        except FileNotFoundError:
            log(f"  (replay {event_id} not downloaded, skipped)")
            continue
        a = next(x for x in scan(snap, ref) if x["city_id"] == city_id)
        out.append(run_case(snap, a, log=log))
    live = weather.load_live_cached()
    if live:
        out.append(run_case(live, scan(live, ref)[0], log=log))      # most threatened city today
    return out


def run_qae_benchmark(n_seeds: int = 40, log=print) -> dict:
    """Error vs number of queries: QAE (ideal and noisy simulator) against classical Monte Carlo."""
    ref = weather.load_reference()
    snap = weather.load_replay("2022-07-23_karachi")
    a = next(x for x in scan(snap, ref) if x["city_id"] == "karachi")
    d = qae.discretise(a["members"], a["thresholds"][ALERT_LEVEL])
    noise = simple_noise_model()
    schedules = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 4, 8), (0, 1, 2, 4, 8, 16)]
    rows = []
    for powers in schedules:
        ideal = [qae.estimate(d, powers=powers, seed=s) for s in range(n_seeds)]
        q = ideal[0]["queries"]
        noisy = [qae.estimate(d, powers=powers, seed=s, noise=noise)["abs_error"] for s in range(min(n_seeds, 15))] if max(powers) <= 8 else None
        mc = [abs(qae.classical_monte_carlo(d, q, seed=s) - d.exact) for s in range(n_seeds)]
        rows.append({"powers": list(powers), "queries": q, "max_cx": ideal[0]["max_cx"],
                     "qae_mean_abs_error": float(np.mean([r["abs_error"] for r in ideal])),
                     "qae_noisy_mean_abs_error": float(np.mean(noisy)) if noisy else None,
                     "classical_mc_mean_abs_error": float(np.mean(mc))})
        log(f"  powers {str(powers):22s} queries {q:5d} QAE {rows[-1]['qae_mean_abs_error']:.4f} "
            f"noisy {rows[-1]['qae_noisy_mean_abs_error'] if noisy else float('nan'):.4f} classical MC {rows[-1]['classical_mc_mean_abs_error']:.4f} "
            f"(max {rows[-1]['max_cx']} CX)")
    return {"case": "Karachi, July 2022 replay: P(72-hour rain >= 90 mm)", "exact": d.exact, "n_members": len(a["members"]),
            "bins": len(d.probs), "n_seeds": n_seeds, "shots_per_circuit": 64, "rows": rows,
            "noise_model": "depolarising 0.1% (1q), 1% (CX), readout 2%; generic, not a specific device"}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Case studies (real weather data, budget 220 PKR m):")
    cases = run_case_studies()
    print("QAE benchmark:")
    bench = run_qae_benchmark()
    out = {"cases": cases, "qae_benchmark": bench}
    (RESULTS_DIR / "case_studies.json").write_text(json.dumps(strip_private(out), indent=2), encoding="utf-8")
    print(f"-> {RESULTS_DIR / 'case_studies.json'}")


if __name__ == "__main__":
    main()
