"""Response planning for one city threat, on the real-terrain model, with a JSON disk cache.

The cache (data/cache/) keeps the app responsive: a QAOA run takes 10-30 s the first time and is then
read back instantly. Cached entries are plain results of seeded runs, so deleting the folder only costs time.

    python -m baadal.planner        # pre-compute plans for every city of every saved event
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import weather
from .classical import brute_force, greedy, polish
from .experiments import SETTINGS, best_feasible_from_counts, decision_marginal, prob_metrics
from .model import build_problem
from .qaoa import QAOASolver
from .qubo import build_qubo, tune_unbalanced
from .realmodel import MODEL_VERSION, city_model
from .threat import ALERT_LEVEL, quantum_risk, scan

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
MIN_DAMAGE_TO_PLAN = 1.0      # PKR million
PLAN_VERSION = "p2-polish"
DEFAULT_BUDGET = 220


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


def _cached(folder: str, key: str, compute):
    path = CACHE / folder / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _plan_key(a: dict, budget: float, p_max: int) -> str:
    s = a["scenario"]
    return _key(MODEL_VERSION, PLAN_VERSION, a["city_id"], round(s["peak_inflow_cusecs"], 1), round(s["rain_72h_mm"], 2), float(budget), p_max)


def is_plan_cached(a: dict, budget: float, p_max: int = 4) -> bool:
    return (CACHE / "plans" / f"{_plan_key(a, budget, p_max)}.json").exists()


def problem_for(a: dict):
    """(preset, terrain, flood, problem or None) for a city assessment."""
    preset, terrain, flood, _ = city_model(a)
    prob = build_problem(preset, flood) if flood.total_damage >= MIN_DAMAGE_TO_PLAN else None
    return preset, terrain, flood, prob


def quantum_plan(a: dict, budget: float, p_max: int = 4) -> dict | None:
    """QAOA plan (unbalanced budget encoding). None when there is nothing to plan."""

    def compute():
        preset, _, _, prob = problem_for(a)
        if prob is None or budget <= 0:
            return None
        lam = tune_unbalanced(prob, preset.budgets)["lambdas"]
        solver = QAOASolver(build_qubo(prob, budget, "unbalanced", lam), seed=SETTINGS["seed"], alpha=SETTINGS["cvar_alpha"])
        t0 = time.perf_counter()
        final = solver.optimize(p_max=p_max)[-1]
        counts = solver.sample(final.params, final.p, shots=SETTINGS["shots"])
        picked = best_feasible_from_counts(prob, budget, counts)
        probs = decision_marginal(final.probs, prob.n)
        m = prob_metrics(prob, budget, probs, SETTINGS["shots"])
        circ = solver.circuit_stats(final.params, final.p)
        pol = polish(prob, budget, picked["x"]) if picked["x"] is not None else {"x": None, "quad_score": None}
        return {
            "x": pol["x"], "score": pol["quad_score"], "x_raw": picked["x"], "score_raw": picked["quad_score"],
            "p_opt": m["p_opt"], "p_feasible": m["p_feasible"], "p_find": m["p_find_opt_in_shots"],
            "qubits": circ["qubits"], "cx": circ["transpiled_cx"], "depth": circ["transpiled_depth"], "layers": final.p,
            "shots": SETTINGS["shots"], "seconds": round(time.perf_counter() - t0, 1),
        }

    return _cached("plans", _plan_key(a, budget, p_max), compute)


def classical_plans(a: dict, budget: float) -> dict | None:
    _, _, _, prob = problem_for(a)
    if prob is None:
        return None
    ex, gr = brute_force(prob, budget), greedy(prob, budget)
    return {"exact": {"x": ex["x"], "score": ex["quad_score"]}, "greedy": {"x": gr["x"], "score": gr["quad_score"]}}


def risk(members, threshold: float) -> dict:
    members = [round(float(v), 2) for v in members]
    return _cached("risk", _key(members, round(float(threshold), 2)),
                   lambda: quantum_risk({"members": members, "thresholds": {ALERT_LEVEL: threshold}}))


def warm(budget: float = DEFAULT_BUDGET, log=print) -> None:
    """Pre-compute risk and plans for every city of every saved snapshot at the default budget."""
    ref = weather.load_reference()
    snaps = [weather.load_replay(e["id"]) for e in weather.available_replays()]
    live = weather.load_live_cached()
    if live:
        snaps.append(live)
    for snap in snaps:
        log(f"{(snap.get('event') or {}).get('name', 'live')} ({snap['as_of']})")
        for a in scan(snap, ref):
            risk(a["members"], a["thresholds"][ALERT_LEVEL])
            if is_plan_cached(a, budget):
                continue
            q = quantum_plan(a, budget)
            if q:
                log(f"  {a['city']:20s} planned in {q['seconds']:.0f} s")


if __name__ == "__main__":
    warm()
