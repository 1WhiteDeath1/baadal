"""Run the full Baadal evaluation: classical baselines vs QAOA (two budget encodings) across budgets
and forecast scenarios, plus a noisy-simulator check. Writes results/results.json."""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer

from .classical import brute_force, greedy, polish
from .flood import build_terrain, run_flood
from .model import PlanningProblem, all_bitstrings, build_problem
from .preset import load_preset
from .qaoa import QAOASolver, simple_noise_model
from .qubo import build_qubo, ground_state_hits, tune_unbalanced

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SETTINGS = {
    "calibration_instance": {"scenario": "exceptional", "budget_pkr_m": 220},
    "cvar_alpha": 0.25,
    "p_max": {"unbalanced": 4, "slack": 3},
    "shots": 512,
    "noise_shots": 4096,
    "noise_p": [1, 2],
    "noise_model": {"depolarizing_1q": 1e-3, "depolarizing_2q": 1e-2, "readout": 0.02},
    "seed": 1234,
    "good_plan_threshold": 0.9,
}


def bits_of(k: int, n: int) -> list[int]:
    return [(k >> i) & 1 for i in range(n)]


def index_of(x) -> int:
    return int(sum(int(v) << i for i, v in enumerate(x)))


def optimal_mask(problem: PlanningProblem, budget: float) -> np.ndarray:
    """Boolean mask over the 2^n plans marking every optimal plan (ties included, e.g. a zero-benefit add-on)."""
    X = all_bitstrings(problem.n)
    feas = problem.feasible_all(X, budget)
    score = problem.quad_score_all(X)
    return feas & (score >= score[feas].max() - 1e-6)


def prob_metrics(problem: PlanningProblem, budget: float, probs_dec: np.ndarray, shots: int) -> dict:
    """Metrics from an exact probability distribution over the 2^n decision plans."""
    n = problem.n
    X = all_bitstrings(n)
    feas = problem.feasible_all(X, budget)
    score = problem.quad_score_all(X)
    opt = score[feas].max()
    good = feas & (score >= SETTINGS["good_plan_threshold"] * opt)
    popt = float(probs_dec[optimal_mask(problem, budget)].sum())
    pfeas = float(probs_dec[feas].sum())
    exp_ratio = float((probs_dec[feas] * score[feas]).sum() / max(pfeas, 1e-12) / opt)
    return {
        "p_opt": popt,
        "p_feasible": pfeas,
        "p_good": float(probs_dec[good].sum()),
        "expected_ratio_given_feasible": exp_ratio,
        "p_find_opt_in_shots": 1.0 - (1.0 - popt) ** shots,
        "shots_for_95pct": (math.log(0.05) / math.log(1.0 - popt)) if 0 < popt < 1 else (1 if popt >= 1 else None),
    }


def decision_marginal(probs_full: np.ndarray, n_dec: int) -> np.ndarray:
    idx = np.arange(len(probs_full)) & ((1 << n_dec) - 1)
    return np.bincount(idx, weights=probs_full, minlength=2 ** n_dec)


def best_feasible_from_counts(problem: PlanningProblem, budget: float, counts: dict[int, int]) -> dict:
    """'Sample, then score classically': keep the best feasible plan among measured bitstrings."""
    n = problem.n
    mask = (1 << n) - 1
    agg: dict[int, int] = {}
    for k, c in counts.items():
        agg[k & mask] = agg.get(k & mask, 0) + c
    total = sum(agg.values())
    best = None
    n_feas = 0
    for k, c in agg.items():
        x = bits_of(k, n)
        if problem.feasible(x, budget):
            n_feas += c
            s = problem.quad_score(x)
            if best is None or s > best[0]:
                best = (s, x)
    return {
        "x": best[1] if best else None,
        "chosen": problem.describe(best[1]) if best else [],
        "quad_score": best[0] if best else None,
        "true": problem.true_eval(best[1]) if best else None,
        "sampled_feasible_fraction": n_feas / total,
        "distinct_plans_sampled": len(agg),
    }


def run_qaoa(problem, budget, encoding, lambdas, opt_x, log) -> dict:
    q = build_qubo(problem, budget, encoding, lambdas if encoding == "unbalanced" else None)
    solver = QAOASolver(q, seed=SETTINGS["seed"], alpha=SETTINGS["cvar_alpha"])
    t0 = time.perf_counter()
    res = solver.optimize(p_max=SETTINGS["p_max"][encoding])
    train_s = time.perf_counter() - t0
    shots = SETTINGS["shots"]
    per_p = []
    for r in res:
        m = prob_metrics(problem, budget, decision_marginal(r.probs, problem.n), shots)
        m.update({"p": r.p, "cvar": r.objective, "evals": r.n_evals})
        per_p.append(m)
    final = res[-1]
    counts = solver.sample(final.params, final.p, shots=shots)
    picked = best_feasible_from_counts(problem, budget, counts)
    opt_mask = optimal_mask(problem, budget)
    found = any(opt_mask[k & ((1 << problem.n) - 1)] for k in counts)
    probs_dec = decision_marginal(final.probs, problem.n)
    top = np.argsort(probs_dec)[::-1][:8]
    top_plans = [{"chosen": problem.describe(bits_of(int(k), problem.n)), "prob": float(probs_dec[k]),
                  "feasible": bool(problem.feasible(bits_of(int(k), problem.n), budget)),
                  "quad_score": float(problem.quad_score(bits_of(int(k), problem.n)))} for k in top]
    out = {
        "encoding": encoding,
        "qubits": solver.n_qubits,
        "penalties": {k: (v if not isinstance(v, (np.floating, float)) else float(v)) for k, v in q.penalties.items()},
        "per_p": per_p,
        "final_p": final.p,
        "final_params": final.params,
        "train_seconds": train_s,
        "circuit": solver.circuit_stats(final.params, final.p),
        "sampled": picked,
        "polished": polish(problem, budget, picked["x"]) if picked["x"] is not None else None,
        "sample_contains_optimum": bool(found),
        "top_plans": top_plans,
        "_solver": solver,
        "_res": res,
    }
    log(f"    {encoding:10s} q={solver.n_qubits:2d} p={final.p} Popt={per_p[-1]['p_opt']:.4f} "
        f"Pfeas={per_p[-1]['p_feasible']:.3f} found={found} plan={picked['chosen']} ({train_s:.0f}s)")
    return out


def run_noise(problem, budget, qres, opt_x, log) -> list[dict]:
    solver: QAOASolver = qres["_solver"]
    nm = simple_noise_model(**{"p1": SETTINGS["noise_model"]["depolarizing_1q"],
                               "p2": SETTINGS["noise_model"]["depolarizing_2q"],
                               "readout": SETTINGS["noise_model"]["readout"]})
    rows = []
    opt_mask = optimal_mask(problem, budget)
    mask = (1 << problem.n) - 1
    for p in SETTINGS["noise_p"]:
        r = qres["_res"][p - 1]
        for label, model in [("ideal", None), ("noisy", nm)]:
            counts = solver.sample(r.params, p, shots=SETTINGS["noise_shots"], noise=model)
            tot = sum(counts.values())
            popt = sum(c for k, c in counts.items() if opt_mask[k & mask]) / tot
            pfeas = sum(c for k, c in counts.items() if problem.feasible(bits_of(k & mask, problem.n), budget)) / tot
            rows.append({"p": p, "sim": label, "p_opt_sampled": popt, "p_feasible_sampled": pfeas,
                         "cx": solver.circuit_stats(r.params, p)["transpiled_cx"]})
            log(f"    noise p={p} {label:5s} Popt={popt:.4f} Pfeas={pfeas:.3f}")
    return rows


def random_baseline(problem, budget, shots, seed) -> dict:
    n = problem.n
    X = all_bitstrings(n)
    feas = problem.feasible_all(X, budget)
    rng = np.random.default_rng(seed)
    ks = rng.integers(0, 2 ** n, size=shots)
    counts: dict[int, int] = {}
    for k in ks:
        counts[int(k)] = counts.get(int(k), 0) + 1
    picked = best_feasible_from_counts(problem, budget, counts)
    popt = float(optimal_mask(problem, budget).sum()) / 2 ** n
    return {
        "p_opt": popt,
        "p_feasible": float(feas.mean()),
        "p_find_opt_in_shots": 1.0 - (1.0 - popt) ** shots,
        "shots_for_95pct": math.log(0.05) / math.log(1.0 - popt),
        "sampled": picked,
        "polished": polish(problem, budget, picked["x"]) if picked["x"] is not None else None,
    }


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def run_all(preset_id: str = "riverine_punjab", scenarios=("exceptional", "very_high"), budgets=None,
            with_noise: bool = True, out_name: str = "results.json") -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "run_log.txt"
    log_f = log_path.open("w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    t_start = time.perf_counter()
    preset = load_preset(preset_id)
    terrain = build_terrain(preset)
    budgets = budgets or preset.budgets
    out = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
            "preset": preset_id,
            "settings": SETTINGS,
            "disclaimer": "All damage, cost and population figures are illustrative (synthetic instance).",
        },
        "scenarios": {},
    }
    for sid in scenarios:
        scen = preset.scenario(sid)
        flood = run_flood(preset, terrain, scen)
        problem = build_problem(preset, flood)
        tuned = tune_unbalanced(problem, preset.budgets)  # always calibrate on the full budget grid, so subsets reproduce
        gs_hits = ground_state_hits(problem, preset.budgets, tuned["lambdas"])      # validation only (brute force)
        log(f"Scenario {sid}: stage {flood.stage_m:.2f} m, baseline damage {flood.total_damage:.0f} PKR m, "
            f"people exposed {flood.total_people:.0f}; unbalanced lambdas {tuned['lambdas']} chosen against the greedy plan "
            f"({tuned['hits']}/{tuned['n_budgets']} budgets); exact check: QUBO ground state is optimal at {gs_hits}/{tuned['n_budgets']}")
        sc = {
            "scenario": scen,
            "stage_m": flood.stage_m,
            "zone_damage": flood.zone_damage,
            "zone_people": flood.zone_people,
            "unbalanced_tuning": {"lambdas": list(tuned["lambdas"]), "hits_vs_greedy": tuned["hits"], "n_budgets": tuned["n_budgets"],
                                  "method": "local search against the greedy plan, no enumeration",
                                  "validation_ground_state_optimal": gs_hits},
            "budgets": {},
        }
        for b in budgets:
            log(f"  Budget {b} PKR m")
            bf = brute_force(problem, b)
            gr = greedy(problem, b)
            log(f"    exact      {bf['chosen']} score {bf['quad_score']:.0f}  | greedy {gr['quad_score']:.0f}")
            row = {"budget_pkr_m": b, "classical": bf, "greedy": gr,
                   "random": random_baseline(problem, b, SETTINGS["shots"], SETTINGS["seed"]),
                   "qaoa": {}}
            for enc in ("unbalanced", "slack"):
                row["qaoa"][enc] = run_qaoa(problem, b, enc, tuned["lambdas"], bf["x"], log)
            if with_noise and sid == SETTINGS["calibration_instance"]["scenario"]:
                row["noise"] = run_noise(problem, b, row["qaoa"]["unbalanced"], bf["x"], log)
            sc["budgets"][str(b)] = strip_private(row)
        out["scenarios"][sid] = sc
    out["meta"]["wall_seconds"] = time.perf_counter() - t_start
    (RESULTS_DIR / out_name).write_text(json.dumps(strip_private(out), indent=2), encoding="utf-8")
    log(f"Done in {out['meta']['wall_seconds']:.0f}s -> {RESULTS_DIR / out_name}")
    log_f.close()
    return out
