"""Classical reference solvers: exact brute force and a cost-effectiveness greedy heuristic."""

from __future__ import annotations

import numpy as np

from .model import PlanningProblem, all_bitstrings


def brute_force(problem: PlanningProblem, budget_pkr_m: float) -> dict:
    """Exact optimum of the quadratic model by enumerating all 2^n plans (n ~ 10 -> 1024 plans)."""
    X = all_bitstrings(problem.n)
    ok = problem.feasible_all(X, budget_pkr_m)
    scores = problem.quad_score_all(X)
    # Ties (e.g. adding a zero-benefit generator when money is left over) are broken towards the cheaper plan.
    tiebreak = 1e-7 * (X @ problem.costs_pkr_m)
    scores_masked = np.where(ok, scores - tiebreak, -np.inf)
    k = int(np.argmax(scores_masked))
    # Also report the optimum of the full multiplicative model, to measure truncation error.
    true_scores = np.array([problem.true_eval(x)["score"] if f else -np.inf for x, f in zip(X, ok)])
    kt = int(np.argmax(true_scores - tiebreak))
    return {
        "x": X[k].tolist(),
        "chosen": problem.describe(X[k]),
        "quad_score": float(scores[k]),
        "true": problem.true_eval(X[k]),
        "true_model_optimum_x": X[kt].tolist(),
        "true_model_optimum_score": float(true_scores[kt]),
        "same_plan_as_true_model": bool(k == kt),
        "n_feasible": int(ok.sum()),
        "n_total": int(len(X)),
    }


def greedy(problem: PlanningProblem, budget_pkr_m: float) -> dict:
    """Add the measure with the best marginal score per rupee until nothing feasible improves the plan.

    A dependency (e.g. BREACH requires EVAC_S) is handled by also considering the pair together.
    """
    x = np.zeros(problem.n, dtype=int)
    while True:
        base = problem.quad_score(x)
        best, best_ratio = None, 0.0
        candidates = [[i] for i in range(problem.n) if not x[i]]
        candidates += [[i, j] for i, j in problem.dependencies if not x[i] and not x[j]]
        for cand in candidates:
            y = x.copy()
            y[cand] = 1
            if not problem.feasible(y, budget_pkr_m):
                continue
            gain = problem.quad_score(y) - base
            cost = problem.costs_pkr_m[cand].sum()
            if gain > 1e-9 and gain / cost > best_ratio:
                best, best_ratio = cand, gain / cost
        if best is None:
            break
        x[best] = 1
    return {"x": x.tolist(), "chosen": problem.describe(x), "quad_score": problem.quad_score(x),
            "true": problem.true_eval(x)}


def polish(problem: PlanningProblem, budget_pkr_m: float, x) -> dict:
    """Classical local search: repeatedly apply the best feasible change of one or two measures.

    Used after QAOA sampling (a standard hybrid post-processing step). It checks n + n(n-1)/2 neighbours per
    round, so it scales polynomially. It is a heuristic and does not guarantee the optimum.
    """
    x = np.array(x, dtype=int)
    best = problem.quad_score(x)
    moves = [[i] for i in range(problem.n)] + [[i, j] for i in range(problem.n) for j in range(i + 1, problem.n)]
    rounds = 0
    while True:
        top, top_y = best, None
        for mv in moves:
            y = x.copy()
            y[mv] = 1 - y[mv]
            if problem.feasible(y, budget_pkr_m):
                sc = problem.quad_score(y)
                if sc > top + 1e-9:
                    top, top_y = sc, y
        if top_y is None:
            break
        x, best, rounds = top_y, top, rounds + 1
    return {"x": x.tolist(), "chosen": problem.describe(x), "quad_score": best, "rounds": rounds}
