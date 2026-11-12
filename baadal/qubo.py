"""QUBO construction and QUBO -> Ising conversion.

QUBO in plain words: a cost function over yes/no variables that contains only
single-variable and pairwise terms, E(y) = sum_i Q_ii y_i + sum_{i<j} Q_ij y_i y_j.
Constraints are turned into penalty terms that make bad plans expensive.

Two encodings of the budget constraint  sum_i c_i x_i <= B  are provided:

* "slack": add binary slack bits s so that  c.x + w.s = B  and penalise
  P * (c.x + w.s - B)^2. Exact, but costs ~log2(B) extra qubits and creates
  all-to-all couplings.
* "unbalanced": penalise  -l1 * h + l2 * h^2  with  h = B - c.x  (no extra
  qubits). This follows "unbalanced penalization" (Montanez-Barrera et al.,
  Quantum Sci. Technol. 9, 025022, 2024). It is a heuristic: the QUBO ground
  state is not guaranteed to be the constrained optimum, so we check it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import PlanningProblem, all_bitstrings


@dataclass
class QUBO:
    Q: np.ndarray            # (N, N) upper triangular, diagonal = linear terms
    offset: float
    n_decision: int          # first n_decision variables are the interventions
    var_names: list[str]
    encoding: str
    budget_pkr_m: float
    penalties: dict

    @property
    def n_vars(self) -> int:
        return self.Q.shape[0]

    def energy(self, y) -> float:
        y = np.asarray(y, dtype=float)
        return float(y @ self.Q @ y + self.offset)

    def energy_all(self) -> np.ndarray:
        """Energy of every basis state, index k <-> bits (k >> i) & 1 (Qiskit order)."""
        Y = all_bitstrings(self.n_vars).astype(float)
        return np.einsum("ki,ij,kj->k", Y, self.Q, Y) + self.offset

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Map y_i = (1 - z_i)/2 so that E = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
        N = self.n_vars
        h = np.zeros(N)
        J = np.zeros((N, N))
        const = self.offset
        for i in range(N):
            q = self.Q[i, i]
            const += q / 2
            h[i] -= q / 2
            for j in range(i + 1, N):
                q = self.Q[i, j]
                if q == 0:
                    continue
                const += q / 4
                h[i] -= q / 4
                h[j] -= q / 4
                J[i, j] += q / 4
        return h, J, const


def slack_weights(B: int) -> list[int]:
    """Bounded binary expansion whose subset sums cover exactly 0..B."""
    if B <= 0:
        return []
    K = int(np.floor(np.log2(B))) + 1
    w = [2 ** k for k in range(K - 1)]
    w.append(B - (2 ** (K - 1) - 1))
    return w


def hard_penalty(problem: PlanningProblem) -> float:
    """A penalty larger than the most any single measure can add to the objective."""
    lin, pair = problem.lin, problem.pair
    sym = pair + pair.T
    gains = np.abs(lin) + np.abs(sym).sum(axis=1)
    return float(1.5 * gains.max())


def build_qubo(problem: PlanningProblem, budget_pkr_m: float, encoding: str = "unbalanced",
               lambdas: tuple[float, float] | None = None) -> QUBO:
    n = problem.n
    B = problem.budget_units(budget_pkr_m)
    c = problem.cost_units.astype(float)
    P = hard_penalty(problem)

    w = slack_weights(B) if encoding == "slack" else []
    N = n + len(w)
    Q = np.zeros((N, N))
    offset = 0.0

    # Objective: maximise score  ->  minimise -score.
    Q[:n, :n] -= np.diag(problem.lin)
    Q[:n, :n] -= np.triu(problem.pair, 1)

    for i, j in problem.exclusions:              # never both
        a, b = min(i, j), max(i, j)
        Q[a, b] += P
    for i, j in problem.dependencies:            # i requires j:  P * x_i (1 - x_j)
        Q[i, i] += P
        a, b = min(i, j), max(i, j)
        Q[a, b] -= P

    penalties = {"hard": P}
    if encoding == "slack":
        a_vec = np.concatenate([c, np.array(w, dtype=float)])
        PB = P
        for i in range(N):
            Q[i, i] += PB * (a_vec[i] ** 2 - 2 * B * a_vec[i])
            for j in range(i + 1, N):
                Q[i, j] += PB * 2 * a_vec[i] * a_vec[j]
        offset += PB * B ** 2
        penalties.update({"budget": PB, "slack_weights": w})
    elif encoding == "unbalanced":
        l1, l2 = lambdas if lambdas is not None else default_lambdas(problem)
        # -l1*(B - c.x) + l2*(B - c.x)^2
        offset += -l1 * B + l2 * B ** 2
        for i in range(n):
            Q[i, i] += l1 * c[i] + l2 * (c[i] ** 2 - 2 * B * c[i])
            for j in range(i + 1, n):
                Q[i, j] += l2 * 2 * c[i] * c[j]
        penalties.update({"lambda1": l1, "lambda2": l2})
    else:
        raise ValueError(encoding)

    names = list(problem.ids) + [f"slack{k}" for k in range(len(w))]
    return QUBO(Q=Q, offset=offset, n_decision=n, var_names=names, encoding=encoding,
                budget_pkr_m=budget_pkr_m, penalties=penalties)


def default_lambdas(problem: PlanningProblem) -> tuple[float, float]:
    """Scale-aware starting point: penalty per cost unit comparable to the best gain per cost unit."""
    ratio = float(np.max(problem.lin / problem.cost_units))
    return 1.0 * ratio, 0.25 * ratio


def qubo_local_minimum(q: QUBO, starts) -> np.ndarray:
    """Best one-flip local minimum of the QUBO energy over a few starting points. No enumeration: cost is polynomial."""
    lin = np.diag(q.Q).copy()
    off = q.Q + q.Q.T - 2 * np.diag(lin)
    best, best_e = None, np.inf
    for y0 in starts:
        y = np.array(y0, dtype=float)
        for _ in range(10 * len(y)):
            delta = (1 - 2 * y) * (lin + off @ y)          # energy change of flipping each bit
            i = int(np.argmin(delta))
            if delta[i] >= -1e-12:
                break
            y[i] = 1 - y[i]
        e = q.energy(y)
        if e < best_e:
            best, best_e = y.astype(int), e
    return best


def tune_unbalanced(problem: PlanningProblem, budgets: list[float]) -> dict:
    """Choose the unbalanced-penalty weights (l1, l2) WITHOUT knowing the optimum and without enumerating plans.

    For each candidate pair, a cheap local search finds a low-energy state of the QUBO. The pair is good for a
    budget when that state is a feasible plan at least as good as the classical greedy plan. We keep the pair
    that is good at the most budgets, preferring smaller penalties (a gentler energy landscape for QAOA).
    Everything here scales polynomially with the number of measures.
    """
    from .classical import greedy

    base = float(np.max(np.abs(problem.lin) / problem.cost_units))
    rng = np.random.default_rng(0)
    n = problem.n
    random_starts = [rng.integers(0, 2, size=n) for _ in range(4)]
    refs = {b: greedy(problem, b) for b in budgets}
    best = None
    for m2 in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]:
        for m1 in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            lam = (m1 * base, m2 * base)
            hits = 0
            for b in budgets:
                q = build_qubo(problem, b, "unbalanced", lam)
                x = qubo_local_minimum(q, [np.zeros(n), np.ones(n), np.array(refs[b]["x"])] + random_starts).tolist()
                hits += int(problem.feasible(x, b) and problem.quad_score(x) >= refs[b]["quad_score"] - 1e-9)
            key = (hits, -(m1 + m2))
            if best is None or key > best[0]:
                best = (key, lam, hits)
    return {"lambdas": best[1], "hits": best[2], "n_budgets": len(budgets), "reference": "greedy plan, local search"}


def ground_state_hits(problem: PlanningProblem, budgets: list[float], lambdas) -> int:
    """Validation only (uses brute force): at how many budgets is the exact QUBO ground state an optimal plan?"""
    from .classical import brute_force

    hits = 0
    for b in budgets:
        k = int(np.argmin(build_qubo(problem, b, "unbalanced", lambdas).energy_all()))
        x = [(k >> i) & 1 for i in range(problem.n)]
        hits += int(problem.feasible(x, b) and abs(problem.quad_score(x) - brute_force(problem, b)["quad_score"]) < 1e-6)
    return hits
