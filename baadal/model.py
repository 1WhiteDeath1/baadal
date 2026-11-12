"""Optimisation model: choose interventions to maximise damage avoided within a budget.

Benefit model
-------------
Each intervention i cuts the remaining damage in zone z by a fraction r_iz
(negative r means it ADDS damage, e.g. a controlled breach floods farmland).
Effects combine multiplicatively, so two measures on the same zone overlap:

Some actions also CREATE damage A_iz elsewhere (a controlled breach floods dry
farmland), which other measures can then reduce (evacuation protects the people
put at risk by the breach):

    remaining_z(x) = (D_z + sum_i A_iz x_i) * prod_i (1 - r_iz x_i) * prod_syn (1 - s x_a x_b)

Expanding to second order gives a quadratic (QUBO-ready) objective:

    avoided_z(x) ~= sum_i [D_z r_iz - A_iz (1 - r_iz)] x_i
                  + sum_{i<j} [-D_z r_iz r_jz            (diminishing returns)
                               + A_iz r_jz + A_jz r_iz]  (one action mitigates another's harm)
                               x_i x_j
                  + sum_syn D_z s x_a x_b                (synergies)

The pairwise terms are exact when at most two measures touch a zone; three or
more measures on one zone introduce a small truncation error, which we measure
against the full multiplicative ("true") model.

The same structure is used for people exposed. The scalar score is
    score = damage avoided (PKR m) + w * people protected
where w (PKR m per person) is a policy weight from the preset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flood import FloodResult
from .preset import Preset


@dataclass
class PlanningProblem:
    preset_id: str
    ids: list[str]
    names: list[str]
    costs_pkr_m: np.ndarray
    cost_units: np.ndarray          # integer costs in units of cost_unit
    cost_unit: float
    people_weight: float
    zone_ids: list[str]
    base_damage: np.ndarray         # (Z,)
    base_people: np.ndarray         # (Z,)
    R: np.ndarray                   # (n, Z) damage reduction fractions
    Q: np.ndarray                   # (n, Z) people reduction fractions
    A: np.ndarray                   # (n, Z) damage ADDED, PKR million
    Ap: np.ndarray                  # (n, Z) people ADDED to exposure
    synergies: list[tuple[int, int, int, float, float]]  # (i, j, zone, damage_extra, people_extra)
    exclusions: list[tuple[int, int]]
    dependencies: list[tuple[int, int]]  # (i, j): choosing i requires j
    lin_damage: np.ndarray = field(init=False)
    pair_damage: np.ndarray = field(init=False)
    lin_people: np.ndarray = field(init=False)
    pair_people: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.ids)
        D, N = self.base_damage, self.base_people
        R, Q, A, Ap = self.R, self.Q, self.A, self.Ap
        self.lin_damage = R @ D - np.sum(A * (1.0 - R), axis=1)
        self.lin_people = Q @ N - np.sum(Ap * (1.0 - Q), axis=1)
        pd = np.zeros((n, n))
        pp = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                pd[i, j] = np.sum(-R[i] * R[j] * D + A[i] * R[j] + A[j] * R[i])
                pp[i, j] = np.sum(-Q[i] * Q[j] * N + Ap[i] * Q[j] + Ap[j] * Q[i])
        for i, j, z, sd, sp in self.synergies:
            a, b = min(i, j), max(i, j)
            pd[a, b] += sd * D[z]
            pp[a, b] += sp * N[z]
        self.pair_damage, self.pair_people = pd, pp

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def lin(self) -> np.ndarray:
        return self.lin_damage + self.people_weight * self.lin_people

    @property
    def pair(self) -> np.ndarray:
        return self.pair_damage + self.people_weight * self.pair_people

    # ---- evaluation -------------------------------------------------------
    def quad_score(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(self.lin @ x + x @ self.pair @ x)

    def quad_score_all(self, X: np.ndarray) -> np.ndarray:
        """Vectorised quadratic score for a (m, n) 0/1 matrix."""
        X = X.astype(float)
        return X @ self.lin + np.einsum("ki,ij,kj->k", X, self.pair, X)

    def true_eval(self, x) -> dict:
        """Full multiplicative model (no second-order truncation)."""
        x = np.asarray(x, dtype=float)
        fac_d = np.prod(1.0 - self.R * x[:, None], axis=0)
        fac_p = np.prod(1.0 - self.Q * x[:, None], axis=0)
        for i, j, z, sd, sp in self.synergies:
            fac_d[z] *= 1.0 - sd * x[i] * x[j]
            fac_p[z] *= 1.0 - sp * x[i] * x[j]
        rem_d = (self.base_damage + self.A.T @ x) * fac_d
        rem_p = (self.base_people + self.Ap.T @ x) * fac_p
        dmg_avoided = float(self.base_damage.sum() - rem_d.sum())
        ppl = float(self.base_people.sum() - rem_p.sum())
        return {
            "damage_avoided_pkr_m": dmg_avoided,
            "people_protected": ppl,
            "score": dmg_avoided + self.people_weight * ppl,
            "remaining_damage": dict(zip(self.zone_ids, rem_d.tolist())),
            "remaining_people": dict(zip(self.zone_ids, rem_p.tolist())),
            "cost_pkr_m": float(self.costs_pkr_m @ x),
        }

    # ---- constraints ------------------------------------------------------
    def budget_units(self, budget_pkr_m: float) -> int:
        return int(np.floor(budget_pkr_m / self.cost_unit + 1e-9))

    def feasible(self, x, budget_pkr_m: float) -> bool:
        x = np.asarray(x, dtype=int)
        if self.cost_units @ x > self.budget_units(budget_pkr_m):
            return False
        if any(x[i] and x[j] for i, j in self.exclusions):
            return False
        if any(x[i] and not x[j] for i, j in self.dependencies):
            return False
        return True

    def feasible_all(self, X: np.ndarray, budget_pkr_m: float) -> np.ndarray:
        X = X.astype(int)
        ok = X @ self.cost_units <= self.budget_units(budget_pkr_m)
        for i, j in self.exclusions:
            ok &= ~((X[:, i] == 1) & (X[:, j] == 1))
        for i, j in self.dependencies:
            ok &= ~((X[:, i] == 1) & (X[:, j] == 0))
        return ok

    def describe(self, x) -> list[str]:
        return [self.ids[i] for i in range(self.n) if int(x[i]) == 1]


def build_problem(preset: Preset, flood: FloodResult, people_weight: float | None = None) -> PlanningProblem:
    zids = preset.zone_ids
    zidx = {z: k for k, z in enumerate(zids)}
    n, Z = len(preset.interventions), len(zids)
    R = np.zeros((n, Z))
    Q = np.zeros((n, Z))
    A = np.zeros((n, Z))
    Ap = np.zeros((n, Z))
    for i, iv in enumerate(preset.interventions):
        for z, v in iv.damage_reduction.items():
            R[i, zidx[z]] = v
        for z, v in iv.people_reduction.items():
            Q[i, zidx[z]] = v
        for z, v in iv.damage_added.items():
            A[i, zidx[z]] = v * flood.zone_assets[z]
        for z, v in iv.people_added.items():
            Ap[i, zidx[z]] = v * flood.zone_population[z]
    syn = [(preset.index(s["pair"][0]), preset.index(s["pair"][1]), zidx[s["zone"]],
            float(s.get("damage_extra", 0.0)), float(s.get("people_extra", 0.0))) for s in preset.synergies]
    exc = [(preset.index(e["pair"][0]), preset.index(e["pair"][1])) for e in preset.exclusions]
    dep = [(preset.index(d["if"]), preset.index(d["requires"])) for d in preset.dependencies]
    costs = np.array([iv.cost_pkr_m for iv in preset.interventions])
    unit = preset.cost_unit
    return PlanningProblem(
        preset_id=preset.id,
        ids=preset.intervention_ids,
        names=[iv.name for iv in preset.interventions],
        costs_pkr_m=costs,
        cost_units=np.round(costs / unit).astype(int),
        cost_unit=unit,
        people_weight=preset.people_weight if people_weight is None else people_weight,
        zone_ids=zids,
        base_damage=np.array([flood.zone_damage[z] for z in zids]),
        base_people=np.array([flood.zone_people[z] for z in zids]),
        R=R, Q=Q, A=A, Ap=Ap, synergies=syn, exclusions=exc, dependencies=dep,
    )


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) matrix; row k holds the bits of k with bit i = (k >> i) & 1 (Qiskit little-endian order)."""
    k = np.arange(2 ** n)[:, None]
    return ((k >> np.arange(n)[None, :]) & 1).astype(np.int8)
