"""Correctness checks that catch the classic QUBO/QAOA bugs (sign, factor-of-2, bit order)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from baadal.classical import brute_force, greedy, polish
from baadal.flood import build_terrain, run_flood
from baadal.model import all_bitstrings, build_problem
from baadal.preset import load_preset
from baadal.qaoa import QAOASolver, cvar
from baadal.qubo import build_qubo, ground_state_hits, qubo_local_minimum, slack_weights, tune_unbalanced


@pytest.fixture(scope="module", params=["riverine_punjab", "urban_karachi"])
def setup(request):
    preset = load_preset(request.param)
    terrain = build_terrain(preset)
    flood = run_flood(preset, terrain, preset.scenario())
    problem = build_problem(preset, flood)
    return preset, problem


def ising_energy_all(qubo):
    h, J, const = qubo.to_ising()
    Y = all_bitstrings(qubo.n_vars).astype(float)
    Z = 1 - 2 * Y
    return const + Z @ h + np.einsum("ki,ij,kj->k", Z, J, Z)


@pytest.mark.parametrize("encoding", ["slack", "unbalanced"])
def test_ising_matches_qubo(setup, encoding):
    _, problem = setup
    q = build_qubo(problem, 220, encoding)
    assert np.allclose(ising_energy_all(q), q.energy_all())


def test_slack_weights_cover_range():
    for B in range(0, 40):
        w = slack_weights(B)
        sums = {sum(w[i] for i in range(len(w)) if (m >> i) & 1) for m in range(2 ** len(w))}
        assert sums == set(range(B + 1))


def test_feasible_plan_energy_equals_minus_score(setup):
    _, problem = setup
    budget = 220
    q = build_qubo(problem, budget, "slack")
    x = np.array(brute_force(problem, budget)["x"])
    B = problem.budget_units(budget)
    rest = B - int(problem.cost_units @ x)
    w = q.penalties["slack_weights"]
    s = next([(m >> i) & 1 for i in range(len(w))] for m in range(2 ** len(w))
             if sum(w[i] for i in range(len(w)) if (m >> i) & 1) == rest)
    assert np.isclose(q.energy(np.concatenate([x, s])), -problem.quad_score(x))


def test_slack_ground_state_is_constrained_optimum(setup):
    preset, problem = setup
    for b in preset.budgets:
        q = build_qubo(problem, b, "slack")
        k = int(np.argmin(q.energy_all()))
        x = [(k >> i) & 1 for i in range(problem.n)]
        assert problem.feasible(x, b), f"budget {b}"
        assert abs(problem.quad_score(x) - brute_force(problem, b)["quad_score"]) < 1e-6, f"budget {b}"


def test_unbalanced_tuning_reports_hits(setup):
    preset, problem = setup
    t = tune_unbalanced(problem, preset.budgets)
    assert 0 <= t["hits"] <= len(preset.budgets)


def test_quadratic_model_picks_same_plan_as_full_model(setup):
    preset, problem = setup
    mismatches = 0
    for b in preset.budgets:
        bf = brute_force(problem, b)
        mismatches += not bf["same_plan_as_true_model"]
        # even when the plan differs, the quadratic optimum must be near-optimal under the full model
        assert bf["true"]["score"] >= 0.97 * bf["true_model_optimum_score"], f"budget {b}"
        assert bf["true"]["damage_avoided_pkr_m"] <= problem.base_damage.sum() + 1e-6
        assert all(v >= -1e-9 for v in bf["true"]["remaining_damage"].values())
    if preset.id == "riverine_punjab":
        assert mismatches == 0


def test_greedy_never_beats_exact(setup):
    preset, problem = setup
    for b in preset.budgets:
        assert greedy(problem, b)["quad_score"] <= brute_force(problem, b)["quad_score"] + 1e-9


def test_bit_order_round_trip():
    bits = [1, 0, 1, 1, 0]  # qubit i <- bits[i]
    qc = QuantumCircuit(5)
    for i, v in enumerate(bits):
        if v:
            qc.x(i)
    sv_c = qc.copy()
    sv_c.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = np.asarray(sim.run(transpile(sv_c, sim)).result().get_statevector())
    k = int(np.argmax(np.abs(sv)))
    assert [(k >> i) & 1 for i in range(5)] == bits
    qc.measure_all()
    key = next(iter(sim.run(transpile(qc, sim), shots=10).result().get_counts()))
    assert [(int(key, 2) >> i) & 1 for i in range(5)] == bits


def test_cost_layer_phases_match_ising_energy(setup):
    """With beta = 0 the circuit applies exp(-i gamma H_norm): relative phases must equal -gamma * dE."""
    _, problem = setup
    q = build_qubo(problem, 160, "unbalanced")
    solver = QAOASolver(q)
    gamma = 0.07
    _, g, b, tsv = solver._compiled(1)
    sv = solver.sim.run(tsv.assign_parameters({g[0]: gamma, b[0]: 0.0})).result().get_statevector()
    amp = np.asarray(sv)
    Y = all_bitstrings(q.n_vars).astype(float)
    Z = 1 - 2 * Y
    E = Z @ solver.h + np.einsum("ki,ij,kj->k", Z, solver.J, Z)
    rel = amp / amp[0]
    expected = np.exp(-1j * gamma * (E - E[0]))
    assert np.allclose(rel, expected, atol=1e-6)


def test_cvar_alpha_one_is_expectation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=64)
    p = rng.random(64)
    p /= p.sum()
    assert np.isclose(cvar(e, p, 1.0), e @ p)
    assert cvar(e, p, 0.1) <= e @ p


def test_calibration_needs_no_brute_force(setup, monkeypatch):
    """The penalty weights must be chosen without the exact optimum and without enumerating all plans."""
    import baadal.classical as classical
    from baadal.qubo import QUBO
    preset, problem = setup
    monkeypatch.setattr(classical, "brute_force", lambda *a, **k: (_ for _ in ()).throw(AssertionError("brute force used")))
    monkeypatch.setattr(QUBO, "energy_all", lambda self: (_ for _ in ()).throw(AssertionError("enumeration used")))
    t = tune_unbalanced(problem, preset.budgets)
    assert t["hits"] >= len(preset.budgets) - 1 and t["lambdas"][0] > 0 and t["lambdas"][1] > 0


def test_calibrated_qubo_ground_state_is_usually_optimal(setup):
    preset, problem = setup
    t = tune_unbalanced(problem, preset.budgets)
    assert ground_state_hits(problem, preset.budgets, t["lambdas"]) >= len(preset.budgets) - 1


def test_qubo_local_minimum_is_a_local_minimum(setup):
    _, problem = setup
    q = build_qubo(problem, 220, "unbalanced")
    y = qubo_local_minimum(q, [np.zeros(problem.n), np.ones(problem.n)])
    e = q.energy(y)
    for i in range(problem.n):
        z = y.copy()
        z[i] = 1 - z[i]
        assert q.energy(z) >= e - 1e-9


def test_polish_keeps_feasibility_and_never_gets_worse(setup):
    preset, problem = setup
    rng = np.random.default_rng(0)
    for b in preset.budgets:
        best = brute_force(problem, b)["quad_score"]
        for _ in range(10):
            x = rng.integers(0, 2, size=problem.n)
            if not problem.feasible(x, b):
                continue
            out = polish(problem, b, x)
            assert problem.feasible(out["x"], b)
            assert problem.quad_score(x) - 1e-9 <= out["quad_score"] <= best + 1e-9
        assert polish(problem, b, brute_force(problem, b)["x"])["quad_score"] == pytest.approx(best)
