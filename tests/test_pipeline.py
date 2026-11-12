"""Checks for the weather -> threat -> risk (QAE) -> plan simulation chain. Runs offline on saved snapshots."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from baadal import qae, weather
from baadal.classical import brute_force
from baadal.flood import build_terrain, run_flood
from baadal.model import build_problem
from baadal.preset import load_preset
from baadal.simulate import apply_plan
from baadal.threat import ALERT_LEVEL, LEVELS, quantum_risk, scan


@pytest.fixture(scope="module")
def ref():
    return weather.load_reference()


def test_reference_statistics_are_ordered(ref):
    assert len(ref["cities"]) == sum(c["kind"] == "riverine" for c in weather.CITIES)
    for r in ref["cities"].values():
        assert 0 < r["q2_m3s"] <= r["q5_m3s"] <= r["q10_m3s"] <= r["qmax_m3s"]
        assert len(r["annual_max_m3s"]) >= 20


def test_every_replay_snapshot_is_complete(ref):
    events = weather.available_replays()
    assert events, "run: python -m baadal.weather --build"
    for e in events:
        snap = weather.load_replay(e["id"])
        assert set(snap["cities"]) == {c["id"] for c in weather.CITIES}
        for cid, d in snap["cities"].items():
            assert len(d["rain_daily_mm"]) == weather.WINDOW_DAYS and len(d["rain_members_72h_mm"]) >= 9
            if weather.CITY_BY_ID[cid]["kind"] == "riverine":
                assert len(d["discharge_daily_m3s"]) == weather.WINDOW_DAYS


def test_2010_superflood_is_detected(ref):
    a = {x["city_id"]: x for x in scan(weather.load_replay("2010-07-28_kp_superflood"), ref)}
    assert a["sukkur"]["level"] == "exceptional" and a["sukkur"]["threat"]
    assert a["nowshera"]["threat"]
    assert not a["karachi"]["threat"]            # no heavy rain in Karachi that week


def test_threat_level_is_monotone_in_the_driver(ref):
    snap = weather.load_replay("2010-07-28_kp_superflood")
    base = snap["cities"]["sukkur"]
    levels = []
    for factor in (0.2, 0.5, 0.7, 0.9, 1.2):
        d = dict(base, discharge_daily_m3s=[v * factor for v in base["discharge_daily_m3s"]],
                 discharge_members_peak_m3s=[v * factor for v in base["discharge_members_peak_m3s"]])
        from baadal.threat import assess_city
        a = assess_city("sukkur", d, ref, snap)
        levels.append(LEVELS.index(a["level"]))
        assert 150_000 <= a["scenario"]["peak_inflow_cusecs"] <= 1_000_000
    assert levels == sorted(levels) and levels[0] < levels[-1]


@pytest.mark.parametrize("k", range(0, 17))
def test_comparator_marks_exactly_the_bins_at_or_above_threshold(k):
    n = 4
    for i in range(16):
        qc = QuantumCircuit(n + 1)
        for j in range(n):
            if (i >> j) & 1:
                qc.x(j)
        qc.compose(qae.mark_geq(n, k), inplace=True)
        assert Statevector(qc).probabilities([n])[1] == pytest.approx(1.0 if i >= k else 0.0)


def test_grover_powers_follow_sine_law():
    rng = np.random.default_rng(3)
    d = qae.discretise(rng.gamma(2.0, 45.0, size=31), 90.0)
    assert d.probs.sum() == pytest.approx(1.0)
    theta = np.arcsin(np.sqrt(d.exact))
    for k in (0, 1, 2, 4):
        p1 = Statevector(qae.qae_circuit(d, k, measure=False)).probabilities([qae.N_STATE_QUBITS])[1]
        assert p1 == pytest.approx(np.sin((2 * k + 1) * theta) ** 2, abs=1e-9)


def test_discretised_risk_equals_empirical_ensemble_fraction():
    rng = np.random.default_rng(5)
    for thr in (20.0, 90.0, 150.0, 400.0):
        x = rng.gamma(2.0, 45.0, size=50)
        assert qae.discretise(x, thr).exact == pytest.approx(float((x >= thr).mean()))


def test_qae_estimate_is_close_to_exact_on_real_ensemble(ref):
    a = next(x for x in scan(weather.load_replay("2022-07-23_karachi"), ref) if x["city_id"] == "karachi")
    errs = [quantum_risk(a, seed=s)["abs_error"] for s in range(5)]
    r = quantum_risk(a)
    assert 0.05 < r["exact"] < 0.95                      # a genuinely uncertain case
    assert r["exact"] == pytest.approx(a["p_alert_exact"])
    assert np.mean(errs) < 0.03


def test_simulated_plan_reproduces_the_planning_model(ref):
    a = next(x for x in scan(weather.load_replay("2010-07-28_kp_superflood"), ref) if x["city_id"] == "sukkur")
    preset = load_preset(a["preset"])
    terrain = build_terrain(preset)
    flood = run_flood(preset, terrain, a["scenario"])
    problem = build_problem(preset, flood)
    x = brute_force(problem, 220)["x"]
    sim = apply_plan(preset, terrain, flood, problem, x)
    assert sim["damage_after"] < sim["damage_before"]
    from baadal.flood import damage_fraction
    for z in preset.zones:                                # the after-map carries the model's remaining damage
        m = terrain.zone_masks[z["id"]]
        dmg = float((damage_fraction(sim["depth_after"][m], z["landuse"]) * z["asset_pkr_m_per_cell"]).sum())
        assert dmg == pytest.approx(sim["eval"]["remaining_damage"][z["id"]], abs=0.5)
    none = apply_plan(preset, terrain, flood, problem, [0] * problem.n)
    assert np.allclose(none["depth_after"], flood.depth, atol=1e-3)
    assert ALERT_LEVEL in a["thresholds"]


# ----------------------------------------------------------------------------- real-terrain model
def test_every_city_has_real_terrain():
    from baadal import geodata
    for c in weather.CITIES:
        site = geodata.load_site(c["id"])
        s, w, n, e = site["bbox"]
        assert s < c["lat"] < n and w < c["lon"] < e
        assert np.array(site["elevation_m"]).shape == (geodata.N, geodata.N)
        assert np.array(site["landcover"]).shape == (geodata.N, geodata.N)


def test_real_zones_are_disjoint_and_dry_land(ref):
    from baadal.realmodel import city_model
    for snap_id, cid in [("2010-07-28_kp_superflood", "sukkur"), ("2022-07-23_karachi", "karachi")]:
        a = next(x for x in scan(weather.load_replay(snap_id), ref) if x["city_id"] == cid)
        preset, terrain, flood, meta = city_model(a)
        total = sum(m.astype(int) for m in terrain.zone_masks.values())
        assert total.max() <= 1                                   # no cell belongs to two zones
        assert not (total.astype(bool) & terrain.channel).any()    # permanent water is never a damage zone
        assert (flood.depth[terrain.channel] == 0).all()
        assert flood.total_damage > 0 and meta["bbox"][0] < meta["bbox"][2]


def test_real_flood_grows_with_the_river(ref):
    from baadal.realmodel import build_site, run_flood_real
    preset, terrain, _ = build_site(load_preset("riverine_punjab"), "sukkur", "riverine")
    wet, dmg = [], []
    for q in (300_000, 500_000, 700_000, 950_000):
        f = run_flood_real(preset, terrain, {"id": "t", "name": "t", "peak_inflow_cusecs": q, "rain_72h_mm": 0.0}, "riverine")
        wet.append(int((f.depth > 0.05).sum()))
        dmg.append(f.total_damage)
    assert wet == sorted(wet) and dmg == sorted(dmg) and wet[0] < wet[-1]
    assert wet[0] == 0                                             # a medium flood stays within the banks


def test_real_plan_is_feasible_and_simulation_matches(ref):
    from baadal import planner
    from baadal.flood import damage_fraction
    a = next(x for x in scan(weather.load_replay("2010-07-28_kp_superflood"), ref) if x["city_id"] == "sukkur")
    preset, terrain, flood, problem = planner.problem_for(a)
    ex = brute_force(problem, 220)
    assert problem.feasible(ex["x"], 220)
    sim = apply_plan(preset, terrain, flood, problem, ex["x"])
    assert sim["damage_after"] < sim["damage_before"]
    for z in preset.zones:
        m = terrain.zone_masks[z["id"]]
        dmg = float((damage_fraction(sim["depth_after"][m], z["landuse"]) * z["asset_pkr_m_per_cell"]).sum())
        assert dmg == pytest.approx(sim["eval"]["remaining_damage"][z["id"]], rel=0.02, abs=1.0)
