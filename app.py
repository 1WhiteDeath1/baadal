"""Baadal application.  Run:  streamlit run app.py

One map-first screen (baadal_ui/index.html) guides the user through five steps:
choose city -> assess threat -> see impact -> plan response -> review plan.
Python does the data collection, modelling and quantum work; the page draws the satellite map,
the flood overlay and the plan, and lets the user build their own plan.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from baadal import planner, weather
from baadal.flood import DEPTH_DAMAGE
from baadal.realmodel import city_model
from baadal.threat import ALERT_LEVEL, LEVEL_LABEL, scan

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Baadal - flood response planner", page_icon="🌧️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important}
.block-container {padding: 0.5rem 0.7rem 2rem !important; max-width: 100% !important}
</style>""", unsafe_allow_html=True)

baadal_map = components.declare_component("baadal_map", path=str(ROOT / "baadal_ui"))


def sample_gain() -> float:
    """How many times fewer samples QAE needed than classical sampling for the same error, from the saved benchmark."""
    try:
        rows = json.loads((ROOT / "results" / "case_studies.json").read_text(encoding="utf-8"))["qae_benchmark"]["rows"]
        row = [r for r in rows if r["powers"][-1] == 8][0]
        return (row["classical_mc_mean_abs_error"] / row["qae_mean_abs_error"]) ** 2
    except (OSError, KeyError, IndexError, ValueError):
        return float("nan")


GAIN = sample_gain()

SHORT = {
    "EMB_TOWN": "Town embankment", "SPURS": "Stone spurs", "BUND_HW": "Barrage bund", "BREACH": "Controlled breach",
    "EMB_RIGHT": "Far-bank dyke", "EVAC_S": "Evacuate far bank", "PUMPS": "Mobile pumps", "GENSETS": "Generators",
    "WETLAND": "Wetland storage", "EWS": "Early warning", "DESILT": "Desilt drains", "ENCROACH": "Clear encroachments",
    "RESETTLE": "Resettle households", "DISPOSAL": "Pumping station", "MOBILE": "Underpass pumps", "TANK": "Storage tank",
    "GREEN": "Rain gardens", "ALERTS": "Flood alerts", "OUTFALL": "Clear outfall",
}


@st.cache_resource
def get_reference():
    return weather.load_reference()


@st.cache_data(show_spinner=False)
def get_scan(key: str, _snapshot: dict):
    return scan(_snapshot, get_reference())


def measure_sites(preset, terrain, bbox) -> list[dict]:
    """For each measure: the zone it mainly affects and a marker position inside that zone.
    Markers show which area a measure affects. They are not construction sites."""
    s, w, n, e = bbox
    rows, cols = terrain.elev.shape
    zidx = {z: k for k, z in enumerate(preset.zone_ids)}
    used: dict[str, int] = {}
    out = []
    for iv in preset.interventions:
        if iv.damage_added:
            zid = max(iv.damage_added, key=iv.damage_added.get)
        elif iv.damage_reduction:
            zid = max(iv.damage_reduction, key=iv.damage_reduction.get)
        elif iv.people_reduction:
            zid = max(iv.people_reduction, key=iv.people_reduction.get)
        else:
            zid = next((x["zone"] for x in preset.synergies if iv.id in x["pair"]), preset.zone_ids[0])
        mask = terrain.zone_masks[zid]
        if not mask.any():
            mask = ~terrain.channel
        rr, cc = np.nonzero(mask)
        k = used.get(zid, 0)
        used[zid] = k + 1
        ring = 0.15 * rows                                   # spread markers that share a zone
        tr = rr.mean() + (ring * math.sin(2.4 * k) if k else 0.0)
        tc = cc.mean() + (ring * math.cos(2.4 * k) if k else 0.0)
        j = int(np.argmin((rr - tr) ** 2 + (cc - tc) ** 2))
        out.append({"zone": zidx[zid], "latlon": [n - (rr[j] + 0.5) / rows * (n - s), w + (cc[j] + 0.5) / cols * (e - w)]})
    return out


def city_payload(a: dict, budget: float) -> dict:
    """Everything the page needs for one city: data, risk, real-terrain flood model and plans."""
    preset, terrain, flood, prob = planner.problem_for(a)
    meta = city_model(a)[3]
    risk = planner.risk(a["members"], a["thresholds"][ALERT_LEVEL])
    rows, cols = terrain.elev.shape
    zone_idx = np.full((rows, cols), -1, dtype=int)
    for k, z in enumerate(preset.zones):
        zone_idx[terrain.zone_masks[z["id"]]] = k
    out = {
        "id": a["city_id"], "name": a["city"], "province": a["province"], "river": a["river"], "kind": a["kind"],
        "level": a["level"], "level_label": LEVEL_LABEL[a["level"]], "best": a["best"], "unit": a["unit"],
        "thresholds": a["thresholds"], "days": a["days"], "rain_daily": [v or 0 for v in a["rain_daily_mm"]],
        "discharge_daily": a["discharge_daily_m3s"],
        "risk": {k: risk[k] for k in ("estimate", "exact", "qubits", "n_members", "queries", "max_cx")},
        "model": {"rows": rows, "cols": cols, "bbox": meta["bbox"],
                  "depth": np.round(flood.depth, 3).ravel().tolist(), "water": terrain.channel.astype(int).ravel().tolist(),
                  "zone_idx": zone_idx.ravel().tolist(),
                  "zones": [{"id": z["id"], "name": z["name"], "landuse": z["landuse"], "asset": z["asset_pkr_m_per_cell"]} for z in preset.zones]},
        "damage_total": flood.total_damage, "people_total": flood.total_people,
        "zone_damage": [flood.zone_damage[z] for z in preset.zone_ids],
        "plan": None,
    }
    if prob is not None:
        cls = planner.classical_plans(a, budget)
        sites = measure_sites(preset, terrain, meta["bbox"])
        out["plan"] = {
            "measures": [{"id": iv.id, "name": iv.name, "short": SHORT.get(iv.id, iv.name), "cost": iv.cost_pkr_m,
                          "description": iv.description, **site} for iv, site in zip(preset.interventions, sites)],
            "R": prob.R.tolist(), "Q": prob.Q.tolist(), "A": prob.A.tolist(), "Ap": prob.Ap.tolist(),
            "synergies": [list(x) for x in prob.synergies], "exclusions": [list(x) for x in prob.exclusions],
            "dependencies": [list(x) for x in prob.dependencies],
            "base_damage": prob.base_damage.tolist(), "base_people": prob.base_people.tolist(),
            "lin": prob.lin.tolist(), "pair": prob.pair.tolist(),
            "curves": {k: [list(v[0]), list(v[1])] for k, v in DEPTH_DAMAGE.items()},
            "exact": cls["exact"], "greedy": cls["greedy"], "quantum": planner.quantum_plan(a, budget),
        }
    return out


# ----------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("mode", "live")
ss.setdefault("event_id", None)
ss.setdefault("city", None)
ss.setdefault("budget", planner.DEFAULT_BUDGET)
ss.setdefault("nonce", None)

events = weather.available_replays()
if ss.mode == "live":
    if "live" not in ss:
        ss.live = weather.load_live_cached()
        if ss.live is None:
            try:
                with st.spinner("Collecting the latest weather from Open-Meteo..."):
                    ss.live = weather.fetch_live()
            except ConnectionError:
                ss.live = None
    snapshot = ss.live
    if snapshot is None:                      # offline and nothing saved: fall back to a replay
        ss.mode, ss.event_id = "replay", events[0]["id"]
        st.rerun()
else:
    if ss.event_id not in {e["id"] for e in events}:
        ss.event_id = events[0]["id"]
    snapshot = weather.load_replay(ss.event_id)

snap_key = f"{snapshot['mode']}|{snapshot['as_of']}|{snapshot['fetched_at']}"
assessments = get_scan(snap_key, snapshot)
by_id = {a["city_id"]: a for a in assessments}
if ss.city not in by_id:
    ss.city = assessments[0]["city_id"]       # most threatened city first
A = by_id[ss.city]
preset = planner.problem_for(A)[0]
budget_max = int(sum(iv.cost_pkr_m for iv in preset.interventions))
ss.budget = int(min(ss.budget, budget_max))

needs_qaoa = planner.problem_for(A)[3] is not None and ss.budget > 0 and not planner.is_plan_cached(A, ss.budget)
if needs_qaoa:
    with st.spinner("Qiskit is optimising the response plan (first time for this city and budget, about 20 seconds)..."):
        city = city_payload(A, ss.budget)
else:
    city = city_payload(A, ss.budget)

payload = {
    "snapshot_key": snap_key, "mode": snapshot["mode"], "event_id": ss.event_id, "as_of": snapshot["as_of"],
    "fetched_at": snapshot["fetched_at"], "events": [{"id": e["id"], "name": e["name"]} for e in events],
    "cities": [{"id": a["city_id"], "name": a["city"], "short": a["city"].split(" (")[0], "river": a["river"], "lat": a["lat"],
                "lon": a["lon"], "level": a["level"], "level_label": LEVEL_LABEL[a["level"]], "threat": a["threat"]} for a in assessments],
    "budget": ss.budget, "budget_max": budget_max, "budget_step": int(preset.cost_unit),
    "city": city,
}
evt = baadal_map(payload=payload, key="baadal_map", default=None)

if evt and evt.get("nonce") != ss.nonce:
    ss.nonce = evt["nonce"]
    if evt["type"] == "city":
        ss.city = evt["id"]
    elif evt["type"] == "budget":
        ss.budget = int(evt["value"])
    elif evt["type"] == "mode":
        ss.mode, ss.city = evt["mode"], None
        if evt["mode"] == "replay":
            ss.event_id = evt.get("event_id") or events[0]["id"]
    elif evt["type"] == "refresh":
        try:
            ss.live = weather.fetch_live()
        except ConnectionError:
            st.toast("Could not reach the weather service. Showing the last saved data.")
        ss.city = None
    st.rerun()

# ----------------------------------------------------------------------------- technical details
with st.expander("Technical details: collected data, quantum methods, benchmarks and limitations"):
    t_data, t_q, t_meas, t_lim = st.tabs(["Collected data", "Quantum methods and benchmarks", "Measures", "Limitations"])
    with t_data:
        st.caption(f"Source: {snapshot['source']} Window starts {snapshot['as_of']}.")
        st.dataframe(pd.DataFrame([{
            "city": a["city"], "type": a["kind"], "river / drainage": a["river"], "driver": f"{a['best']:,.0f} {a['unit']}",
            "% of flood threshold": round(100 * a["best"] / a["thresholds"][ALERT_LEVEL]), "72 h rain (mm)": round(a["rain_72h_best_mm"], 1),
            "P(reach high flood), ensemble": round(a["p_alert_exact"], 2), "level": LEVEL_LABEL[a["level"]],
            "threat": "yes" if a["threat"] else ""} for a in assessments]), hide_index=True, use_container_width=True)
        st.caption("A threat is declared when the forecast reaches 'high flood' (the location's 2-year flood, or 90 mm of rain in 72 h "
                   "for urban cities), or when the ensemble gives at least a 30% chance of it. Ensemble for the selected city: "
                   f"{A['member_source']}.")
    with t_q:
        r = city["risk"]
        st.markdown(f"""
**Risk: quantum amplitude estimation.** The forecast ensemble for {A['city']} ({r['n_members']} members) is loaded into {r['qubits'] - 1} qubits as a
{2 ** (r['qubits'] - 1)}-bin distribution. A comparator circuit flags the bins above the flood threshold on one more qubit, and Grover-style rotations with a
maximum-likelihood fit estimate the exceedance probability: **{r['estimate']:.3f}** (exact {r['exact']:.3f}), using {r['queries']:,}
applications of the loading circuit; the deepest circuit has {r['max_cx']:,} CX gates. In our benchmark on an ideal simulator this
method needed about {GAIN:.1f} times fewer samples than classical Monte Carlo for the same accuracy, the quadratic speed-up theory predicts.

**Response: QAOA.** Each measure is one qubit. Overlaps, synergies, the breach-evacuation rule and the budget become pairwise terms of
a QUBO. QAOA prepares a state that favours good plans; 512 samples are scored classically, the best feasible plan is kept, and a
classical local search then tries single changes and swaps. The penalty weights are chosen against a greedy plan, without enumerating
plans. Exact brute force over all 1,024 plans is used only to check the answer.
""")
        cs_path = ROOT / "results" / "case_studies.json"
        if cs_path.exists():
            cs = json.loads(cs_path.read_text(encoding="utf-8"))
            st.markdown("**End-to-end case studies on real weather data (budget 220 PKR m)**")
            st.dataframe(pd.DataFrame([{
                "city": c["city"], "window start": c["as_of"], "level": c["level_label"],
                "risk QAE": round(c["risk"]["estimate"], 3), "risk exact": round(c["risk"]["exact"], 3),
                "damage before": round(c["damage_no_action_pkr_m"]),
                "damage after": round(c["plan"]["damage_after_pkr_m"]) if c["plan"] else None,
                "QAOA vs exact": ("matches" if c["plan"]["qaoa_matches"] else f"{c['plan']['qaoa_ratio']:.1%}") if c["plan"] else "-",
            } for c in cs["cases"]]), hide_index=True, use_container_width=True)
        for name, cap in [("fig_qae_benchmark.png", "Risk estimate: error against number of queries, QAE and classical sampling"),
                          ("fig_success_vs_budget.png", "QAOA benchmark: probability that 512 shots contain the optimal plan")]:
            if (ROOT / "docs" / "figures" / name).exists():
                st.image(str(ROOT / "docs" / "figures" / name), caption=cap, width=760)
    with t_meas:
        st.markdown("Measures drawn from Pakistani flood-management practice. Costs and effects are illustrative.")
        st.dataframe(pd.DataFrame([{"measure": iv.name, "type": iv.category, "cost (PKR m)": int(iv.cost_pkr_m),
                                    "what it does": iv.description} for iv in preset.interventions]), hide_index=True, use_container_width=True)
        for x in preset.synergies:
            st.write(f"➕ {x['pair'][0]} + {x['pair'][1]}: {x['reason']}")
        for x in preset.exclusions:
            st.write(f"⛔ {x['pair'][0]} / {x['pair'][1]}: {x['reason']}")
        for x in preset.dependencies:
            st.write(f"🔗 {x['if']} requires {x['requires']}: {x['reason']}")
    with t_lim:
        st.markdown(f"""
**What the quantum steps achieve**

- **Risk estimate.** On an ideal simulator, quantum amplitude estimation reached the same accuracy as classical sampling with about {GAIN:.1f} times fewer samples, which is the quadratic speed-up the theory predicts.
- **Response plan.** In nine real-event case studies the Qiskit plan was checked against every possible plan. The match counts are in the case-study table.
- **Built to scale.** Ten measures give 1,024 possible plans, which a laptop can check one by one. Forty measures would give about a trillion. The formulation, the penalty calibration and the local search all scale polynomially.

**What to keep in mind**

- **This is a simulator result at small size.** With 10 decisions and an 8-bin risk distribution, classical methods also give exact answers in milliseconds, so the quantum steps are validated here rather than needed. Whether QAOA helps at large sizes is an open research question.
- **Baadal does not forecast weather.** It collects forecasts (Open-Meteo: GEFS ensemble, GloFAS river discharge) and acts on them.
- **Replay is a hindcast.** Past events use reanalysis data, not the forecasts available at the time. Reanalysis rain underestimates cloudbursts, and GloFAS peak timing can differ from observations by days. Replay discharge uncertainty is an assumed 15% error, and replay rain uncertainty is the spread across nearby grid points.
- **The flood map is a simple fill model.** It uses real elevation (about 30 m data averaged to 100 m cells) and real land cover, but no hydraulics, embankments or drains, so it shows where water tends to go, not a verified flood extent.
- **Exposure and costs are illustrative.** Built-up ground and cropland come from satellite land cover; asset values, population densities, measure costs and measure effects are assumed. (We tried to use the WorldPop population service, but it was not reachable.)
- **The 'with plan' map converts the plan's effect into water depth per area.** It is consistent with the planning model but is not a new flood simulation. Markers show which area a measure affects, not a construction site.
- **QAE circuits are still deep** (several hundred CX gates after we reduced the register to 3 qubits). Under realistic noise the accuracy gain disappears, so this step needs error-corrected hardware.
- **QAOA is a heuristic** and misses the optimum in some cases; the app says so when it happens. In our 12-instance benchmark it found the best plan 10 times, and random sampling followed by the same local search found it 9 times, so at this size QAOA adds little over a simple classical method.
- **Decision support only.** Breaching and evacuation are decisions for irrigation departments, PDMAs and district administrations.
""")
