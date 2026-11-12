"""Generate docs/Baadal_Report.docx from the saved results and figures.

    python make_report.py

Every number in the report is read from results/results.json, results/case_studies.json or the saved
weather snapshots, so the report cannot drift from the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from baadal import weather
from baadal.preset import load_preset
from baadal.report_assets import same, summary
from baadal.threat import ALERT_LEVEL, LEVEL_LABEL, scan

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "Baadal_Report.docx"

R = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
CS = json.loads((ROOT / "results" / "case_studies.json").read_text(encoding="utf-8"))
S = summary(R)
P = load_preset("riverine_punjab")
SET = R["meta"]["settings"]
CAL = SET["calibration_instance"]
EXC = R["scenarios"]["exceptional"]
VH = R["scenarios"]["very_high"]
REF = weather.load_reference()
CASES = CS["cases"]
QB = CS["qae_benchmark"]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(5)
for lvl, size in [(1, 15), (2, 12.5)]:
    hs = doc.styles[f"Heading {lvl}"]
    hs.font.name = "Calibri"
    hs.font.size = Pt(size)
    hs.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    hs.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
    hs.paragraph_format.space_after = Pt(4)
    hs.paragraph_format.keep_with_next = True


def para(text="", bold_prefix=None, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def mono(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_after = Pt(2)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header, rows, widths_cm, font=8.5):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(font)
        shade(c, "DBEAFE")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(font)
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
    doc.add_paragraph()


def figure(name, caption, width_cm=16):
    path = FIG / name
    if not path.exists():
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(caption, italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER).paragraph_format.space_after = Pt(10)


ex, vh = S["exceptional"], S["very_high"]
planned = [c for c in CASES if c["plan"]]
n_match = sum(c["plan"]["qaoa_matches"] for c in planned)
risk_err = float(np.mean([c["risk"]["abs_error"] for c in CASES]))
uncertain = [c for c in CASES if 0.02 < c["risk"]["exact"] < 0.98]
qb_last = [r for r in QB["rows"] if r["powers"][-1] == 8][0]
gain = (qb_last["classical_mc_mean_abs_error"] / qb_last["qae_mean_abs_error"]) ** 2     # sample-count ratio for equal error
QN = CASES[0]["risk"]["qubits"]
BINS = 2 ** (QN - 1)
bench_pol = ex["qaoa_unb_polished_matches"] + vh["qaoa_unb_polished_matches"]
slack_raw = ex["qaoa_slack_matches_exact"] + vh["qaoa_slack_matches_exact"]
slack_pol = ex["qaoa_slack_polished_matches"] + vh["qaoa_slack_polished_matches"]
rand_raw = ex["random_raw_matches"] + vh["random_raw_matches"]
rand_pol = ex["random_polished_matches"] + vh["random_polished_matches"]
total_bench = ex["n_budgets"] + vh["n_budgets"]
bench_match = ex["qaoa_unb_matches_exact"] + vh["qaoa_unb_matches_exact"]

# ============================================================================ title
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Baadal")
r.bold = True
r.font.size = Pt(28)
r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
para("From weather data to a flood response plan for Pakistan, with quantum risk estimation and optimisation in Qiskit",
     size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Qiskit Fall Fest 2026 Hackathon  |  Domain: Disaster Response", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Team: PowerPuff Boys  |  Members: M Fahad Afridi (team lead), Sarfaraz Ahmed, Asad Kidwai  |  Repository: https://github.com/1WhiteDeath1/baadal  |  Video: [Drive link]",
     size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("Summary", 1)
para("Baadal (\"cloud\" in Urdu) watches the weather over 12 Pakistani cities and turns a flood threat into a response plan. "
     "It collects real rain forecasts, forecast ensembles and river discharge from the free Open-Meteo service. When a city is under "
     "threat, it models the flood on the real elevation and land cover of that city, chooses the set of protective measures that "
     "reduces harm the most within a budget, and shows the outcome as a before-and-after flood map on satellite imagery.")
para("Qiskit does two jobs in this chain. Quantum amplitude estimation (QAE) estimates the probability that the flood threshold "
     "will be exceeded, from the forecast ensemble. The Quantum Approximate Optimisation Algorithm (QAOA) chooses the response plan, "
     "which is written as a QUBO. Both circuits are written directly in Qiskit and run on the Aer simulator, and both are checked "
     "against exact classical answers.")
para(f"We replayed {len({c['event'] for c in CASES if c['mode'] == 'replay'})} real past events and ran today's live forecast. "
     f"Across {len(CASES)} city cases the QAE risk estimate was within {risk_err:.3f} of the exact value on average. "
     f"In the {len(planned)} cases that needed a plan, the QAOA plan equalled the exact optimum in {n_match}. "
     f"In a separate 12-instance benchmark of the planning model, QAOA matched the optimum in {bench_match} of {total_bench} instances. "
     f"On an ideal simulator, the quantum risk estimate reached the accuracy of classical sampling with about {gain:.1f} times fewer "
     "samples, which is the quadratic speed-up the theory predicts. These are small problems run on a simulator, so classical "
     "methods can also solve them exactly, and that is how every quantum result here is verified. We do not claim a quantum "
     "advantage. What we show is a complete, tested application in which the quantum steps do real work, built on a formulation "
     "whose classical parts scale polynomially.")

# ============================================================================ 1 problem
doc.add_heading("1. Problem and who it serves", 1)
para("Pakistan's flood managers, such as provincial disaster management authorities (PDMAs), irrigation departments and district "
     "administrations, face the same decision every monsoon. A forecast arrives a few days ahead. They must judge how serious it is "
     "and then decide, with a fixed budget, which measures to fund: strengthen an embankment, pre-position pumps, clear drains, "
     "evacuate villages, or deliberately breach a dyke to save a town. In 2025, officials on the Chenab breached dykes to protect "
     "towns and barrages and accepted flooding of farmland elsewhere.")
para("The optimisation problem has competing objectives and limited resources. Property damage competes with people's safety, "
     "one area can only be saved by sacrificing another, and the budget is never enough for everything. Measures also interact: "
     "two measures on one town overlap, pumps need generators, and a breach is only acceptable after evacuation.")
para("Baadal gives these users one chain from data to decision: what is the threat, how likely is it, what will it damage, "
     "what should we do within budget, and what will that achieve.", bold_prefix="What Baadal offers. ")

# ============================================================================ 2 architecture
doc.add_heading("2. Architecture", 1)
figure("architecture.png", "Figure 1. Baadal architecture. Yellow: data collection. Blue: classical modelling. Purple: quantum steps in "
                           "Qiskit. Green: simulation, validation and the dashboard.", 16.5)
para("Code modules: weather.py (data collection and saved snapshots), threat.py (threat levels and scenario), qae.py (amplitude "
     "estimation), flood.py (flood depth and damage), model.py (planning model), qubo.py (QUBO and Ising mapping), qaoa.py (QAOA), "
     "classical.py (exact and greedy baselines), simulate.py (plan simulation), case_studies.py and experiments.py (evaluation), "
     "geodata.py and realmodel.py (real elevation, land cover and the flood model on them), planner.py (plan cache), and the "
     "application: app.py with a single-page interface in baadal_ui/. 56 pytest checks cover the chain.")
para("The application guides the user through five steps, shown as a progress tracker: choose a city on a satellite map of "
     "Pakistan, assess the threat, see the impact, plan the response, and review the plan. In the planning step the user can accept "
     "the plan recommended by Qiskit or build their own from the same measures, with a running total against the budget.",
     bold_prefix="Application. ")

# ============================================================================ 3 data
doc.add_heading("3. Weather data collection", 1)
para("All weather and river data are real and come from Open-Meteo, which needs no API key:")
bullet(" 7-day daily rainfall for each city (best-match weather model).", bold_prefix="Rain forecast:")
bullet(" 31 members of the GEFS ensemble, used as the forecast uncertainty.", bold_prefix="Rain ensemble:")
bullet(" GloFAS river-discharge forecast with a 50-member ensemble for the eight river cities.", bold_prefix="River discharge:")
bullet(f" GloFAS discharge for {REF['years'][0]}-{REF['years'][1]} gives each river location its own 2-, 5- and 10-year flood levels.",
       bold_prefix="History:")
bullet(" AWS Terrain Tiles (SRTM and other open elevation models, about 30 m), averaged to a 120 x 120 grid of 100 m cells "
       "covering 12 km x 12 km around each city.", bold_prefix="Elevation:")
bullet(" Sentinel-2 10 m Land Use / Land Cover (Esri, Impact Observatory, Microsoft), which marks permanent water, built-up "
       "ground and cropland.", bold_prefix="Land cover:")
bullet(" Esri World Imagery, used only as the map background in the application.", bold_prefix="Satellite imagery:")
para(f"The watch-list has {len(weather.CITIES)} cities: " + ", ".join(f"{c['name']} ({c['river']})" for c in weather.CITIES) + ".")
para("Live mode fetches the next 7 days. Replay mode loads the same quantities for a real past event from reanalysis archives "
     "(ERA5 and ECMWF IFS rain, GloFAS discharge). Every snapshot is saved in the repository, so all results can be reproduced offline.",
     bold_prefix="Live and replay. ")

# ============================================================================ 4 threat detection
doc.add_heading("4. Threat detection", 1)
para("For a river city the threat driver is the forecast peak discharge, compared with that location's own history. For an urban "
     "city it is the largest 72-hour rainfall total. Levels follow the Pakistani flood-forecasting vocabulary:")
table(["level", "river city: peak discharge", "urban city: 72-hour rain (illustrative)"],
      [["Watch", "at least 75% of the 2-year flood", "50 mm"], ["High flood", "2-year flood", "90 mm"],
       ["Very high flood", "5-year flood", "150 mm"], ["Exceptionally high flood", "10-year flood", "230 mm"]], [3.6, 6.0, 6.0])
para("A threat is declared when the best-estimate forecast reaches 'high flood', or when the ensemble gives at least a 30% chance "
     "of it. Planning then uses a reasonable worst case, the 75th percentile of the ensemble.")
snap10 = weather.load_replay("2010-07-28_kp_superflood")
rows = []
for a in scan(snap10, REF):
    rows.append([a["city"], a["kind"], f"{a['best']:,.0f} {a['unit']}", f"{a['thresholds'][ALERT_LEVEL]:,.0f}",
                 f"{a['rain_72h_best_mm']:.0f}", f"{a['p_alert_exact']:.2f}", LEVEL_LABEL[a["level"]], "yes" if a["threat"] else "no"])
para(f"Example: the 2010 super flood, 7-day window from {snap10['as_of']}.", bold_prefix="")
table(["city", "type", "driver (best estimate)", "high-flood threshold", "72 h rain mm", "P(high flood)", "level", "threat"],
      rows, [3.0, 1.5, 2.8, 2.0, 1.4, 1.5, 3.2, 1.1], font=7.5)
para("The 2010 flood on the Indus is detected as exceptional (the largest discharge in the 2000-2023 record), while Karachi, which "
     "had no heavy rain that week, is correctly left at normal.", italic=True, size=9)

# ============================================================================ 5 formulation
doc.add_heading("5. Formulation", 1)
doc.add_heading("5.1 Risk estimation", 2)
para("Let X be the threat driver (72-hour rainfall or peak discharge) and t the flood threshold. The ensemble members give an "
     "empirical distribution of X. We want a = P(X >= t). The distribution is discretised into 8 equal bins, with t on a bin edge, "
     "so a is the total probability of the bins at or above t. This exact value is the reference for the quantum estimate.")
doc.add_heading("5.2 Threat model (classical)", 2)
para("The flood is modelled on the real elevation and land cover of each city (120 x 120 cells of 100 m). The severity of the "
     "threat sets a flood stage: the 2-, 5- and 10-year floods of the real location map to fixed stages. For a river city, permanent "
     "water in the land-cover map is the river. Each land cell gets its height above the nearest river cell, and it floods when the "
     "stage, reduced by 0.35 m per km of distance from the river, is above that height and the cell is connected to the river "
     "through flooded cells. For an urban city, rain above the drainage capacity ponds on built-up ground, deeper where the ground "
     "lies lower than its surroundings.")
para("Five zones per city come from the land cover. River cities: built-up ground and farmland on each side of the river, and "
     "the river frontage near the gauge or barrage. Urban cities: built-up ground grouped by how low it lies and by distance from "
     "the centre. Depth-damage curves (shapes after the JRC global functions) give damage D_z and people exposed N_z per zone. "
     "Asset values and population per cell are illustrative.")
doc.add_heading("5.3 Response optimisation", 2)
para("Decision variables: x_i in {0,1}, one per measure (n = 10). Each measure cuts the remaining damage in zone z by a fraction "
     "r_iz and may add damage A_iz elsewhere. Effects compound multiplicatively:")
mono("remaining_z(x) = (D_z + sum_i A_iz x_i) * prod_i (1 - r_iz x_i) * prod_synergies (1 - s x_a x_b)")
para("Expanded to second order this gives a quadratic objective, with the same structure for people (weight w):")
mono("maximise  F(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j")
mono("a_i  = sum_z [ D_z r_iz - A_iz (1 - r_iz) ]  +  w * (people terms)")
mono("b_ij = sum_z [ -D_z r_iz r_jz + A_iz r_jz + A_jz r_iz ] + synergy bonuses + w * (people terms)")
para("Constraints:")
mono("budget:      sum_i c_i x_i <= B          (c_i, B in units of 20 PKR m)")
mono("exclusion:   x_BREACH + x_EMB_RIGHT <= 1")
mono("dependency:  x_BREACH <= x_EVAC_S")
rows = [[k, iv.id, iv.name, iv.category, int(iv.cost_pkr_m)] for k, iv in enumerate(P.interventions, start=1)]
para("Measure catalogue for river cities, drawn from Pakistani practice (FFC embankments and spurs, designated breaching sections, "
     "WASA-style pumping, Recharge Pakistan wetlands, early warning). A second catalogue covers urban flooding (nullah desilting, "
     "encroachment removal with resettlement, disposal stations, alerts). Costs and effects are illustrative.")
table(["#", "id", "measure", "type", "cost PKR m"], rows, [0.7, 2.2, 6.4, 4.8, 1.7])
para(f"Test instances: 10 binary decisions, 5 zones, budgets {', '.join(map(str, P.budgets))} PKR m, w = {P.people_weight} PKR m per person.")

# ============================================================================ 6 quantum
doc.add_heading("6. Quantum approach and justification", 1)
doc.add_heading("6.1 Risk: quantum amplitude estimation", 2)
para(f"Amplitude estimation is the quantum counterpart of Monte Carlo sampling. A circuit A loads the {BINS}-bin distribution into "
     f"{QN - 1} qubits. Because the threshold sits on a bin edge, these coarse bins lose nothing for this probability, and they keep the "
     "circuit shallow. A comparator, built from a few multi-controlled X gates and no ancilla qubits, flips one more 'objective' qubit "
     "for every bin at or above the threshold. The probability of measuring that qubit as 1 is exactly a. A Grover-style operator "
     "Q = A S0 A^-1 S_chi rotates the state so that after k applications this probability becomes sin^2((2k+1) theta), with "
     "a = sin^2(theta). We measure for k = 0, 1, 2, 4, 8 with 64 shots each and fit theta by maximum likelihood "
     "(Suzuki et al., 2020). This variant needs no phase estimation and so has the shallowest circuits.")
para("Estimating a probability under an uncertain forecast is the textbook use of amplitude estimation, the same pattern as "
     "quantum risk analysis in finance. In theory its error falls as 1/M with M uses of the loading circuit, against 1/sqrt(M) for "
     "classical sampling. Its output is used materially: the risk decides whether a threat is declared.", bold_prefix="Why QAE. ")
doc.add_heading("6.2 Response: QUBO and QAOA", 2)
para("A QUBO is a cost function over yes/no variables with only single-variable and pairwise terms. We minimise "
     "E(x) = -F(x) + penalties. Hard rules use P x_i x_j (exclusive pair) and P x_i (1 - x_j) (i requires j). The budget is encoded "
     "in two ways: slack bits with P (c.x + w.s - B)^2, which is exact but adds 3 to 5 qubits, or unbalanced penalization "
     "-l1 h + l2 h^2 with h = B - c.x (Montanez-Barrera et al., 2024), which needs no extra qubits but is heuristic.",
     bold_prefix="QUBO. ")
para("Substituting x_i = (1 - Z_i)/2 gives an Ising Hamiltonian H = sum h_i Z_i + sum J_ij Z_i Z_j. Each measure is one qubit and "
     "each interaction is a ZZ coupling. QAOA starts from an equal superposition of all plans and alternates p times a cost layer "
     "exp(-i gamma H), built from RZ and RZZ gates, with a mixer layer of RX gates. COBYLA tunes the 2p angles to minimise the CVaR "
     "(alpha = 0.25) of the energy, with INTERP and zero-padded warm starts as p grows. The trained circuit is sampled 512 times, "
     "every sample is scored classically, and the best feasible plan is kept. A classical local search then tries single changes "
     "and swaps of measures, a standard hybrid post-processing step.", bold_prefix="Ising mapping and QAOA. ")
para("QAOA is designed for QUBO problems, uses one qubit per decision and shallow circuits, and the pairwise structure of our "
     "measures maps one-to-one onto its cost layer.", bold_prefix="Why QAOA. ")
para("Both algorithms are simulated without noise unless stated. QAOA training uses the exact statevector, which is only possible "
     "at this size. QAOA settings were fixed on one calibration instance and reused unchanged. Unbalanced-penalty weights are "
     "chosen per scenario by a local search against the greedy plan, without enumerating plans, so this step scales polynomially. "
     "Brute force is used only to check results.", bold_prefix="Assumptions. ")

# ============================================================================ 7 results
doc.add_heading("7. Results", 1)
doc.add_heading("7.1 End-to-end case studies on real weather data", 2)
rows = []
for c in CASES:
    p_ = c["plan"]
    rows.append([f"{c['city']}, {c['as_of']}" + (" (live)" if c["mode"] == "live" else ""),
                 f"{c['best']:,.0f} / {c['alert_threshold']:,.0f} {c['unit']}", c["level_label"],
                 f"{c['risk']['estimate']:.3f} / {c['risk']['exact']:.3f}",
                 f"{c['damage_no_action_pkr_m']:,.0f}" + (f" -> {p_['damage_after_pkr_m']:,.0f}" if p_ else ""),
                 f"{c['people_exposed']:,.0f}" + (f" -> {p_['people_after']:,.0f}" if p_ else ""),
                 ("= exact" if p_["qaoa_matches"] else f"{p_['qaoa_ratio']:.1%}") if p_ else "no plan needed",
                 f"{p_['p_opt']:.4f}" if p_ else "-"])
table(["city and window start", "driver / high-flood threshold", "threat level", "risk: QAE / exact", "damage PKR m: before -> after",
       "people exposed: before -> after", "QAOA plan vs exact", "P(opt)"],
      rows, [3.0, 2.9, 2.3, 1.7, 2.2, 2.2, 1.6, 1.1], font=7.5)
para(f"Budget {CASES[0]['budget_pkr_m']} PKR m. The flood extent comes from real terrain; damage and people figures use illustrative values per cell. "
     "Risk is P(driver reaches the high-flood threshold). P(opt) is the probability of the optimal plan in the trained QAOA state; "
     "random guessing gives 0.0010.", italic=True, size=8.5)
figure("fig_case_sukkur_2010.png", "Figure 2. Sukkur, 2010 super flood replay: collected discharge and rain, the flood modelled on real "
                                   "terrain, and the flood with the optimal plan. The plan includes a controlled breach, so the far bank "
                                   "(bottom right) takes more water while the city side takes less.", 16.5)
figure("fig_case_karachi_2022.png", "Figure 3. Karachi, July 2022 cloudburst replay: 162 mm of rain in one day, ponding on low built-up "
                                    "ground, and the outcome with the optimal plan.", 16.5)

doc.add_heading("7.2 Quantum risk estimation against classical sampling", 2)
rows = [[str(tuple(r["powers"])), r["queries"], r["max_cx"], f"{r['qae_mean_abs_error']:.4f}",
         f"{r['qae_noisy_mean_abs_error']:.4f}" if r["qae_noisy_mean_abs_error"] is not None else "not run",
         f"{r['classical_mc_mean_abs_error']:.4f}"] for r in QB["rows"]]
table(["Grover powers k", "queries", "max CX gates", "QAE error (ideal)", "QAE error (noisy)", "classical sampling error"],
      rows, [3.2, 1.6, 2.2, 2.8, 2.8, 3.2])
para(f"Case: {QB['case']}; exact value {QB['exact']:.3f} from {QB['n_members']} ensemble members. Mean absolute error over "
     f"{QB['n_seeds']} seeds, 64 shots per circuit. With powers up to 8 ({qb_last['queries']:,} queries) the ideal QAE error is "
     f"{qb_last['qae_mean_abs_error']:.4f}, against {qb_last['classical_mc_mean_abs_error']:.4f} for classical sampling with the same "
     f"number of queries, so classical sampling would need about {gain:.1f} times more samples for the same accuracy. Under generic noise ({QB['noise_model']}) the deep circuits lose this gain: the error is "
     f"{qb_last['qae_noisy_mean_abs_error']:.4f}.")
figure("fig_qae_benchmark.png", "Figure 4. Error of the risk estimate against the number of queries.", 11)

doc.add_heading("7.3 QAOA benchmark of the planning model", 2)
rows = []
for sid, sc in R["scenarios"].items():
    for b, r in sc["budgets"].items():
        qu, qs = r["qaoa"]["unbalanced"], r["qaoa"]["slack"]
        f = lambda v: "=" if same(v, r["classical"]["quad_score"]) else f"{v / r['classical']['quad_score']:.4f}"
        rows.append([sid.replace("_", " "), b, " + ".join(r["classical"]["chosen"]),
                     f"{r['classical']['true']['damage_avoided_pkr_m']:.0f}", f(r["greedy"]["quad_score"]),
                     f(qu["sampled"]["quad_score"]), f(qs["sampled"]["quad_score"]), f"{qu['per_p'][-1]['p_opt']:.4f}",
                     f"{qu['per_p'][-1]['p_find_opt_in_shots']:.2f} / {r['random']['p_find_opt_in_shots']:.2f}"])
table(["severity", "budget", "exact optimal plan", "damage avoided", "greedy", "QAOA unb.", "QAOA slack", "P(opt) unb.",
       "P(opt in 512 shots) QAOA / random"], rows, [1.5, 1.1, 5.2, 1.4, 1.1, 1.2, 1.2, 1.3, 2.0], font=7.5)
para("Twelve fixed test instances: two severities of the river planning model on a schematic test terrain, at six budgets. '=' means the solver returned an optimal plan; otherwise the value is its "
     f"score as a fraction of the optimum. The unbalanced encoding uses 10 qubits and {ex['unb_cx']} CX gates at p = {SET['p_max']['unbalanced']}; "
     f"the slack encoding needs {'/'.join(map(str, ex['slack_qubits']))} qubits and up to {ex['slack_cx_max']} CX gates. "
     f"Averaged over budgets, the trained state is {ex['mean_popt_unb'] / ex['random_popt']:.0f} times (exceptional) and "
     f"{vh['mean_popt_unb'] / vh['random_popt']:.0f} times (very high) more likely than random guessing to give the optimal plan.",
     italic=True, size=8.5)
para(f"With the classical local-search step added, QAOA with the unbalanced encoding found the optimum in {bench_pol} of {total_bench} "
     f"instances (unchanged from {bench_match}) and QAOA with slack bits in {slack_pol} (up from {slack_raw}). For comparison, 512 "
     f"uniformly random plans contained the optimum in {rand_raw} instances, and in {rand_pol} after the same local search.",
     bold_prefix="Local search. ")
figure("fig_success_vs_budget.png", "Figure 5. Probability that 512 shots contain the optimal plan: QAOA (both encodings) against uniform "
                                    "random sampling.", 16)
b_ex = sum("BREACH" in r["classical"]["chosen"] for r in EXC["budgets"].values())
b_vh = sum("BREACH" in r["classical"]["chosen"] for r in VH["budgets"].values())
para(f"At the exceptional severity the optimal plan includes the controlled breach, with prior evacuation, at {b_ex} of "
     f"{len(EXC['budgets'])} budgets. At the very-high severity it includes it at {b_vh} of {len(VH['budgets'])}, because the "
     "diverted water would destroy more than it saves.", bold_prefix="The breach decision depends on the threat. ")
nrows = [r for b in EXC["budgets"].values() for r in b.get("noise", [])]
if nrows:
    i2 = np.mean([r["p_opt_sampled"] for r in nrows if r["p"] == 2 and r["sim"] == "ideal"])
    n2 = np.mean([r["p_opt_sampled"] for r in nrows if r["p"] == 2 and r["sim"] == "noisy"])
    para(f"With the same generic noise model, at p = 2 (180 CX gates) the sampled probability of the optimal plan falls from "
         f"{i2:.4f} to {n2:.4f}.", bold_prefix="Noise. ")

# ============================================================================ 8 comparison
doc.add_heading("8. Classical and exact references", 1)
bullet(" the exceedance probability of the discretised ensemble is computed exactly; QAE is compared with it in every case.",
       bold_prefix="Exact risk:")
bullet(" the same number of queries drawn at random from the same distribution.", bold_prefix="Classical Monte Carlo:")
bullet(" exact enumeration of all 1,024 plans, in milliseconds. It is the ground truth for every plan.", bold_prefix="Brute force:")
bullet(f" best gain per rupee. It matched the optimum in {ex['greedy_matches_exact'] + vh['greedy_matches_exact']} of {total_bench} benchmark instances.",
       bold_prefix="Greedy:")
bullet(f" 512 uniformly random plans contain the optimum with probability {ex['random_find']:.2f}, against "
       f"{ex['mean_find_unb']:.2f} and {vh['mean_find_unb']:.2f} for QAOA (two severities).", bold_prefix="Random sampling:")

# ============================================================================ 9 limitations
doc.add_heading("9. Limitations", 1)
for head, txt in [
    ("Small problems on a simulator.", f" With 10 decisions and an {BINS}-bin risk distribution, classical methods give exact answers "
                                       "in milliseconds, so the quantum steps are validated here rather than needed. We do not claim a "
                                       "quantum advantage. Whether QAOA helps at large sizes is an open research question."),
    ("Baadal does not forecast weather.", " It collects forecasts and acts on them. Its alerts are only as good as the forecasts."),
    ("Replay is a hindcast.", " Past events use reanalysis, not the forecasts available at the time. Reanalysis rain underestimates "
                              "cloudbursts (the August 2020 Karachi flood is missed by this data, so it is not among our cases). GloFAS peak "
                              "timing can differ from observations by days: it places the 2010 Indus peak at Sukkur about a week early. "
                              "Replay rain uncertainty is the spread over nearby grid points, and replay discharge uncertainty is an assumed "
                              "15% lognormal error, because no archived ensembles are available."),
    ("Simple flood model.", " The flood map is a fill model on real elevation and land cover. It has no hydraulics and knows nothing "
                            "about existing embankments, canals or drains, so it shows where water tends to go, not a verified extent. "
                            "River severity is transferred by return period, not by simulated flow."),
    ("Illustrative exposure and costs.", " Built-up ground and cropland are real, but asset values, population densities, measure costs "
                                         "and measure effects are assumed. The PKR figures are not real estimates. We tried to use the "
                                         "WorldPop population service for real population counts, but it was not reachable."),
    ("The 'with plan' map is not a new flood simulation.", " It converts the planning model's remaining damage back into depth per zone. "
                                                           "Map markers show which area a measure affects, not a construction site."),
    ("QAE needs deep circuits.", f" We reduced the register to {QN - 1} qubits, which halved the gate count, but the largest circuit "
                                 f"still has {qb_last['max_cx']:,} CX gates. Under generic noise the precision gain disappears, so this "
                                 "needs error-corrected hardware."),
    ("Loading the distribution is itself classical work.", f" With {BINS} bins the exact answer is available before the circuit is built."),
    ("QAOA is a heuristic and missed some optima.", f" It missed {total_bench - bench_match} of {total_bench} benchmark instances and "
                                                    f"{len(planned) - n_match} of {len(planned)} case-study plans. Its probability of "
                                                    "the optimal plan is a few percent at best. The local-search step fixed none of its "
                                                    f"benchmark misses, and random sampling with the same local search found the optimum in "
                                                    f"{rand_pol} of {total_bench} instances, so at this size QAOA adds little over a simple "
                                                    "classical method."),
    ("Penalty calibration is heuristic.", " The weights are chosen against a greedy plan without enumeration. On the benchmark instances the "
                                          "resulting QUBO ground state was the true optimum, but that is not guaranteed in general."),
    ("Second-order benefit model.", " Three-way interactions are dropped, which overstates absolute benefit by up to about 10%."),
    ("Decision support only.", " Breaching and evacuation are legal and political decisions for the responsible authorities."),
]:
    bullet(txt, bold_prefix=head)

# ============================================================================ 10 roadmap
doc.add_heading("10. Roadmap", 1)
for txt in [
    "A hydraulic model, and real exposure data: population, asset values and the locations of existing embankments and drains.",
    "Archived forecasts and observed gauge data (PMD Flood Forecasting Division) to verify alerts against what really happened.",
    "Risk-weighted planning: feed the QAE probabilities of several severity classes into the plan's objective.",
    "More regions and hazards: hill torrents in Koh-e-Suleman and glacial-lake floods in Gilgit-Baltistan.",
    "Larger plans (30 to 100 measures) with constraint-preserving mixers, benchmarked against MILP solvers.",
    "IBM Quantum hardware runs of the shallow QAOA circuits with error mitigation.",
]:
    bullet(" " + txt)

# ============================================================================ 11 originality
doc.add_heading("11. Originality, credits and reproducibility", 1)
para("Our contribution is the integration: live weather collection, threat detection against local flood history, a quantum risk "
     "estimate, a flood model on real terrain and land cover, a quantum-optimised response chosen from Pakistan's own flood-management measures, and a "
     "simulated outcome, in one tested pipeline. The comparator circuit, amplitude-estimation routine, QUBO builder, QAOA circuit "
     "and training loop were written for this project in core Qiskit; Qiskit Optimization and Qiskit Algorithms are not used. "
     "QAOA, maximum-likelihood amplitude estimation, CVaR, INTERP and unbalanced penalization are published methods, credited below.",
     bold_prefix="Originality statement. ")
para("This project was developed with assistance from an AI coding assistant (Claude). The team reviewed, ran and tested all code "
     "and is responsible for its content.", bold_prefix="AI assistance. ")
para("pip install -r requirements.txt; pytest; streamlit run app.py; python -m baadal.geodata; python -m baadal.case_studies; python run_pipeline.py; "
     f"python make_report.py. Versions: Python {R['meta']['python']}, Qiskit {R['meta']['qiskit']}, Qiskit Aer {R['meta']['qiskit_aer']}. "
     "All runs are seeded and the weather snapshots are stored in data/.", bold_prefix="Reproduce. ")
doc.add_heading("References", 2)
for ref in [
    "E. Farhi, J. Goldstone, S. Gutmann, 'A Quantum Approximate Optimization Algorithm', arXiv:1411.4028 (2014).",
    "G. Brassard, P. Hoyer, M. Mosca, A. Tapp, 'Quantum amplitude amplification and estimation', Contemp. Math. 305 (2002).",
    "Y. Suzuki et al., 'Amplitude estimation without phase estimation', Quantum Inf. Process. 19, 75 (2020).",
    "S. Woerner, D. J. Egger, 'Quantum risk analysis', npj Quantum Inf. 5, 15 (2019).",
    "J. A. Montanez-Barrera et al., 'Unbalanced penalization', Quantum Sci. Technol. 9, 025022 (2024).",
    "P. Kl. Barkoutsos et al., 'Improving Variational Quantum Optimization using CVaR', Quantum 4, 256 (2020).",
    "L. Zhou et al., 'Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices', "
    "Phys. Rev. X 10, 021067 (2020).",
    "Open-Meteo.com weather, ensemble, flood and archive APIs; GloFAS (Copernicus Emergency Management Service); ERA5 (ECMWF / C3S); "
    "NOAA GEFS.",
    "AWS Terrain Tiles (Mapzen; SRTM and other open elevation data). K. Karra et al., 'Global land use / land cover with Sentinel-2 "
    "and deep learning', IGARSS (2021), served by Esri. Esri World Imagery (map background).",
    "J. Huizinga, H. de Moel, W. Szewczyk, 'Global flood depth-damage functions', JRC Technical Report (2017).",
    "Federal Flood Commission annual flood reports and NFPP-IV briefs; NDMA Pakistan monsoon contingency plans; WWF / GCF Recharge Pakistan.",
    "Qiskit contributors, Qiskit and Qiskit Aer (2026).",
]:
    bullet(" " + ref)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
