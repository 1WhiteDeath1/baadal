"""Figures, results table and architecture diagram generated from results/results.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from .flood import build_terrain, run_flood
from .model import build_problem
from .preset import load_preset

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
RES = ROOT / "results"

C_QAOA = "#2563eb"
C_SLACK = "#7c3aed"
C_RAND = "#9ca3af"
C_EXACT = "#059669"
C_BAD = "#dc2626"


def same(a, b) -> bool:
    """Tie-aware 'matches the exact optimum' test on objective values."""
    return a is not None and b is not None and abs(a - b) < 1e-6


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- maps
def fig_flood_map(preset_id="riverine_punjab", scenario="exceptional", chosen=None, name="fig_flood_map.png"):
    preset = load_preset(preset_id)
    terrain = build_terrain(preset)
    flood = run_flood(preset, terrain, preset.scenario(scenario))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    depth = np.ma.masked_where(flood.depth <= 0.01, flood.depth)
    ax.imshow(terrain.elev, cmap="Greys_r", alpha=0.55)
    im = ax.imshow(depth, cmap="Blues", vmin=0, vmax=max(1.5, float(flood.depth.max())))
    ax.imshow(np.ma.masked_where(~terrain.channel, terrain.channel), cmap="winter", alpha=0.9)
    for z in preset.zones:
        r0, r1, c0, c1 = z["rect"]
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False, lw=1.6, ec="#b45309"))
        ax.text(c0, r0 - 0.9, z["name"], fontsize=7, color="#7c2d12", weight="bold")
    for k, iv in enumerate(preset.interventions, start=1):
        r, c = iv.location
        sel = chosen is None or iv.id in chosen
        ax.scatter([c], [r], s=150, marker="o", c=C_EXACT if sel else "white", edgecolors="black", zorder=5)
        ax.text(c, r, str(k), ha="center", va="center", fontsize=7, zorder=6, color="white" if sel else "black")
    ax.set_title(f"Baadal - {preset.scenario(scenario)['name']}\nflood depth (m), zones, numbered candidate measures", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="flood depth (m)")
    legend = "\n".join(f"{k}. {iv.id}" for k, iv in enumerate(preset.interventions, start=1))
    ax.text(1.32, 0.5, legend, transform=ax.transAxes, fontsize=7, va="center", family="monospace")
    _save(fig, name)


def fig_scenario_damage(preset_id="riverine_punjab"):
    preset = load_preset(preset_id)
    terrain = build_terrain(preset)
    zones = preset.zone_ids
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(preset.scenarios)
    for k, s in enumerate(preset.scenarios):
        fl = run_flood(preset, terrain, s)
        ax.bar(np.arange(len(zones)) + k * w, [fl.zone_damage[z] for z in zones], w,
               label=f"{s['name'].replace(' (forecast)', '')} ({s['peak_inflow_cusecs'] // 1000}k cusecs, {s['rain_72h_mm']} mm)")
    ax.set_xticks(np.arange(len(zones)) + 0.4 - w / 2)
    ax.set_xticklabels([z["name"] for z in preset.zones], fontsize=7)
    ax.set_ylabel("expected damage (PKR m, illustrative)")
    ax.set_title("Classical stage: forecast -> damage per zone", fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, "fig_scenario_damage.png")


def fig_before_after(results, scenario="exceptional", budget="220"):
    preset = load_preset(results["meta"]["preset"])
    terrain = build_terrain(preset)
    flood = run_flood(preset, terrain, preset.scenario(scenario))
    problem = build_problem(preset, flood)
    row = results["scenarios"][scenario]["budgets"][budget]
    plan_exact = row["classical"]["x"]
    plan_q = row["qaoa"]["unbalanced"]["sampled"]["x"]
    ev_e = problem.true_eval(plan_exact)
    ev_q = problem.true_eval(plan_q)
    zones = preset.zone_ids
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(zones))
    ax.bar(x - 0.27, [flood.zone_damage[z] for z in zones], 0.27, color=C_BAD, label="no action")
    ax.bar(x, [ev_e["remaining_damage"][z] for z in zones], 0.27, color=C_EXACT, label="exact optimum plan")
    ax.bar(x + 0.27, [ev_q["remaining_damage"][z] for z in zones], 0.27, color=C_QAOA, label="QAOA plan (sampled)")
    ax.set_xticks(x)
    ax.set_xticklabels([z["name"] for z in preset.zones], fontsize=7)
    ax.set_ylabel("remaining damage (PKR m)")
    ax.set_title(f"Before / after at budget {budget} PKR m ({scenario})", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, "fig_before_after.png")


# --------------------------------------------------------------------------- QAOA evaluation
def fig_success_vs_budget(results):
    sids = list(results["scenarios"])
    fig, axes = plt.subplots(1, len(sids), figsize=(5 * len(sids), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    shots = results["meta"]["settings"]["shots"]
    for ax, sid in zip(axes, sids):
        bs = list(results["scenarios"][sid]["budgets"])
        rows = [results["scenarios"][sid]["budgets"][b] for b in bs]
        x = np.arange(len(bs))
        ax.bar(x - 0.27, [r["qaoa"]["unbalanced"]["per_p"][-1]["p_find_opt_in_shots"] for r in rows], 0.27, color=C_QAOA, label="QAOA, unbalanced (10 q)")
        ax.bar(x, [r["qaoa"]["slack"]["per_p"][-1]["p_find_opt_in_shots"] for r in rows], 0.27, color=C_SLACK, label="QAOA, slack (13-15 q)")
        ax.bar(x + 0.27, [r["random"]["p_find_opt_in_shots"] for r in rows], 0.27, color=C_RAND, label="uniform random")
        ax.set_xticks(x)
        ax.set_xticklabels(bs)
        ax.set_xlabel("budget (PKR m)")
        ax.set_title(f"{sid}: P(optimum among {shots} shots)", fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("probability")
    axes[0].legend(fontsize=7, loc="lower left")
    _save(fig, "fig_success_vs_budget.png")


def fig_vs_depth(results):
    cal = results["meta"]["settings"]["calibration_instance"]
    row = results["scenarios"][cal["scenario"]]["budgets"][str(cal["budget_pkr_m"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for enc, col in [("unbalanced", C_QAOA), ("slack", C_SLACK)]:
        pp = row["qaoa"][enc]["per_p"]
        axes[0].plot([m["p"] for m in pp], [m["p_opt"] for m in pp], "o-", color=col, label=enc)
        axes[1].plot([m["p"] for m in pp], [m["p_feasible"] for m in pp], "o-", color=col, label=enc)
    axes[0].axhline(row["random"]["p_opt"], color=C_RAND, ls="--", label="uniform random")
    axes[1].axhline(row["random"]["p_feasible"], color=C_RAND, ls="--", label="uniform random")
    axes[0].set_title("P(optimal plan) vs QAOA depth p", fontsize=10)
    axes[1].set_title("P(feasible plan) vs QAOA depth p", fontsize=10)
    for a in axes:
        a.set_xlabel("p (layers)")
        a.legend(fontsize=7)
    _save(fig, "fig_vs_depth.png")


def fig_top_plans(results):
    cal = results["meta"]["settings"]["calibration_instance"]
    row = results["scenarios"][cal["scenario"]]["budgets"][str(cal["budget_pkr_m"])]
    tops = row["qaoa"]["unbalanced"]["top_plans"]
    opt = set(row["classical"]["chosen"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = ["+".join(t["chosen"]) or "(none)" for t in tops]
    cols = [C_EXACT if (t["feasible"] and same(t["quad_score"], row["classical"]["quad_score"])) else (C_QAOA if t["feasible"] else C_BAD) for t in tops]
    ax.barh(range(len(tops))[::-1], [t["prob"] for t in tops], color=cols)
    ax.set_yticks(range(len(tops))[::-1])
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("probability in final QAOA state")
    ax.set_title("Most likely plans (green = exact optimum, blue = feasible, red = violates a constraint)", fontsize=9)
    _save(fig, "fig_top_plans.png")


def fig_noise(results):
    cal = results["meta"]["settings"]["calibration_instance"]
    sc = results["scenarios"][cal["scenario"]]["budgets"]
    rows = [r for b in sc.values() for r in b.get("noise", [])]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for k, metric in enumerate(["p_opt_sampled", "p_feasible_sampled"]):
        for j, sim in enumerate(["ideal", "noisy"]):
            vals = [np.mean([r[metric] for r in rows if r["p"] == p and r["sim"] == sim]) for p in (1, 2)]
            axes[k].bar(np.array([1, 2]) + (j - 0.5) * 0.35, vals, 0.35, label=sim, color=C_QAOA if sim == "ideal" else "#f59e0b")
        axes[k].set_xticks([1, 2])
        axes[k].set_xlabel("p")
    axes[0].set_title("P(optimal), mean over budgets", fontsize=9)
    axes[1].set_title("P(feasible), mean over budgets", fontsize=9)
    axes[0].legend(fontsize=7)
    _save(fig, "fig_noise.png")


# --------------------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec="#1f2937", lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9, weight="bold")
        ax.text(x + w / 2, y + h - 5.8, body, ha="center", va="top", fontsize=7.0, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, ha="center", fontsize=6.6, color="#374151")

    data, classical, quantum, ui = "#fef3c7", "#e0f2fe", "#ede9fe", "#dcfce7"
    box(1, 32, 26, 20, "1. Weather collection", "Open-Meteo (free, no key)\nrain forecast + 31-member ensemble\nGloFAS river discharge + 50 members\n12 Pakistani cities, live or replay\nsnapshots saved for offline reruns", data)
    box(31, 32, 25, 20, "2. Threat detection", "river: peak discharge vs local\n2 / 5 / 10-year floods (2000-2023)\nurban: 72 h rainfall thresholds\nthreat level per city", classical)
    box(60, 32, 27, 20, "3. Risk: QAE (Qiskit)", "ensemble -> 8 bins on 3 qubits\ncomparator flags bins >= threshold\nGrover powers + max-likelihood fit\n-> P(flood threshold exceeded)\nvs exact and classical sampling", quantum)
    box(91, 32, 28, 20, "4. Threat model", "real elevation + satellite land cover\nfill model: height above the river,\nrain ponding on low built-up ground\n-> flood depth, damage and people\n(values per cell are illustrative)", classical)
    box(91, 4, 28, 22, "5. Planning model + QUBO", "Pakistan measure catalogue\noverlap, synergy, breach-evacuation\nbudget via unbalanced penalty\n(10 qubits) or slack bits (13-15)", classical)
    box(57, 4, 30, 22, "6. Response: QAOA (Qiskit)", "QUBO -> Ising (h, J)\nH + p x [RZ/RZZ cost, RX mixer]\nCVaR objective, COBYLA, INTERP\nAer statevector training\n512 shots scored classically + local search", quantum)
    box(1, 4, 52, 22, "7. Simulation, validation and application", "plan applied -> before / after flood map on satellite imagery\nvalidated against: exact brute force (2^10 plans), greedy,\nuniform random sampling, exact ensemble risk, classical Monte Carlo\nguided app: city -> threat -> impact -> response -> review", ui)
    arrow(27.6, 42, 30.4, 42)
    arrow(56.6, 42, 59.4, 42, "ensemble")
    arrow(87.6, 42, 90.4, 42, "threat")
    arrow(105, 31.4, 105, 26.8)
    ax.text(106.5, 28.6, "damage per zone", fontsize=6.6, color="#374151")
    arrow(90.4, 15, 87.6, 15, "h, J")
    arrow(56.4, 15, 53.6, 15, "counts")
    ax.text(60, 55, "Baadal architecture: data (yellow), classical (blue), quantum in Qiskit (purple), decision layer (green)",
            ha="center", fontsize=10, weight="bold")
    _save(fig, "architecture.png")


def fig_qae_benchmark(cs):
    b = cs["qae_benchmark"]
    q = [r["queries"] for r in b["rows"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.loglog(q, [r["classical_mc_mean_abs_error"] for r in b["rows"]], "s--", color=C_RAND, label="classical Monte Carlo")
    ax.loglog(q, [r["qae_mean_abs_error"] for r in b["rows"]], "o-", color=C_QAOA, label="QAE, ideal simulator")
    nz = [(r["queries"], r["qae_noisy_mean_abs_error"]) for r in b["rows"] if r["qae_noisy_mean_abs_error"] is not None]
    ax.loglog([x for x, _ in nz], [y for _, y in nz], "^:", color="#f59e0b", label="QAE, noisy simulator")
    ax.set_xlabel("queries (uses of the loading circuit)")
    ax.set_ylabel("mean absolute error")
    ax.set_title(f"Risk estimate error, {b['case']}\nexact value {b['exact']:.3f}, mean over {b['n_seeds']} seeds", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "fig_qae_benchmark.png")


def fig_case_study(event_id, city_id, name, budget=220):
    """Collected data, modelled flood and flood with the optimal plan, on the real land-cover grid."""
    from matplotlib.colors import ListedColormap

    from . import geodata, planner, weather
    from .classical import brute_force
    from .simulate import apply_plan
    from .threat import LEVEL_LABEL, scan

    snap = weather.load_replay(event_id)
    a = next(x for x in scan(snap, weather.load_reference()) if x["city_id"] == city_id)
    preset, terrain, flood, problem = planner.problem_for(a)
    sim = apply_plan(preset, terrain, flood, problem, brute_force(problem, budget)["x"])
    lc = np.array(geodata.load_site(city_id)["landcover"])
    lc_cmap = ListedColormap(["#9cc3e6", "#8fb58a", "#a9d6b8", "#e9dfa6", "#c9a9a6", "#e6e1da", "#f2f6f9", "#d0d0d0", "#ddceb5"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    days = [d[5:] for d in a["days"]]
    ax.bar(days, [v or 0 for v in a["rain_daily_mm"]], color="#7fb2d6", label="daily rain (mm)")
    ax.set_ylabel("rain, mm per day")
    if a["discharge_daily_m3s"]:
        ax2 = ax.twinx()
        ax2.plot(days, a["discharge_daily_m3s"], "o-", color=C_BAD, lw=2, label="river discharge")
        ax2.axhline(a["thresholds"]["high"], color="#ea580c", ls="--", lw=1)
        ax2.axhline(a["thresholds"]["exceptional"], color="#7f1d1d", ls="--", lw=1)
        ax2.set_ylabel("river discharge (m3/s)")
        ax2.set_ylim(0, None)
    ax.set_title(f"Collected data: {a['city']}, from {snap['as_of']}: {LEVEL_LABEL[a['level']]}"
                 + ("\n(dashed lines: 2-year and 10-year flood)" if a["discharge_daily_m3s"] else ""), fontsize=9)
    fig.subplots_adjust(wspace=0.32)
    zmax = max(1.2, float(flood.depth.max()))
    for axm, depth, title in [(axes[1], flood.depth, f"No action: {flood.total_damage:,.0f} PKR m damage"),
                              (axes[2], sim["depth_after"], f"With the optimal plan: {sim['damage_after']:,.0f} PKR m damage")]:
        axm.imshow(lc, cmap=lc_cmap, vmin=0, vmax=8, interpolation="nearest")
        im = axm.imshow(np.ma.masked_where(depth <= 0.05, depth), cmap="Blues", vmin=0, vmax=zmax, alpha=0.85, interpolation="bilinear")
        axm.set_title(title, fontsize=9)
        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_xlabel("12 km x 12 km. Background: land cover (blue water, red built-up, yellow crops)", fontsize=7)
    fig.colorbar(im, ax=axes[2], fraction=0.045, label="flood depth (m)")
    _save(fig, name)


# --------------------------------------------------------------------------- tables
def results_table(results):
    lines = []
    header = ["scenario", "budget (PKR m)", "exact plan", "exact score", "damage avoided (PKR m)", "people protected",
              "greedy score", "QAOA-unb plan score", "QAOA-unb = exact?", "QAOA-unb + local search = exact?", "random 512 + local search = exact?", "QAOA-unb P(opt)", "QAOA-unb P(find in 512)",
              "QAOA-slack P(opt)", "QAOA-slack P(find in 512)", "random P(find in 512)", "QAOA-unb P(feasible)", "random P(feasible)"]
    rows = []
    for sid, sc in results["scenarios"].items():
        for b, r in sc["budgets"].items():
            qu, qs = r["qaoa"]["unbalanced"], r["qaoa"]["slack"]
            ex = r["classical"]
            rows.append([
                sid, b, " + ".join(ex["chosen"]), f"{ex['quad_score']:.0f}",
                f"{ex['true']['damage_avoided_pkr_m']:.0f}", f"{ex['true']['people_protected']:.0f}",
                f"{r['greedy']['quad_score']:.0f}",
                f"{qu['sampled']['quad_score']:.0f}" if qu["sampled"]["quad_score"] is not None else "-",
                "yes" if same(qu["sampled"]["quad_score"], ex["quad_score"]) else "no",
                "yes" if same(qu["polished"]["quad_score"], ex["quad_score"]) else "no",
                "yes" if same(r["random"]["polished"]["quad_score"], ex["quad_score"]) else "no",
                f"{qu['per_p'][-1]['p_opt']:.4f}", f"{qu['per_p'][-1]['p_find_opt_in_shots']:.2f}",
                f"{qs['per_p'][-1]['p_opt']:.4f}", f"{qs['per_p'][-1]['p_find_opt_in_shots']:.2f}",
                f"{r['random']['p_find_opt_in_shots']:.2f}",
                f"{qu['per_p'][-1]['p_feasible']:.2f}", f"{r['random']['p_feasible']:.2f}",
            ])
    RES.mkdir(exist_ok=True)
    with (RES / "results_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    (RES / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return header, rows


def summary(results) -> dict:
    """Headline numbers used by README / report / pitch."""
    out = {}
    for sid, sc in results["scenarios"].items():
        rs = list(sc["budgets"].values())
        qu = [r["qaoa"]["unbalanced"] for r in rs]
        qs = [r["qaoa"]["slack"] for r in rs]
        out[sid] = {
            "n_budgets": len(rs),
            "qaoa_unb_matches_exact": sum(same(q["sampled"]["quad_score"], r["classical"]["quad_score"]) for q, r in zip(qu, rs)),
            "qaoa_slack_matches_exact": sum(same(q["sampled"]["quad_score"], r["classical"]["quad_score"]) for q, r in zip(qs, rs)),
            "greedy_matches_exact": sum(abs(r["greedy"]["quad_score"] - r["classical"]["quad_score"]) < 1e-6 for r in rs),
            "qaoa_unb_polished_matches": sum(same(q["polished"]["quad_score"], r["classical"]["quad_score"]) for q, r in zip(qu, rs)),
            "qaoa_slack_polished_matches": sum(same(q["polished"]["quad_score"], r["classical"]["quad_score"]) for q, r in zip(qs, rs)),
            "random_raw_matches": sum(same(r["random"]["sampled"]["quad_score"], r["classical"]["quad_score"]) for r in rs),
            "random_polished_matches": sum(same(r["random"]["polished"]["quad_score"], r["classical"]["quad_score"]) for r in rs),
            "mean_popt_unb": float(np.mean([q["per_p"][-1]["p_opt"] for q in qu])),
            "mean_popt_slack": float(np.mean([q["per_p"][-1]["p_opt"] for q in qs])),
            "random_popt": rs[0]["random"]["p_opt"],
            "mean_pfeas_unb": float(np.mean([q["per_p"][-1]["p_feasible"] for q in qu])),
            "mean_pfeas_slack": float(np.mean([q["per_p"][-1]["p_feasible"] for q in qs])),
            "random_pfeas_mean": float(np.mean([r["random"]["p_feasible"] for r in rs])),
            "mean_find_unb": float(np.mean([q["per_p"][-1]["p_find_opt_in_shots"] for q in qu])),
            "mean_find_slack": float(np.mean([q["per_p"][-1]["p_find_opt_in_shots"] for q in qs])),
            "random_find": rs[0]["random"]["p_find_opt_in_shots"],
            "unb_cx": qu[0]["circuit"]["transpiled_cx"],
            "slack_cx_max": max(q["circuit"]["transpiled_cx"] for q in qs),
            "slack_qubits": sorted({q["qubits"] for q in qs}),
        }
    (RES / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def make_all(results):
    fig_architecture()
    cs_path = RES / "case_studies.json"
    if cs_path.exists():
        fig_qae_benchmark(json.loads(cs_path.read_text(encoding="utf-8")))
        fig_case_study("2010-07-28_kp_superflood", "sukkur", "fig_case_sukkur_2010.png")
        fig_case_study("2022-07-23_karachi", "karachi", "fig_case_karachi_2022.png")
    fig_scenario_damage()
    cal = results["meta"]["settings"]["calibration_instance"]
    row = results["scenarios"][cal["scenario"]]["budgets"][str(cal["budget_pkr_m"])]
    fig_flood_map(scenario=cal["scenario"], chosen=set(row["classical"]["chosen"]))
    fig_flood_map(scenario="very_high", chosen=set(results["scenarios"]["very_high"]["budgets"][str(cal["budget_pkr_m"])]["classical"]["chosen"]) if "very_high" in results["scenarios"] else None,
                  name="fig_flood_map_very_high.png")
    fig_before_after(results, cal["scenario"], str(cal["budget_pkr_m"]))
    fig_success_vs_budget(results)
    fig_vs_depth(results)
    fig_top_plans(results)
    fig_noise(results)
    results_table(results)
    s = summary(results)
    print(json.dumps(s, indent=2))
    print(f"Figures -> {FIG}\nTables -> {RES}")
