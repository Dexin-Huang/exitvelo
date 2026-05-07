"""Tier 2 comparison figure: CMA-ES (Tier 2A) vs PPO (Tier 2B).

The headline figure for the slide deck and report. Shows that the two
optimizers converge to the same carry (~184 ft) but different residual
configurations, evidence of a ridge in residual space.

Panels:
  A. Bar: baseline vs CMA-ES vs PPO (carry, exit_mph, bat_speed)
  B. Residuals: 3 named scalars, baseline=0 vs CMA-ES vs PPO
  C. Search progress: CMA-ES best score by gen, PPO best carry by step
  D. Sweep along hip_fire from -0.30 to +0.30 (with other dims = 0):
     shows the response surface, lets reader see the "ridge"
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.swing_residuals import SwingResiduals
from src.optim.kinematic_evaluator import kinematic_rollout

CMA_DIR = PROJECT_ROOT / "results" / "tier2a_cma" / "full_v1"
PPO_DIR = PROJECT_ROOT / "results" / "tier2b_ppo"
FIG_PATH = PROJECT_ROOT / "results" / "tier2_comparison.png"


def evaluate_metrics(residuals: SwingResiduals):
    r = kinematic_rollout(residuals=residuals)
    return {
        "carry_ft": r.carry_ft,
        "exit_mph": r.exit_mph,
        "bat_speed": r.sweet_speed_at_contact,
        "contact": r.contact,
    }


def main():
    cma = json.loads((CMA_DIR / "best.json").read_text())
    ppo = json.loads((PPO_DIR / "best.json").read_text())
    cma_res = SwingResiduals(**cma["residuals"])
    ppo_res = SwingResiduals(**ppo["best_residuals"])

    base = evaluate_metrics(SwingResiduals())
    cma_eval = evaluate_metrics(cma_res)
    ppo_eval = evaluate_metrics(ppo_res)
    print(f"baseline:  {base}")
    print(f"cma-es:    {cma_eval}")
    print(f"ppo:       {ppo_eval}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # Panel A: outcome metrics
    ax = axes[0, 0]
    methods = ["baseline", "CMA-ES", "PPO"]
    colors = ["#888", "#c33", "#3a3"]
    metrics = [
        ("carry (ft)",      [base["carry_ft"],  cma_eval["carry_ft"],  ppo_eval["carry_ft"]]),
        ("exit (mph)",      [base["exit_mph"],  cma_eval["exit_mph"],  ppo_eval["exit_mph"]]),
        ("bat speed (m/s)", [base["bat_speed"], cma_eval["bat_speed"], ppo_eval["bat_speed"]]),
    ]
    x = np.arange(len(metrics))
    w = 0.27
    for i, (label, color) in enumerate(zip(methods, colors)):
        vals = [m[1][i] for m in metrics]
        ax.bar(x + (i - 1) * w, vals, w, color=color, label=label)
    for j, (label, vals) in enumerate(metrics):
        for i, v in enumerate(vals):
            ax.text(j + (i - 1) * w, v + 0.01 * max(vals), f"{v:.1f}",
                    ha="center", fontsize=8, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("A. Outcome metrics")

    # Panel B: residuals
    ax = axes[0, 1]
    names = ["swing_timing\n(s)", "hip_fire\n(rad)", "uppercut\n(rad)"]
    cma_vals = [cma_res.swing_timing_s, cma_res.hip_fire_rad, cma_res.uppercut_rad]
    ppo_vals = [ppo_res.swing_timing_s, ppo_res.hip_fire_rad, ppo_res.uppercut_rad]
    x = np.arange(len(names))
    w = 0.35
    bounds_lo = [SwingResiduals.BOUNDS[k][0] for k in
                 ("swing_timing_s", "hip_fire_rad", "uppercut_rad")]
    bounds_hi = [SwingResiduals.BOUNDS[k][1] for k in
                 ("swing_timing_s", "hip_fire_rad", "uppercut_rad")]
    for xi, lo, hi in zip(x, bounds_lo, bounds_hi):
        ax.fill_between([xi - 0.4, xi + 0.4], lo, hi, color="#ddd", alpha=0.7)
    ax.bar(x - w/2, cma_vals, w, color=colors[1], label="CMA-ES")
    ax.bar(x + w/2, ppo_vals, w, color=colors[2], label="PPO")
    for i, (cv, pv) in enumerate(zip(cma_vals, ppo_vals)):
        ax.text(i - w/2, cv + (0.01 if cv >= 0 else -0.04), f"{cv:+.3f}",
                ha="center", fontsize=8, color=colors[1])
        ax.text(i + w/2, pv + (0.01 if pv >= 0 else -0.04), f"{pv:+.3f}",
                ha="center", fontsize=8, color=colors[2])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("B. Discovered residuals (gray = bounds)")

    # Panel C: search progress
    ax = axes[1, 0]
    # CMA-ES: read generations.csv
    gens = []
    cma_scores = []
    cma_carries = []
    with open(CMA_DIR / "generations.csv") as fp:
        for row in csv.DictReader(fp):
            gens.append(int(row["gen"]))
            cma_scores.append(float(row["best_score"]))
            cma_carries.append(float(row["best_carry_ft"]))
    # PPO: from history.json
    ppo_hist = json.loads((PPO_DIR / "history.json").read_text())
    ppo_steps = [h["step"] for h in ppo_hist]
    ppo_carries_running = []
    best = -1
    for h in ppo_hist:
        best = max(best, h["carry_ft"])
        ppo_carries_running.append(best)
    # CMA-ES running best
    cma_running = []
    best = -1
    for c in cma_carries:
        best = max(best, c)
        cma_running.append(best)
    # Two x axes (gens vs env steps), use top secondary axis
    ax.plot(gens, cma_running, color=colors[1], lw=2, label="CMA-ES (gen)")
    ax2 = ax.twiny()
    ax2.plot(ppo_steps, ppo_carries_running, color=colors[2], lw=2, alpha=0.85, label="PPO (env steps)")
    ax.set_xlabel("CMA-ES generation", color=colors[1])
    ax2.set_xlabel("PPO env steps", color=colors[2])
    ax.set_ylabel("running best carry (ft)")
    ax.tick_params(axis="x", colors=colors[1])
    ax2.tick_params(axis="x", colors=colors[2])
    ax.grid(alpha=0.3)
    ax.set_title("C. Search progress")

    # Panel D: hip_fire response surface (all other dims = 0)
    ax = axes[1, 1]
    sweep_vals = np.linspace(SwingResiduals.BOUNDS["hip_fire_rad"][0],
                             SwingResiduals.BOUNDS["hip_fire_rad"][1], 21)
    sweep_carries = []
    for hf in sweep_vals:
        r = kinematic_rollout(SwingResiduals(hip_fire_rad=float(hf)))
        sweep_carries.append(r.carry_ft if r.contact else 0.0)
    ax.plot(np.degrees(sweep_vals), sweep_carries, color="k", lw=2)
    ax.scatter(np.degrees([cma_res.hip_fire_rad]), [cma_eval["carry_ft"]],
               color=colors[1], s=80, zorder=5, label="CMA-ES")
    ax.scatter(np.degrees([ppo_res.hip_fire_rad]), [ppo_eval["carry_ft"]],
               color=colors[2], s=80, zorder=5, label="PPO")
    ax.scatter([0], [base["carry_ft"]], color=colors[0], s=80, zorder=5, label="baseline")
    ax.set_xlabel("hip_fire (degrees)")
    ax.set_ylabel("carry (ft)")
    ax.set_title("D. hip_fire response surface (other dims = 0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center")

    fig.suptitle(
        "Tier 2 — CMA-ES vs PPO on 3-d named swing residuals\n"
        f"baseline {base['carry_ft']:.1f} ft  →  CMA-ES {cma_eval['carry_ft']:.1f} ft / PPO {ppo_eval['carry_ft']:.1f} ft",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {FIG_PATH}")


if __name__ == "__main__":
    main()
