"""Tier 2A analysis figure: baseline mocap vs CMA-ES-optimized swing.

Produces a 2x2 figure for the slide deck and report:
  panel A: ball trajectory comparison (baseline vs optimum)
  panel B: bar chart — carry, exit_mph, bat_speed
  panel C: residual values (3 named scalars, optimum vs baseline=0)
  panel D: counterfactual ablation — zero out each residual dim of the
           optimum and re-evaluate, showing per-dim contribution.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.swing_residuals import SwingResiduals
from src.env.contacts import integrate_ball_trajectory, nathan_exit_velocity
from src.optim.kinematic_evaluator import kinematic_rollout

RESULTS_DIR = PROJECT_ROOT / "results" / "tier2a_cma" / "full_v1"
FIGURE_PATH = PROJECT_ROOT / "results" / "tier2a_cma" / "tier2a_summary.png"


@dataclass
class Trace:
    label: str
    color: str
    residuals: SwingResiduals
    carry_ft: float = 0.0
    exit_mph: float = 0.0
    launch_deg: float = 0.0
    bat_speed_ms: float = 0.0
    contact: bool = False
    traj: np.ndarray | None = None


def evaluate_with_trajectory(residuals: SwingResiduals) -> Trace:
    res = kinematic_rollout(residuals=residuals)
    # Also compute the post-contact trajectory for plotting
    if res.contact:
        # Recompute exit_velocity at contact frame for traj plot
        # (kinematic_rollout already saved carry_ft/exit_mph; we need the
        # post-contact ball trajectory to draw it).
        # Re-run a thin computation: at contact, sweet pos was at ~target;
        # we don't have the exact position here, so approximate by
        # initializing from a placeholder height.
        bat_dir = np.array([1.0, 0.0, 0.0])
        bat_vel = bat_dir * (res.sweet_speed_at_contact)
        ball_in = np.array([-41.6, 0.0, 0.0])
        exit_v = nathan_exit_velocity(bat_vel, ball_in)
        # Use a typical contact spot height for traj viz
        contact_xyz = np.array([0.43, -0.95, 1.35])
        traj, _, _ = integrate_ball_trajectory(contact_xyz, exit_v)
    else:
        traj = None
    return Trace(
        label="",
        color="",
        residuals=residuals,
        carry_ft=res.carry_ft,
        exit_mph=res.exit_mph,
        launch_deg=res.launch_deg,
        bat_speed_ms=res.sweet_speed_at_contact,
        contact=res.contact,
        traj=traj,
    )


def main():
    print("=" * 70)
    print("Tier 2A analysis figure")
    print("=" * 70)

    blob = json.loads((RESULTS_DIR / "best.json").read_text())
    best_res = SwingResiduals(**blob["residuals"])

    baseline = evaluate_with_trajectory(SwingResiduals())
    baseline.label = "baseline (mocap)"
    baseline.color = "#888"

    optimum = evaluate_with_trajectory(best_res)
    optimum.label = "CMA-ES optimum"
    optimum.color = "#c33"

    # Counterfactual: zero one dim of the optimum at a time
    cf_traces: list[Trace] = []
    for name in ("swing_timing_s", "hip_fire_rad", "uppercut_rad"):
        kwargs = asdict(best_res)
        kwargs[name] = 0.0
        cf = evaluate_with_trajectory(SwingResiduals(**kwargs))
        cf.label = f"opt - {name}"
        cf.color = "#369"
        cf_traces.append((name, cf))

    print(f"  baseline:  carry={baseline.carry_ft:6.1f}  exit={baseline.exit_mph:5.1f}  "
          f"bat={baseline.bat_speed_ms:5.2f}")
    print(f"  optimum:   carry={optimum.carry_ft:6.1f}  exit={optimum.exit_mph:5.1f}  "
          f"bat={optimum.bat_speed_ms:5.2f}")
    for name, cf in cf_traces:
        print(f"  - {name:20s}: carry={cf.carry_ft:6.1f}  "
              f"loss vs opt = {optimum.carry_ft - cf.carry_ft:+5.2f} ft")

    # Build the figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel A: ball trajectory
    ax = axes[0, 0]
    if baseline.traj is not None:
        ax.plot(baseline.traj[:, 0], baseline.traj[:, 2],
                color=baseline.color, lw=2, label=baseline.label)
    if optimum.traj is not None:
        ax.plot(optimum.traj[:, 0], optimum.traj[:, 2],
                color=optimum.color, lw=2, label=optimum.label)
    ax.set_xlabel("x distance from contact (m)")
    ax.set_ylabel("height z (m)")
    ax.set_title("A. Ball flight (drag + bounce)")
    ax.axhline(y=0.037, color="brown", lw=0.5, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend()

    # Panel B: bar chart
    ax = axes[0, 1]
    metrics = ["carry (ft)", "exit (mph)", "bat speed (m/s)"]
    base_vals = [baseline.carry_ft, baseline.exit_mph, baseline.bat_speed_ms]
    opt_vals = [optimum.carry_ft, optimum.exit_mph, optimum.bat_speed_ms]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, base_vals, w, color=baseline.color, label=baseline.label)
    ax.bar(x + w/2, opt_vals,  w, color=optimum.color,  label=optimum.label)
    for i, (b, o) in enumerate(zip(base_vals, opt_vals)):
        ax.text(i + w/2, o + 0.01 * max(opt_vals), f"+{o-b:.1f}",
                ha="center", fontsize=9, color=optimum.color)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("B. Outcome metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel C: residual values
    ax = axes[1, 0]
    names = ["swing_timing\n(s)", "hip_fire\n(rad)", "uppercut\n(rad)"]
    vals = [
        best_res.swing_timing_s,
        best_res.hip_fire_rad,
        best_res.uppercut_rad,
    ]
    bounds_lo = [SwingResiduals.BOUNDS[k][0] for k in
                 ("swing_timing_s", "hip_fire_rad", "uppercut_rad")]
    bounds_hi = [SwingResiduals.BOUNDS[k][1] for k in
                 ("swing_timing_s", "hip_fire_rad", "uppercut_rad")]
    x = np.arange(len(names))
    # Bound rectangles
    for xi, lo, hi in zip(x, bounds_lo, bounds_hi):
        ax.fill_between([xi - 0.4, xi + 0.4], lo, hi, color="#ccc", alpha=0.4)
    ax.bar(x, vals, 0.6, color=optimum.color)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 * max(abs(v), 0.05)) * (1 if v >= 0 else -1),
                f"{v:+.3f}", ha="center", fontsize=10, color=optimum.color)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("C. CMA-ES optimum residuals (gray = bounds)")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: counterfactual ablation (per-dim contribution)
    ax = axes[1, 1]
    contrib_names = []
    contrib_loss = []
    for name, cf in cf_traces:
        contrib_names.append(name.replace("_", "\n"))
        contrib_loss.append(optimum.carry_ft - cf.carry_ft)
    x = np.arange(len(contrib_names))
    bars = ax.bar(x, contrib_loss, 0.6, color="#369")
    for bar, v in zip(bars, contrib_loss):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f"{v:+.2f}", ha="center", fontsize=10, color="#369")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(contrib_names)
    ax.set_ylabel("carry loss when this dim is zeroed (ft)")
    ax.set_title("D. Per-dim contribution")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Tier 2A — CMA-ES on 3 named swing residuals", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
