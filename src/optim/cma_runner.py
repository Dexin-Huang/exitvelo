"""Tier 2A: CMA-ES on the 3 active swing residuals.

Searches normalized [-1, +1]^3 (swing_timing, hip_fire, uppercut), pads
the deferred barrel_roll / plate_reach with zeros, and minimizes
cma_objective_kin from src/optim/kinematic_evaluator.py. Logs each
generation to generations.csv and dumps the best-rollout JSON.

Entry point:
  scripts/run/run_tier2a_full.py  -- CMA-ES driver
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cma
import numpy as np

from src.controllers.swing_residuals import SwingResiduals
from src.optim.kinematic_evaluator import (
    cma_objective_kin as cma_objective,
    evaluate_residuals_kinematic as evaluate_residuals,
    kin_aggregate as aggregate,
    kin_asdict_list as asdict_list,
    DEFAULT_START_FRAME,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "tier2a_cma"

# 3 active dims, 2 deferred dims fixed at zero.
ACTIVE_NAMES = ["swing_timing_s", "hip_fire_rad", "uppercut_rad"]


def expand_3d_to_residuals(x3: np.ndarray) -> SwingResiduals:
    """Take a 3-vector in normalized [-1, +1] over the active dims and
    return a SwingResiduals with the 2 deferred dims set to 0."""
    x5 = np.zeros(5, dtype=np.float64)
    x5[:3] = np.asarray(x3, dtype=np.float64)
    return SwingResiduals.from_normalized(x5)


def run_cma(
    *,
    popsize: int = 10,
    max_gens: int = 40,
    sigma0: float = 0.35,
    seeds_per_eval: int = 3,
    deterministic_until_gen: int = 10,
    pitch_jitter: bool = True,
    out_dir: Path | None = None,
    tag: str = "v1",
) -> dict:
    """Run CMA-ES on the 3-d active residual space.

    For generations < deterministic_until_gen, evaluate with 1
    deterministic seed (fast search). After that, evaluate with
    seeds_per_eval jittered seeds (variance estimate).
    """
    out_dir = out_dir or (RESULTS_DIR / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    es = cma.CMAEvolutionStrategy(
        x0=np.zeros(3),
        sigma0=sigma0,
        inopts={
            "popsize": popsize,
            "bounds": [[-1.0] * 3, [1.0] * 3],
            "maxiter": max_gens,
            "verbose": -9,  # silence default printer; we log ourselves
        },
    )

    history: list[dict] = []
    best_overall = {"score": np.inf, "x3": None, "residuals": None,
                    "agg": {}, "rollouts": []}

    csv_path = out_dir / "generations.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "gen", "popsize", "best_score", "best_swing_timing_s",
            "best_hip_fire_rad", "best_uppercut_rad", "best_carry_ft",
            "best_exit_mph", "best_launch_deg", "best_contact_rate",
            "mean_score", "sigma", "wall_time_s",
        ])

        gen = 0
        while not es.stop():
            t0 = time.time()
            xs = es.ask()  # list of 3-d vectors (already in normalized space)
            scores: list[float] = []
            cand_records: list[dict] = []
            cand_results: list[list] = []

            seeds = list(range(seeds_per_eval)) if gen >= deterministic_until_gen else [0]

            for cand_idx, x in enumerate(xs):
                residuals = expand_3d_to_residuals(np.asarray(x))
                results = evaluate_residuals(
                    residuals=residuals, seeds=seeds, pitch_jitter=pitch_jitter,
                )
                score = cma_objective(results)
                agg = aggregate(results)
                scores.append(score)
                cand_records.append({
                    "x3": list(map(float, x)),
                    "residuals": asdict(residuals),
                    "score": score,
                    "agg": agg,
                })
                cand_results.append(results)

            es.tell(xs, scores)

            best_idx = int(np.argmin(scores))
            best_rec = cand_records[best_idx]
            wall_s = time.time() - t0

            if best_rec["score"] < best_overall["score"]:
                best_overall = {
                    "score":      best_rec["score"],
                    "x3":         best_rec["x3"],
                    "residuals":  best_rec["residuals"],
                    "agg":        best_rec["agg"],
                    "rollouts":   asdict_list(cand_results[best_idx]),
                    "gen":        gen,
                }

            row = [
                gen, popsize, best_rec["score"],
                best_rec["residuals"]["swing_timing_s"],
                best_rec["residuals"]["hip_fire_rad"],
                best_rec["residuals"]["uppercut_rad"],
                best_rec["agg"].get("mean_carry_ft", 0.0),
                best_rec["agg"].get("mean_exit_mph", 0.0),
                best_rec["agg"].get("mean_launch_deg", 0.0),
                best_rec["agg"].get("contact_rate", 0.0),
                float(np.mean(scores)),
                float(es.sigma),
                wall_s,
            ]
            writer.writerow(row)
            fp.flush()

            print(
                f"gen {gen:3d} | popsize {popsize} | sigma {es.sigma:.3f} | "
                f"best score {best_rec['score']:8.2f} | "
                f"carry {best_rec['agg'].get('mean_carry_ft', 0):6.1f} ft | "
                f"contact {best_rec['agg'].get('contact_rate', 0)*100:5.1f}% | "
                f"{wall_s:5.1f}s"
            )
            history.append({"gen": gen, "candidates": cand_records, "best_idx": best_idx})
            gen += 1

    # Save the best-overall rollout package
    (out_dir / "best.json").write_text(json.dumps(best_overall, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print()
    print("=" * 70)
    print("CMA-ES converged")
    print("=" * 70)
    print(f"  best score:       {best_overall['score']:.4f}")
    print(f"  best residuals:   {best_overall['residuals']}")
    print(f"  best agg:         {best_overall['agg']}")
    print(f"  saved:            {out_dir}")
    return best_overall
