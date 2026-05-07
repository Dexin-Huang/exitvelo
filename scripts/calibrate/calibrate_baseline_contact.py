"""Find a (start_frame, pitch_x) combo where zero residuals make contact.

Codex flagged this as the blocker before launching CMA-ES: without a
working baseline, the optimizer has no contact signal.

We sweep start_frame and pitch_x in a coarse grid and report which
combos produce contact with zero residuals + no jitter. The best combo
becomes the env default for Tier 2A.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.swing_residuals import SwingResiduals
from src.optim.kinematic_evaluator import evaluate_residuals_kinematic


def main():
    print("=" * 70)
    print("Baseline-contact calibration sweep")
    print("=" * 70)
    print("  zero residuals, no jitter, vary start_frame x pitch_x")
    print()

    start_frames = list(range(280, 460, 20))  # 9 values
    pitch_xs = [-3.0, -5.0, -8.0, -12.0, -16.49]  # close -> MLB
    print(f"  start_frames: {start_frames}")
    print(f"  pitch_xs:     {pitch_xs}")
    print()

    contacts = []
    for sf in start_frames:
        for px in pitch_xs:
            results = evaluate_residuals_kinematic(
                residuals=SwingResiduals(),
                seeds=[0], pitch_jitter=False,
                start_frame=sf,
                pitch_release_x=px, pitch_release_z=1.5,
                pitch_velocity=(41.6, 0.0, 0.0),
            )
            r = results[0]
            tag = "HIT" if r.contact else "miss"
            print(f"  sf={sf:3d}  px={px:+6.2f}  {tag}  "
                  f"min_d={r.min_bat_ball_dist:5.3f}m  "
                  f"exit={r.exit_mph:5.1f} mph  "
                  f"carry={r.carry_ft:6.1f} ft  "
                  f"steps={r.steps:3d}  "
                  f"sweet_spd={r.sweet_speed_at_contact:.1f}m/s")
            if r.contact:
                contacts.append((sf, px, r))

    print()
    print("=" * 70)
    print(f"Contacts found: {len(contacts)}")
    print("=" * 70)
    if contacts:
        best = max(contacts, key=lambda t: t[2].carry_ft)
        sf, px, r = best
        print(f"  BEST: start_frame={sf}, pitch_x={px:+.2f}")
        print(f"        exit={r.exit_mph:.2f} mph  launch={r.launch_deg:.1f} deg  "
              f"carry={r.carry_ft:.1f} ft  total={r.total_ft:.1f} ft")
    else:
        print("  No contacts found in this grid. Need to widen the search,")
        print("  drop pd-control gains, or accept that CMA-ES will operate")
        print("  on the min-distance proxy until residuals find the ball.")


if __name__ == "__main__":
    main()
