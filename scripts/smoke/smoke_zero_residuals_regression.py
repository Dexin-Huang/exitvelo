"""Regression smoke: zero residuals reproduce the pre-refactor controller.

Codex flagged this as a blocker before launching Tier 2A. We embed the
pre-refactor `get_action` logic inline as `legacy_get_action()` and
compare it action-by-action against the new
`CMUTrackingController(residuals=None).get_action()` for K steps from
the milestone start frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.motion.cmu_replay import CMUMocapReplay
from src.controllers.cmu_tracking_controller import CMUTrackingController

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
START_FRAME = 297
N_STEPS = 60

_BAT_GRIP_SCENE_IDX = 41


def legacy_get_action(replay: CMUMocapReplay) -> np.ndarray:
    """Exact pre-refactor logic. Reads from replay's current step_idx
    and advances it. Same shift-and-drop transform as the old controller."""
    qpos = replay.qpos_trajectory[:, replay.step_idx].copy()
    mocap_joints = qpos[7:].copy()
    scene_action = np.zeros(57, dtype=np.float64)
    scene_action[:_BAT_GRIP_SCENE_IDX] = mocap_joints[:_BAT_GRIP_SCENE_IDX]
    scene_action[_BAT_GRIP_SCENE_IDX] = 0.0
    scene_action[_BAT_GRIP_SCENE_IDX + 1:57] = mocap_joints[_BAT_GRIP_SCENE_IDX:56]
    replay.step_idx += 1
    return scene_action[:56].astype(np.float32)


def main():
    print("=" * 70)
    print("Regression: zero residuals == pre-refactor controller actions")
    print("=" * 70)

    legacy_replay = CMUMocapReplay(AMC_PATH)
    legacy_replay.reset(START_FRAME)
    new_ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=START_FRAME)

    max_diff = 0.0
    for step in range(N_STEPS):
        a_legacy = legacy_get_action(legacy_replay)
        a_new = new_ctrl.get_action()
        d = float(np.max(np.abs(a_legacy - a_new)))
        if d > max_diff:
            max_diff = d
        if d > 1e-6:
            print(f"  step {step}: max|diff| = {d:.6f}  (NONZERO!)")
            diff_idx = np.where(np.abs(a_legacy - a_new) > 1e-6)[0]
            print(f"    diverging indices: {diff_idx}")
            print(f"    legacy: {a_legacy[diff_idx]}")
            print(f"    new:    {a_new[diff_idx]}")
            raise AssertionError(
                f"Action divergence at step {step} above 1e-6 tolerance"
            )

    print(f"\n  All {N_STEPS} steps match within 1e-6 (max|diff| = {max_diff:.2e})")
    print("  PASS: zero residuals are bitwise-equivalent to the pre-refactor controller.")


if __name__ == "__main__":
    main()
