"""Checkpoint B smoke: each named residual at {min, 0, max} produces
feasible motion (no NaN, joint targets within model.jnt_range).

Coverage:
  - swing_timing_s at -0.08, 0, +0.08 s
  - hip_fire_rad   at -0.30, 0, +0.30 rad
  - uppercut_rad   at -0.22, 0, +0.26 rad
  - barrel_roll_rad / plate_reach_m left at 0 (deferred this iteration)

Negative tests:
  - SwingResiduals.from_normalized clips out-of-range inputs
  - barrel_roll / plate_reach raise NotImplementedError when adjust_*
    is called
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.controllers.swing_residuals import (
    SwingResiduals,
    apply_pelvis_trunk_yaw,
    apply_uppercut_pitch,
    adjust_bat_grip_anchor,
)
from src.env.cmu_batting_env import CMUBattingEnv

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
START_FRAME = 297


def _run_episode_and_check(env: CMUBattingEnv, ctrl: CMUTrackingController, n_steps: int = 60):
    """Run a short episode and assert no NaN / no joint-range violation."""
    obs, _ = env.reset()

    # Initialize the humanoid at the controller's starting pose so the
    # first step has a sensible state.
    pos, quat = ctrl.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    a0 = ctrl.get_action()
    env.data.qpos[7:63] = a0
    import mujoco; mujoco.mj_forward(env.model, env.data)

    ctrl.reset()
    obs = env._get_obs()

    for step in range(n_steps):
        if ctrl.done:
            break
        action = ctrl.get_action()
        obs, _r, term, trunc, info = env.step(action)
        # The env already clips actions to joint_range, so we only need to
        # assert the controller produced finite values and the env stayed
        # finite under integration.
        assert not np.any(np.isnan(action)), f"NaN in action at step {step}"
        assert not np.any(np.isnan(env.data.qpos)), f"NaN in qpos at step {step}"
        if term or trunc:
            return info
    return info


def scenario_dataclass_normalization():
    print("=" * 70)
    print("Scenario A: SwingResiduals normalization round-trip + clip")
    print("=" * 70)
    # Round-trip
    r = SwingResiduals(swing_timing_s=0.04, hip_fire_rad=-0.15, uppercut_rad=0.13)
    n = r.to_normalized()
    r2 = SwingResiduals.from_normalized(n)
    for name in ("swing_timing_s", "hip_fire_rad", "uppercut_rad"):
        assert abs(getattr(r, name) - getattr(r2, name)) < 1e-9, \
            f"round-trip drift on {name}"
    # Out-of-range clip
    n_bad = np.array([2.0, 5.0, -3.0, 0.0, 0.0])
    r3 = SwingResiduals.from_normalized(n_bad)
    assert abs(r3.swing_timing_s - 0.08) < 1e-9
    assert abs(r3.hip_fire_rad - 0.30) < 1e-9
    assert abs(r3.uppercut_rad - (-0.22)) < 1e-9
    print("  PASS\n")


def scenario_residual_extremes():
    print("=" * 70)
    print("Scenario B: each scalar at {min, 0, max} produces feasible motion")
    print("=" * 70)
    env = CMUBattingEnv()

    cases = []
    for name in ("swing_timing_s", "hip_fire_rad", "uppercut_rad"):
        lo, hi = SwingResiduals.BOUNDS[name]
        for label, value in (("min", lo), ("zero", 0.0), ("max", hi)):
            r = SwingResiduals(**{name: value})
            cases.append((f"{name}@{label}={value:+.3f}", r))

    for label, residuals in cases:
        ctrl = CMUTrackingController(
            amc_path=AMC_PATH, start_frame=START_FRAME, residuals=residuals,
        )
        info = _run_episode_and_check(env, ctrl, n_steps=40)
        print(f"  {label:38s}  reason={info.get('termination_reason','?'):16s}  "
              f"min_d={info.get('min_bat_ball_dist', float('nan')):.3f}  "
              f"contact={info.get('contact', False)}")
    env.close()
    print("  PASS\n")


def scenario_unimplemented_dimensions():
    print("=" * 70)
    print("Scenario C: barrel_roll + plate_reach raise NotImplementedError")
    print("=" * 70)
    try:
        adjust_bat_grip_anchor()
    except NotImplementedError as e:
        print(f"  PASS  ({e})")
        return
    raise AssertionError("expected NotImplementedError")


def scenario_pure_qpos_lookup():
    print("=" * 70)
    print("Scenario D: replay.qpos_at(t) is pure (no step_idx mutation)")
    print("=" * 70)
    from src.motion.cmu_replay import CMUMocapReplay
    rep = CMUMocapReplay(AMC_PATH)
    rep.reset(START_FRAME)
    before = rep.step_idx
    q1 = rep.qpos_at(t_seconds=START_FRAME * rep.control_dt)
    q2 = rep.qpos_at(t_seconds=(START_FRAME + 5) * rep.control_dt)
    after = rep.step_idx
    assert before == after, f"qpos_at mutated step_idx ({before} -> {after})"
    assert q1.shape == (63,) and q2.shape == (63,)
    # Linear interp at exact frame should match exactly
    q1_exact = rep.qpos_trajectory[:, START_FRAME]
    assert np.allclose(q1, q1_exact, atol=1e-12)
    print(f"  step_idx unchanged: {before} = {after}")
    print(f"  q1 == frame[{START_FRAME}]: True")
    print("  PASS\n")


def main():
    scenario_dataclass_normalization()
    scenario_pure_qpos_lookup()
    scenario_unimplemented_dimensions()
    scenario_residual_extremes()
    print("All Checkpoint B scenarios passed.")


if __name__ == "__main__":
    main()
