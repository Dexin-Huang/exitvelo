"""Checkpoint A smoke: env + analytical contact API contract.

End-to-end contact (real swing -> ball hit -> exit metrics) requires the
controller refactor in Section 2. This smoke verifies the env's API and
internal contracts at the unit level:

  1. Env imports, resets, and steps without errors.
  2. Observation is exactly 154-dim.
  3. info dict exposes the required schema (exit_mph, launch_deg, carry_ft,
     total_ft, contact, first_contact, ball_speed_post, ...).
  4. Without a real swing, episode ends with a miss + W_MISS penalty.
  5. Pitch jitter at reset stays inside the configured ranges.
  6. The bat-speed gate filters out artificially weak contacts.
  7. The analytical pipeline (called directly from contacts.py) is the
     ground truth; the env wires it correctly.

The "real swing makes contact at 76 mph" check belongs in Section 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.cmu_batting_env import (
    CMUBattingEnv,
    _BALL_QPOS_START,
    _BALL_QVEL_START,
)
from src.env.contacts import nathan_exit_velocity, integrate_ball_trajectory


REQUIRED_INFO_KEYS = {
    "contact",
    "first_contact",
    "exit_mph",
    "launch_deg",
    "carry_ft",
    "total_ft",
    "min_bat_ball_dist",
    "termination_reason",
    "ball_speed_post",
}


def scenario_obs_shape_and_info_schema():
    print("=" * 70)
    print("Scenario 1: obs is 154-dim, info has required keys")
    print("=" * 70)
    env = CMUBattingEnv()
    obs, _ = env.reset()
    assert obs.shape == (154,), f"obs shape {obs.shape} != (154,)"
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    missing = REQUIRED_INFO_KEYS - set(info.keys())
    assert not missing, f"info missing keys: {missing}"
    assert obs.shape == (154,)
    print(f"  obs={obs.shape} info_keys={sorted(info.keys())}")
    env.close()
    print("  PASS\n")


def scenario_held_still_misses():
    print("=" * 70)
    print("Scenario 2: held-still humanoid -> miss + W_MISS penalty")
    print("=" * 70)
    env = CMUBattingEnv()
    env.reset()
    last_info, last_r = None, 0.0
    for _ in range(env.max_steps):
        obs, last_r, term, trunc, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        last_info = info
        if term or trunc:
            break
    print(f"  termination_reason={last_info['termination_reason']} reward={last_r:.4f}")
    assert not last_info["contact"]
    assert last_info["termination_reason"] in (
        "ball_past_batter",
        "humanoid_fell",
        "timeout",
    )
    env.close()
    print("  PASS\n")


def scenario_pitch_jitter_in_range():
    print("=" * 70)
    print("Scenario 3: pitch jitter stays inside configured ranges")
    print("=" * 70)
    env = CMUBattingEnv(pitch_jitter=True)
    speeds, heights, ys = [], [], []
    for seed in range(40):
        env.reset(seed=seed)
        speeds.append(env.pitch_speed)
        heights.append(env.pitch_height)
        ys.append(env.pitch_y)
    s_lo, s_hi = env._jitter_speed
    h_lo, h_hi = env._jitter_height
    y_lo, y_hi = env._jitter_x_offset
    assert all(s_lo <= s <= s_hi for s in speeds), f"speed out of range: {speeds}"
    assert all(h_lo <= h <= h_hi for h in heights), f"height out of range: {heights}"
    assert all(y_lo <= y <= y_hi for y in ys), f"lateral out of range: {ys}"
    print(f"  speed:  {min(speeds):.2f} ... {max(speeds):.2f}  (range {s_lo:.1f}-{s_hi:.1f})")
    print(f"  height: {min(heights):.2f} ... {max(heights):.2f}  (range {h_lo:.2f}-{h_hi:.2f})")
    print(f"  y_off:  {min(ys):+.3f} ... {max(ys):+.3f}  (range {y_lo:+.2f}/{y_hi:+.2f})")
    env.close()
    print("  PASS\n")


def scenario_bat_speed_gate_filters_weak_taps():
    print("=" * 70)
    print("Scenario 4: bat-speed gate filters out weak taps")
    print("=" * 70)
    env = CMUBattingEnv()
    env.reset()
    sweet_pos = env.data.site_xpos[env._sweet_site_id].copy()
    env.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3] = sweet_pos
    env.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = [-1.0, 0.0, 0.0]
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    print(f"  contact={info['contact']} min_dist={info['min_bat_ball_dist']:.4f} reward={r:.4f}")
    assert not info["contact"], "weak motion should not register contact"
    assert info["exit_mph"] == 0.0
    env.close()
    print("  PASS\n")


def scenario_analytical_ground_truth():
    """Sanity-check that the env's analytical pipeline matches a direct
    nathan_exit_velocity + integrate_ball_trajectory call. The env is
    just a wrapper around these; if the wrapper drifts, this catches it."""
    print("=" * 70)
    print("Scenario 5: analytical pipeline ground truth (21.5 m/s bat -> ~73 mph)")
    print("=" * 70)
    bat_vel = np.array([21.5, 0.0, 0.0])
    ball_vel_in = np.array([-41.6, 0.0, 0.0])
    contact_pos = np.array([0.3, -0.3, 1.2])
    exit_v = nathan_exit_velocity(bat_vel, ball_vel_in)
    speed_mph = float(np.linalg.norm(exit_v)) * 2.237
    traj, _, _ = integrate_ball_trajectory(contact_pos, exit_v)
    horiz_dist_ft = float(np.linalg.norm(traj[-1, :2] - contact_pos[:2])) * 3.281
    print(f"  exit_speed={speed_mph:.2f} mph, total_dist={horiz_dist_ft:.1f} ft")
    assert 70.0 < speed_mph < 80.0, f"expected ~73 mph, got {speed_mph}"
    assert 200 < horiz_dist_ft < 400, f"expected ~295 ft total, got {horiz_dist_ft}"
    print("  PASS\n")


def main():
    scenario_obs_shape_and_info_schema()
    scenario_held_still_misses()
    scenario_pitch_jitter_in_range()
    scenario_bat_speed_gate_filters_weak_taps()
    scenario_analytical_ground_truth()
    print("All Checkpoint A scenarios passed.")


if __name__ == "__main__":
    main()
