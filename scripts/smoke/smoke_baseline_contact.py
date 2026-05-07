"""Smoke test: does the saved milestone config still produce contact in the
current env/controller code path?

If YES, we have a known-good baseline to anchor RL training.
If NO, we need to either (a) re-run the search to find a new config, or
(b) accept that env code drift broke the milestone and audit recent changes.

Loads results/final/cmu_milestone_results.json (76 mph result) and replays it
with full step-by-step logging: tip-to-ball distance, ball position, contact
state, termination reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
MILESTONE = PROJECT_ROOT / "results" / "final" / "cmu_milestone_results.json"


def main():
    blob = json.loads(MILESTONE.read_text())
    cfg = blob["pitch_config"] if "pitch_config" in blob else blob
    sf = cfg["replay_start_frame"]
    ball_start = cfg["pitch"]["start_pos"]
    ball_vel = cfg["pitch"]["velocity"]

    print("=" * 70)
    print("Smoke test: milestone config")
    print("=" * 70)
    print(f"  start_frame:   {sf}")
    print(f"  ball_start:    [{ball_start[0]:+.3f}, {ball_start[1]:+.3f}, {ball_start[2]:+.3f}]")
    print(f"  ball_velocity: [{ball_vel[0]:+.3f}, {ball_vel[1]:+.3f}, {ball_vel[2]:+.3f}]")
    print(f"  saved exit:    {cfg['results']['exit_velocity_ms']:.2f} m/s "
          f"({cfg['results']['exit_velocity_mph']:.1f} mph)")
    print(f"  saved launch:  {cfg['results']['launch_angle_deg']:.1f} deg")
    print()

    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=sf)
    env = CMUBattingEnv(
        render_mode=None,
        pitch_x=float(ball_start[0]),
        pitch_y=float(ball_start[1]),
        pitch_height=float(ball_start[2]),
        pitch_velocity=ball_vel,
    )
    obs, _ = env.reset()

    pos, quat = ctrl.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    action = ctrl.get_action(obs)
    env.data.qpos[7:63] = action
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    ctrl.reset()

    min_dist = 1e9
    min_dist_step = -1
    print(f"  {'step':>4} {'tip_x':>7} {'tip_y':>7} {'tip_z':>7}   {'ball_x':>7} {'ball_y':>7} {'ball_z':>7}   {'dist':>6}")
    for step in range(200):
        if ctrl.done:
            print("  controller exhausted")
            break
        action = ctrl.get_action(obs)
        rp, rq = ctrl.get_root_state()
        env.data.qpos[0:3] = rp
        env.data.qpos[3:7] = rq
        env.data.qvel[0:6] = 0.0
        obs, _, term, trunc, info = env.step(action)

        tip_p = obs[125:128]
        ball_p = obs[131:134]
        dist = float(np.linalg.norm(ball_p - tip_p))
        if dist < min_dist:
            min_dist = dist
            min_dist_step = step
        if step % 2 == 0 or info.get("contact", False) or term:
            print(f"  {step:>4d} "
                  f"{tip_p[0]:+7.3f} {tip_p[1]:+7.3f} {tip_p[2]:+7.3f}   "
                  f"{ball_p[0]:+7.3f} {ball_p[1]:+7.3f} {ball_p[2]:+7.3f}   "
                  f"{dist:6.3f}")
        if info.get("contact", False):
            print(f"\n  CONTACT at step {step}")
            print(f"  ball_speed_post = {info.get('ball_speed_post', 0.0):.2f} m/s")
            print(f"  termination_reason = {info.get('termination_reason', '?')}")
            env.close()
            return
        if term or trunc:
            print(f"\n  TERMINATED at step {step}")
            print(f"  reason = {info.get('termination_reason', '?')}")
            print(f"  min_tip_to_ball_dist = {min_dist:.3f} m at step {min_dist_step}")
            env.close()
            return
    env.close()
    print(f"\n  loop ended without contact or termination")
    print(f"  min_tip_to_ball_dist = {min_dist:.3f} m at step {min_dist_step}")


if __name__ == "__main__":
    main()
