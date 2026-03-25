"""Reference-hit synchronisation with a fixed MLB-scaled fastball.

The pitch has fixed, deterministic physics every time:
  - Speed:    33.24 m/s (Froude-scaled 85 mph)
  - Distance: 12.82m (scaled mound-to-plate)
  - Height:   released from 1.40m (scaled pitcher release)

The pitch arrives at wherever the bat sweeps — "middle-middle" is defined
relative to the batter's actual swing plane, not a fixed coordinate.

Search: find the mocap start frame that lines up the swing with the ball.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from src.env.batting_env import (
    BattingEnv,
    _BALL_QPOS_START,
    _BALL_QVEL_START,
    _QPOS_JOINT_END,
    _QPOS_JOINT_START,
    _QVEL_JOINT_END,
)
from src.env.contacts import detect_bat_ball_contact

CONFIG_DIR = _PROJECT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_PATH = _PROJECT / "data" / "processed" / "swing_124_07.npz"

CONTROL_DT = 0.002 * 5          # 0.01 s
CONTROL_FPS = 1.0 / CONTROL_DT  # 100 Hz
GRAVITY = np.array([0.0, 0.0, -9.81])

# ── Fixed MLB-scaled pitch physics ──
SCALE = 1.4 / 1.83
RELEASE_DIST = 16.76 * SCALE       # 12.82m
RELEASE_HEIGHT = 1.83 * SCALE      # 1.40m
ARRIVAL_SPEED = 38.0 * np.sqrt(SCALE)  # 33.24 m/s


def _load_traj():
    d = np.load(TRAJECTORY_PATH)
    return d["joint_targets"], d["root_pos"], d["root_quat"], float(d["fps"])


def _pd_bat_trajectory(env, jt, rp, rq, fps, start_mocap, n_steps=100):
    """PD-controlled root-clamped replay. Returns (n_steps, 3) bat-tip pos."""
    n_mocap = len(jt)
    mf0 = min(start_mocap, n_mocap - 1)
    env.reset()
    env.data.qpos[0:3] = rp[mf0]
    env.data.qpos[3:7] = rq[mf0]
    env.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END] = jt[mf0]
    mujoco.mj_forward(env.model, env.data)

    positions = np.zeros((n_steps, 3))
    for step in range(n_steps):
        mf = min(start_mocap + int(step * CONTROL_FPS / fps), n_mocap - 1)
        env.data.qpos[0:3] = rp[mf]
        env.data.qpos[3:7] = rq[mf]
        env.data.qvel[0:3] = 0.0
        env.data.qvel[3:6] = 0.0
        obs, _, _, _, _ = env.step(jt[mf].copy())
        positions[step] = obs[47:50]
    return positions


def _solve_pitch_to_target(target_pos):
    """Compute ball start pos and velocity for the fixed MLB pitch to reach target_pos.

    The ball starts RELEASE_DIST behind the target in x, at RELEASE_HEIGHT,
    at the same y as the target. Speed is fixed at ARRIVAL_SPEED.
    """
    t_flight = RELEASE_DIST / ARRIVAL_SPEED
    start = np.array([
        target_pos[0] - RELEASE_DIST,
        target_pos[1],
        RELEASE_HEIGHT,
    ])
    vel = (target_pos - start - 0.5 * GRAVITY * t_flight**2) / t_flight
    return start, vel, t_flight


def _test_contact(env, jt, rp, rq, fps, ball_start, ball_vel,
                  start_mocap, max_steps=100):
    """Test swing with ball. Returns (contact, step, exit_speed, launch_angle, min_dist)."""
    n_mocap = len(jt)
    env.reset()

    # Set ball
    env.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START+3] = ball_start
    env.data.qpos[_BALL_QPOS_START+3:_BALL_QPOS_START+7] = [1, 0, 0, 0]
    env.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START+3] = ball_vel
    env.data.qvel[_BALL_QVEL_START+3:_BALL_QVEL_START+6] = [0, 0, 0]

    # Set humanoid
    mf0 = min(start_mocap, n_mocap - 1)
    env.data.qpos[0:3] = rp[mf0]
    env.data.qpos[3:7] = rq[mf0]
    env.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END] = jt[mf0]
    mujoco.mj_forward(env.model, env.data)

    min_dist = float("inf")
    for step in range(max_steps):
        mf = min(start_mocap + int(step * CONTROL_FPS / fps), n_mocap - 1)
        env.data.qpos[0:3] = rp[mf]
        env.data.qpos[3:7] = rq[mf]
        env.data.qvel[0:3] = 0.0
        env.data.qvel[3:6] = 0.0
        obs, _, terminated, truncated, info = env.step(jt[mf].copy())

        bat_p, ball_p = obs[47:50], obs[53:56]
        d = float(np.linalg.norm(bat_p - ball_p))
        min_dist = min(min_dist, d)

        if info.get("contact", False):
            bv = env.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START+3].copy()
            exit_speed = float(np.linalg.norm(bv))
            horiz = float(np.linalg.norm(bv[:2]))
            launch_angle = float(np.degrees(np.arctan2(bv[2], horiz))) if exit_speed > 0 else 0
            return True, step, exit_speed, launch_angle, min_dist
        if terminated or truncated:
            break

    return False, None, 0.0, 0.0, min_dist


def main():
    print("=" * 60)
    print("Reference-Hit Sync (Fixed MLB Fastball)")
    print("=" * 60)
    print(f"  Pitch speed:    {ARRIVAL_SPEED:.2f} m/s ({ARRIVAL_SPEED*2.237:.1f} mph)")
    print(f"  Release dist:   {RELEASE_DIST:.2f} m")
    print(f"  Release height: {RELEASE_HEIGHT:.2f} m")

    jt, rp, rq, fps = _load_traj()
    n_mocap = len(jt)
    print(f"  Trajectory:     {n_mocap} frames @ {fps} fps\n")

    # Create one reusable env (ball disabled for trajectory scans)
    env = BattingEnv(render_mode=None, frame_skip=5, max_steps=9999,
                     pitch_x=-200, pitch_speed=0, pitch_height=0)

    # ── Phase 1: Find bat sweep zone for each candidate start frame ──
    print("Phase 1: Scanning bat trajectories ...")
    best = None
    total_contacts = 0

    for start_mocap in range(380, min(n_mocap - 50, 800), 5):
        # Get bat tip trajectory under PD control
        bat_pos = _pd_bat_trajectory(env, jt, rp, rq, fps, start_mocap, n_steps=80)
        vel = np.diff(bat_pos, axis=0) / CONTROL_DT
        speed = np.linalg.norm(vel, axis=1)
        pk = int(np.argmax(speed))
        pk_speed = float(speed[pk])

        if pk_speed < 5.0:
            continue

        # Try aiming the ball at bat-tip positions around the peak
        for offset in range(-8, 8, 2):
            target_idx = pk + 1 + offset
            if target_idx < 2 or target_idx >= len(bat_pos):
                continue
            target_pos = bat_pos[target_idx]
            target_step = target_idx  # ball should arrive at this ctrl step

            # Flight time must match: ball arrives at target_step * CONTROL_DT
            t_flight_needed = target_step * CONTROL_DT
            if t_flight_needed < 0.1:
                continue

            # Solve pitch from MLB distance
            p_start, v_ball, t_flight_mlb = _solve_pitch_to_target(target_pos)
            ball_speed = float(np.linalg.norm(v_ball))

            # Ball should travel at roughly MLB speed (allow some tolerance for vz)
            if ball_speed > 50 or ball_speed < 15:
                continue

            # Test with the actual flight time from MLB distance
            c, cs, exit_spd, la, md = _test_contact(
                env, jt, rp, rq, fps, p_start, v_ball, start_mocap, 80)

            if c:
                total_contacts += 1
                if best is None or exit_spd > best["exit_speed"]:
                    best = {
                        "start_mocap": start_mocap,
                        "start_pos": p_start.tolist(),
                        "velocity": v_ball.tolist(),
                        "ball_speed": ball_speed,
                        "contact_step": cs,
                        "exit_speed": exit_spd,
                        "launch_angle": la,
                        "t_flight": t_flight_mlb,
                    }
                if total_contacts <= 8:
                    print(f"  HIT: frame={start_mocap}, step={cs}, "
                          f"exit_vel={exit_spd:.1f} m/s ({exit_spd*2.237:.0f} mph), "
                          f"launch={la:.1f}°, pitch_spd={ball_speed:.1f} m/s")

            if total_contacts >= 40:
                break
        if total_contacts >= 40:
            break

    env.close()

    # ── Save ──
    print(f"\n{'='*60}")
    print(f"Total contacts found: {total_contacts}")

    if best is not None:
        bv = np.array(best["velocity"])
        print(f"BEST CONFIG:")
        print(f"  Mocap start:    frame {best['start_mocap']}")
        print(f"  Ball start:     {best['start_pos']}")
        print(f"  Ball velocity:  [{bv[0]:.3f}, {bv[1]:.3f}, {bv[2]:.3f}]")
        print(f"  Pitch speed:    {best['ball_speed']:.2f} m/s ({best['ball_speed']*2.237:.1f} mph)")
        print(f"  Exit velocity:  {best['exit_speed']:.2f} m/s ({best['exit_speed']*2.237:.1f} mph)")
        print(f"  Launch angle:   {best['launch_angle']:.1f}°")

        cfg = {
            "pitch": {
                "start_pos": best["start_pos"],
                "velocity": best["velocity"],
            },
            "replay": {
                "replay_start_mocap": best["start_mocap"],
                "contact_step": best["contact_step"],
                "t_flight": best["t_flight"],
            },
            "mlb_scaled": {
                "scale_factor": float(SCALE),
                "release_dist_m": float(RELEASE_DIST),
                "release_height_m": float(RELEASE_HEIGHT),
                "arrival_speed_ms": float(ARRIVAL_SPEED),
                "arrival_speed_mph": float(ARRIVAL_SPEED * 2.237),
            },
            "results": {
                "exit_velocity_ms": best["exit_speed"],
                "exit_velocity_mph": best["exit_speed"] * 2.237,
                "launch_angle_deg": best["launch_angle"],
                "pitch_speed_ms": best["ball_speed"],
                "pitch_speed_mph": best["ball_speed"] * 2.237,
            },
        }
    else:
        print("NO CONTACT — using default MLB pitch params")
        default_start = np.array([-RELEASE_DIST, 0.0, RELEASE_HEIGHT])
        default_vel = np.array([ARRIVAL_SPEED, 0.0, 0.0])
        cfg = {
            "pitch": {
                "start_pos": default_start.tolist(),
                "velocity": default_vel.tolist(),
            },
            "replay": {"replay_start_mocap": 410},
            "mlb_scaled": {
                "scale_factor": float(SCALE),
                "release_dist_m": float(RELEASE_DIST),
                "release_height_m": float(RELEASE_HEIGHT),
                "arrival_speed_ms": float(ARRIVAL_SPEED),
            },
            "results": {"note": "No contact achieved."},
        }

    cfg_path = CONFIG_DIR / "pitch_v1.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to: {cfg_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
