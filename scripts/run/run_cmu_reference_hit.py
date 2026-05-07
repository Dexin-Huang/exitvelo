"""Reference-hit synchronization for the CMU humanoid batting scene.

1. Replay the mocap swing with PD control and root clamping, recording
   bat tip / sweet-spot positions at each step.
2. Find the peak bat-tip speed frame (the swing's contact window).
3. For the MLB arrival speed (33.24 m/s), compute ball start position
   and velocity so the ball arrives at the bat at the right time.
4. Scan configurations to find the best contact.
5. Save the best config to configs/pitch_v1.json.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv

# ── Fixed MLB-scaled pitch parameters ──
MLB_SPEED = 33.24        # m/s (Froude-scaled 85 mph)
CONTROL_DT = 0.01        # 100 Hz
g = 9.81

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")


def get_bat_trajectory(sf, n_steps=80):
    """Replay mocap from *sf* with PD control + root clamping.
    Returns bat tip positions, sweet-spot positions, and tip speeds.
    """
    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=sf)
    env = CMUBattingEnv(render_mode=None, pitch_x=-200)
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

    tips, sweets, speeds = [], [], []
    for _ in range(n_steps):
        if ctrl.done:
            break
        action = ctrl.get_action(obs)
        rp, rq = ctrl.get_root_state()
        env.data.qpos[0:3] = rp
        env.data.qpos[3:7] = rq
        env.data.qvel[0:6] = 0.0
        obs, _, term, trunc, _ = env.step(action)
        tips.append(obs[125:128].copy())
        sweets.append(env.data.site("bat_sweet").xpos.copy())
        speeds.append(float(np.linalg.norm(obs[128:131])))
        if term or trunc:
            break

    env.close()
    return np.array(tips), np.array(sweets), np.array(speeds)


def test_pitch(sf, ball_start, ball_vel, n_steps=200):
    """Run PD-controlled episode with given pitch; return contact info."""
    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=sf)
    env = CMUBattingEnv(
        render_mode=None,
        pitch_x=ball_start[0],
        pitch_y=ball_start[1],
        pitch_height=ball_start[2],
        pitch_velocity=ball_vel.tolist(),
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
    for step in range(n_steps):
        if ctrl.done:
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
        min_dist = min(min_dist, dist)

        if info.get("contact", False):
            sp = info.get("ball_speed_post", 0.0)
            bvel = env.data.qvel[63:66].copy()
            hz = float(np.linalg.norm(bvel[:2]))
            la = float(np.degrees(np.arctan2(bvel[2], hz))) if hz > 0 else 0.0
            env.close()
            return True, float(sp), min_dist, step, float(la)
        if term or trunc:
            break

    env.close()
    return False, 0.0, min_dist, -1, 0.0


def main():
    print("=" * 70)
    print("CMU Humanoid Reference-Hit Synchronization")
    print("=" * 70)

    # ── Step 1: Find best start frames ──
    print("\n[1] Scanning start frames...")
    candidates = []
    for sf in range(350, 450, 10):
        tips, sweets, speeds = get_bat_trajectory(sf, n_steps=60)
        if len(speeds) == 0:
            continue
        peak_step = int(np.argmax(speeds))
        peak_speed = speeds[peak_step]
        if peak_speed > 20.0:
            candidates.append((sf, peak_step, peak_speed, tips, sweets, speeds))
            print(f"  sf={sf}: peak_step={peak_step}  peak={peak_speed:.1f} m/s")

    candidates.sort(key=lambda x: -x[2])
    print(f"  {len(candidates)} candidates with peak > 20 m/s")

    # ── Step 2: Search for contact ──
    print(f"\n[2] Searching for contact (MLB speed = {MLB_SPEED} m/s)...")

    best_result = None
    best_exit = 0.0
    test_count = 0

    # Focus on the top 5 candidates, key arrival steps near peak speed
    for ci, (sf, peak_step, peak_speed, tips, sweets, speeds) in enumerate(candidates[:5]):
        print(f"\n  Candidate {ci+1}: sf={sf}, peak at step {peak_step}")

        # Focus on arrival steps near where the bat is fast
        fast_steps = [s for s in range(len(speeds)) if speeds[s] > peak_speed * 0.6]
        test_steps = sorted(set(fast_steps + list(range(5, min(25, len(tips))))))

        for arrival_step in test_steps:
            arrival_time = arrival_step * CONTROL_DT
            rdist = MLB_SPEED * arrival_time

            for tname, tarr in [("sweet", sweets), ("tip", tips)]:
                if arrival_step >= len(tarr):
                    continue
                target = tarr[arrival_step]
                bx = target[0] - rdist
                by = target[1]
                bz = target[2]
                vz = 0.5 * g * arrival_time

                for dy in [0.0, -0.15, 0.15, -0.3, 0.3]:
                    bs = np.array([bx, by + dy, bz])
                    bv = np.array([
                        MLB_SPEED,
                        -dy / arrival_time if arrival_time > 0 else 0.0,
                        vz,
                    ])
                    test_count += 1
                    contact, sp, md, cs, la = test_pitch(sf, bs, bv)
                    if contact:
                        print(f"    HIT! arr={arrival_step} {tname:5s} "
                              f"rdist={rdist:.2f}m dy={dy:+.2f} "
                              f"exit={sp:.1f} m/s LA={la:.1f}")
                        if sp > best_exit:
                            best_exit = sp
                            best_result = {
                                "start_frame": sf,
                                "ball_start": bs.tolist(),
                                "ball_vel": bv.tolist(),
                                "ball_speed_post": float(sp),
                                "target_pos": target.tolist(),
                                "peak_bat_speed": float(peak_speed),
                                "pitch_distance": float(rdist),
                                "launch_angle": float(la),
                            }

    # ── Step 3: Save config ──
    print(f"\n\nTotal tests: {test_count}")

    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)

    if best_result:
        config = {
            "replay_start_frame": best_result["start_frame"],
            "pitch": {
                "start_pos": best_result["ball_start"],
                "velocity": best_result["ball_vel"],
            },
            "bat_target": best_result["target_pos"],
            "peak_bat_speed": best_result["peak_bat_speed"],
            "contact_achieved": True,
            "mlb_arrival_speed": MLB_SPEED,
            "actual_pitch_distance": best_result["pitch_distance"],
            "results": {
                "exit_velocity_ms": best_result["ball_speed_post"],
                "exit_velocity_mph": best_result["ball_speed_post"] * 2.237,
                "launch_angle_deg": best_result["launch_angle"],
            },
        }
    else:
        sf, peak_step, peak_speed, tips, sweets, speeds = candidates[0]
        target = tips[peak_step]
        rdist = 3.0
        arr_time = rdist / MLB_SPEED
        bx = target[0] - rdist
        vz = 0.5 * g * arr_time
        config = {
            "replay_start_frame": sf,
            "pitch": {
                "start_pos": [float(bx), float(target[1]), float(target[2])],
                "velocity": [MLB_SPEED, 0.0, float(vz)],
            },
            "bat_target": target.tolist(),
            "peak_bat_speed": float(peak_speed),
            "contact_achieved": False,
            "mlb_arrival_speed": MLB_SPEED,
        }

    config_path = configs_dir / "pitch_v1.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if config["contact_achieved"]:
        r = config["results"]
        print(f"  Contact:          YES")
        print(f"  Start frame:      {config['replay_start_frame']}")
        print(f"  Pitch distance:   {config['actual_pitch_distance']:.2f} m")
        print(f"  MLB speed:        {config['mlb_arrival_speed']:.2f} m/s")
        print(f"  Exit velocity:    {r['exit_velocity_ms']:.2f} m/s "
              f"({r['exit_velocity_mph']:.0f} mph)")
        print(f"  Launch angle:     {r['launch_angle_deg']:.1f} deg")
        print(f"  Peak bat speed:   {config['peak_bat_speed']:.1f} m/s")
    else:
        print(f"  Contact:          NO")

    print(f"\n  Config saved: {config_path}")
    print(f"\n{json.dumps(config, indent=2)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
