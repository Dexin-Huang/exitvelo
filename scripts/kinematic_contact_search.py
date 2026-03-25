"""Search for ball trajectory that produces contact using kinematic humanoid replay.

Sets humanoid qpos directly from mocap (no PD oscillation) while letting
the ball move through physics.  This is the most reliable way to find
contact because the bat follows the exact mocap trajectory.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv
from src.env.contacts import detect_bat_ball_contact
from src.motion.cmu_replay import CMUMocapReplay

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
g = 9.81
dt = 0.01  # 100 Hz control


def precompute_bat_trajectory():
    """Compute bat tip/sweet/mid positions for all mocap frames."""
    replay = CMUMocapReplay(AMC_PATH, control_dt=dt)
    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=0)

    env = CMUBattingEnv(render_mode=None, pitch_x=-200)
    env.reset()

    all_tip = []
    all_sweet = []
    all_mid = []

    for f in range(replay.num_frames):
        ctrl.replay.step_idx = f
        qpos_full = ctrl.get_full_qpos()
        env.data.qpos[:63] = qpos_full
        env.data.qpos[63] = 0.0
        mujoco.mj_forward(env.model, env.data)

        all_tip.append(env.data.site("bat_tip").xpos.copy())
        all_sweet.append(env.data.site("bat_sweet").xpos.copy())
        all_mid.append(env.data.site("bat_mid").xpos.copy())

    env.close()

    all_tip = np.array(all_tip)
    all_sweet = np.array(all_sweet)
    all_mid = np.array(all_mid)

    # Tip speeds
    tip_vels = np.diff(all_tip, axis=0) / dt
    tip_speeds = np.linalg.norm(tip_vels, axis=1)

    return all_tip, all_sweet, all_mid, tip_speeds, replay.num_frames


def test_kinematic_pitch(start_frame, arrival_frame, target_pos,
                         rdist, dy=0.0, dz=0.0, n_frames_total=None):
    """Test a pitch configuration with kinematic humanoid replay."""
    n_steps_to_arrival = arrival_frame - start_frame
    arrival_time = n_steps_to_arrival * dt

    if arrival_time <= 0.001:
        return False, 0.0, 1e9, -1

    flight_time = arrival_time
    vx = rdist / flight_time
    target_y = target_pos[1] + dy
    target_z = target_pos[2] + dz
    bx = target_pos[0] - rdist
    by = target_y
    bz = target_z
    vz = 0.5 * g * flight_time  # compensate gravity

    ball_start = np.array([bx, by, bz])
    ball_vel = np.array([vx, 0.0, vz])

    env = CMUBattingEnv(render_mode=None, pitch_x=-200, max_steps=500)
    env.reset()

    # Set ball
    env.data.qpos[64:67] = ball_start
    env.data.qpos[67:71] = [1, 0, 0, 0]
    env.data.qvel[63:66] = ball_vel
    env.data.qvel[66:69] = [0, 0, 0]

    # Set initial humanoid pose
    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=start_frame)
    qpos_full = ctrl.get_full_qpos()
    env.data.qpos[:63] = qpos_full
    env.data.qpos[63] = 0.0
    mujoco.mj_forward(env.model, env.data)

    min_dist = 1e9
    max_frame = n_frames_total if n_frames_total else (arrival_frame + 50)
    run_steps = min(max_frame - start_frame, 200)

    for step in range(run_steps):
        current_frame = start_frame + step
        if current_frame >= (n_frames_total or 10000):
            break

        # Set humanoid kinematically
        ctrl.replay.step_idx = current_frame
        qpos_full = ctrl.get_full_qpos()
        env.data.qpos[:63] = qpos_full
        env.data.qpos[63] = 0.0

        # Zero humanoid velocities (kinematic)
        env.data.qvel[:63] = 0.0

        # Step physics (ball moves, contact detection)
        mujoco.mj_step(env.model, env.data)

        # Check contact
        contact, contact_info = detect_bat_ball_contact(env.model, env.data)

        # Ball-bat distances
        tip_p = env.data.site("bat_tip").xpos
        sweet_p = env.data.site("bat_sweet").xpos
        mid_p = env.data.site("bat_mid").xpos
        ball_p = env.data.qpos[64:67]

        d_tip = float(np.linalg.norm(ball_p - tip_p))
        d_sweet = float(np.linalg.norm(ball_p - sweet_p))
        d_mid = float(np.linalg.norm(ball_p - mid_p))
        d_min = min(d_tip, d_sweet, d_mid)
        min_dist = min(min_dist, d_min)

        if contact:
            ball_vel_post = env.data.qvel[63:66].copy()
            speed_post = float(np.linalg.norm(ball_vel_post))
            env.close()
            return True, speed_post, min_dist, step

        # Ball gone past
        if ball_p[0] > 3.0:
            break

    env.close()
    return False, 0.0, min_dist, -1


def main():
    print("=" * 70)
    print("Kinematic Contact Search")
    print("=" * 70)

    # Step 1: Precompute bat trajectory
    print("\n[1] Precomputing bat trajectory...")
    all_tip, all_sweet, all_mid, tip_speeds, n_frames = precompute_bat_trajectory()
    peak_frame = int(np.argmax(tip_speeds))
    print(f"    Peak tip speed: {tip_speeds[peak_frame]:.2f} m/s at frame {peak_frame}")
    print(f"    Tip at peak: {all_tip[peak_frame]}")
    print(f"    Sweet at peak: {all_sweet[peak_frame]}")

    # Find all frames with high speed
    fast_frames = np.where(tip_speeds > 15.0)[0]
    print(f"    Frames with speed > 15 m/s: {len(fast_frames)}")

    # Step 2: Search for contact
    print("\n[2] Searching for contact...")

    best_result = None
    best_min_dist = 1e9
    test_count = 0

    # Define swing regions: clusters of fast frames
    swing_regions = []
    if len(fast_frames) > 0:
        # Cluster consecutive frames
        clusters = []
        current_cluster = [fast_frames[0]]
        for i in range(1, len(fast_frames)):
            if fast_frames[i] - fast_frames[i - 1] <= 5:
                current_cluster.append(fast_frames[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [fast_frames[i]]
        clusters.append(current_cluster)

        for cl in clusters:
            swing_regions.append((max(0, cl[0] - 5), min(n_frames - 1, cl[-1] + 5)))
            print(f"    Swing region: frames {cl[0]}-{cl[-1]}")

    # Search each swing region
    for region_start, region_end in swing_regions:
        print(f"\n  Searching frames {region_start}-{region_end}...")

        for arrival_frame in range(region_start, region_end + 1):
            # Target all three bat sections
            targets = [
                ("tip", all_tip[arrival_frame]),
                ("sweet", all_sweet[arrival_frame]),
                ("mid", all_mid[arrival_frame]),
            ]

            for target_name, target_pos in targets:
                for rdist in [5.0, 3.0, 2.0, 1.0, 0.5]:
                    for n_steps_ahead in [15, 20, 25, 30, 40, 50]:
                        start_frame = arrival_frame - n_steps_ahead
                        if start_frame < 0:
                            continue

                        for dy in [0.0, -0.1, 0.1]:
                            test_count += 1
                            contact, sp, md, cs = test_kinematic_pitch(
                                start_frame, arrival_frame, target_pos,
                                rdist, dy=dy, dz=0.0, n_frames_total=n_frames,
                            )

                            if md < best_min_dist:
                                best_min_dist = md

                            if contact:
                                # Recompute ball params for saving
                                n_to_arr = arrival_frame - start_frame
                                arr_time = n_to_arr * dt
                                vx = rdist / arr_time
                                bx = target_pos[0] - rdist
                                by = target_pos[1] + dy
                                bz = target_pos[2]
                                vz = 0.5 * g * arr_time

                                print(
                                    f"  CONTACT! af={arrival_frame} sf={start_frame} "
                                    f"{target_name} rdist={rdist} dy={dy} "
                                    f"sp={sp:.2f} step={cs}"
                                )

                                if best_result is None or sp > best_result["ball_speed_post"]:
                                    best_result = {
                                        "start_frame": start_frame,
                                        "arrival_frame": arrival_frame,
                                        "ball_start": [bx, by, bz],
                                        "ball_vel": [vx, 0.0, vz],
                                        "ball_speed_post": sp,
                                        "target_pos": target_pos.tolist(),
                                        "target_section": target_name,
                                        "peak_bat_speed": float(
                                            tip_speeds[min(arrival_frame, len(tip_speeds) - 1)]
                                        ),
                                    }

            if arrival_frame % 5 == 0 and arrival_frame > region_start:
                status = "FOUND" if best_result else f"min_d={best_min_dist:.4f}"
                print(f"    frame {arrival_frame}: tests={test_count} {status}")

        if best_result is not None:
            # Found contact in this region, continue to search more
            pass

    # Step 3: Save config
    print(f"\n\nTotal tests: {test_count}")
    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)

    if best_result:
        print(f"\nBEST CONTACT:")
        print(f"  Start frame: {best_result['start_frame']}")
        print(f"  Arrival frame: {best_result['arrival_frame']}")
        print(f"  Ball speed post: {best_result['ball_speed_post']:.2f} m/s")
        print(f"  Target section: {best_result['target_section']}")
        print(f"  Target pos: {best_result['target_pos']}")

        config = {
            "replay_start_frame": best_result["start_frame"],
            "pitch": {
                "start_pos": best_result["ball_start"],
                "velocity": best_result["ball_vel"],
                "ball_speed_post": best_result["ball_speed_post"],
            },
            "bat_target": best_result["target_pos"],
            "peak_bat_speed": best_result["peak_bat_speed"],
            "contact_achieved": True,
            "mlb_arrival_speed": 33.24,
            "notes": "Found via kinematic contact search",
        }
        with open(configs_dir / "pitch_v1.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved to configs/pitch_v1.json")
    else:
        print(f"\nNo contact found. Best min distance: {best_min_dist:.4f}")

        # Save a fallback config using the highest speed frame
        top_frame = peak_frame
        target = all_sweet[top_frame]
        rdist = 5.0
        sf = max(0, top_frame - 30)
        arr_time = (top_frame - sf) * dt
        vx = rdist / arr_time
        bx = target[0] - rdist
        by = target[1]
        bz = target[2]
        vz = 0.5 * g * arr_time

        config = {
            "replay_start_frame": sf,
            "pitch": {
                "start_pos": [bx, by, bz],
                "velocity": [vx, 0.0, vz],
                "ball_speed_post": 0.0,
            },
            "bat_target": target.tolist(),
            "peak_bat_speed": float(tip_speeds[top_frame]),
            "contact_achieved": False,
            "best_min_distance": float(best_min_dist),
            "mlb_arrival_speed": 33.24,
            "notes": "Fallback config (no contact achieved)",
        }
        with open(configs_dir / "pitch_v1.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved fallback config to configs/pitch_v1.json")


if __name__ == "__main__":
    main()
