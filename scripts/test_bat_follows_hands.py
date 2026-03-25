"""Test: bat follows both hands during mocap replay.

Creates the CMU batting env, replays mocap with both arms driven by mocap,
positions the bat between both hands each frame, renders a GIF, and opens it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController, compute_bat_state
from src.env.cmu_batting_env import CMUBattingEnv

AMC_PATH = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"


def main():
    print("=" * 60)
    print("Test: Bat Follows Both Hands")
    print("=" * 60)

    # Load pitch config
    config_path = PROJECT_ROOT / "configs" / "pitch_v1.json"
    with open(config_path) as f:
        config = json.load(f)

    start_frame = config["replay_start_frame"]
    pitch = config["pitch"]

    print(f"\nConfig: start_frame={start_frame}")
    print(f"  Ball start: {pitch['start_pos']}")
    print(f"  Ball velocity: {pitch['velocity']}")

    # 1. Create environment (no ball for this test -- focus on bat positioning)
    print("\n[1] Creating CMU batting environment...")
    env = CMUBattingEnv(
        render_mode="rgb_array",
        frame_skip=5,
        max_steps=500,
        pitch_x=-100.0,   # ball far away so it doesn't interfere
        pitch_y=0.0,
        pitch_height=1.0,
        pitch_velocity=[0.0, 0.0, 0.0],
    )

    print(f"  Model: nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

    # 2. Create tracking controller with env reference
    print("\n[2] Creating tracking controller...")
    controller = CMUTrackingController(
        amc_path=str(AMC_PATH),
        start_frame=start_frame,
        control_dt=0.01,
        env=env,
    )

    # 3. Reset and replay with bat positioning
    print("\n[3] Replaying mocap with bat following hands...")
    obs, _ = env.reset()
    controller.reset()

    # Set initial humanoid pose
    qpos_init = controller.get_full_qpos()
    env.set_humanoid_qpos(qpos_init)
    mujoco.mj_forward(env.model, env.data)

    # Position bat between hands
    controller.update_bat()
    mujoco.mj_forward(env.model, env.data)

    controller.reset()

    # Print initial hand/bat info
    lhand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'lhand')
    rhand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'rhand')
    bat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'bat')

    lhand_pos = env.data.xpos[lhand_id]
    rhand_pos = env.data.xpos[rhand_id]
    bat_pos = env.data.xpos[bat_id]
    hand_dist = np.linalg.norm(rhand_pos - lhand_pos)

    print(f"  lhand pos: [{lhand_pos[0]:.3f}, {lhand_pos[1]:.3f}, {lhand_pos[2]:.3f}]")
    print(f"  rhand pos: [{rhand_pos[0]:.3f}, {rhand_pos[1]:.3f}, {rhand_pos[2]:.3f}]")
    print(f"  bat pos:   [{bat_pos[0]:.3f}, {bat_pos[1]:.3f}, {bat_pos[2]:.3f}]")
    print(f"  hand distance: {hand_dist:.3f}m")

    # 4. Run replay loop
    frames = []
    n_steps = 200  # ~2 seconds of motion
    render_every = 2

    for step in range(n_steps):
        if controller.done:
            print(f"  Mocap done at step {step}")
            break

        # Get mocap action and apply
        action = controller.get_action(obs)

        # Clamp root to mocap
        pos, quat = controller.get_root_state()
        env.data.qpos[0:3] = pos
        env.data.qpos[3:7] = quat
        env.data.qvel[0:3] = 0.0
        env.data.qvel[3:6] = 0.0

        # Step physics with PD control
        obs, _, terminated, truncated, info = env.step(action)

        # Update bat to follow hands
        mujoco.mj_forward(env.model, env.data)
        controller.update_bat()
        mujoco.mj_forward(env.model, env.data)

        # Render
        if step % render_every == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Print diagnostics occasionally
        if step % 50 == 0:
            lhand_p = env.data.xpos[lhand_id]
            rhand_p = env.data.xpos[rhand_id]
            bat_p = env.data.xpos[bat_id]
            hd = np.linalg.norm(rhand_p - lhand_p)
            print(f"  Step {step:3d}: lhand=[{lhand_p[0]:.2f},{lhand_p[1]:.2f},{lhand_p[2]:.2f}] "
                  f"rhand=[{rhand_p[0]:.2f},{rhand_p[1]:.2f},{rhand_p[2]:.2f}] "
                  f"bat=[{bat_p[0]:.2f},{bat_p[1]:.2f},{bat_p[2]:.2f}] "
                  f"hand_dist={hd:.3f}")

        if info.get("contact"):
            print(f"  BAT-BALL CONTACT at step {step}!")

        if terminated or truncated:
            print(f"  Episode ended: {info.get('termination_reason', 'unknown')}")
            break

    print(f"\n  Rendered {len(frames)} frames")

    # 5. Save GIF
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    gif_path = results_dir / "cmu_ik_swing.gif"

    print(f"\n[4] Saving GIF to {gif_path}...")
    if len(frames) > 0:
        # Subsample if too many frames
        if len(frames) > 120:
            step_size = max(1, len(frames) // 100)
            sub_frames = frames[::step_size]
        else:
            sub_frames = frames
        imageio.mimsave(str(gif_path), sub_frames, duration=80, loop=0)
        size_kb = os.path.getsize(gif_path) / 1024
        print(f"  Saved! ({len(sub_frames)} frames, {size_kb:.0f} KB)")
    else:
        print("  ERROR: No frames rendered!")
        env.close()
        return

    env.close()

    # 6. Open the GIF
    print(f"\n[5] Opening GIF...")
    subprocess.Popen(["cmd", "/c", "start", "", str(gif_path)],
                     creationflags=subprocess.CREATE_NO_WINDOW)

    print("\nDone!")


if __name__ == "__main__":
    main()
