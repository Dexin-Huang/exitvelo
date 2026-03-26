"""Visualize the batting swing with strike zone, batter's box, and pitch trajectory.

Replays the mocap swing, shows the ball coming in, and renders GIFs from
multiple camera angles.
"""

import json
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv
from src.motion.cmu_replay import CMUMocapReplay

AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
CONTROL_DT = 0.01


def load_pitch_config():
    config_path = PROJECT_ROOT / "configs" / "pitch_v1.json"
    with open(config_path) as f:
        return json.load(f)


def render_full_swing(config, camera_name, output_path, max_steps=300):
    """Render the full swing from well before the contact point.

    Starts the mocap replay early enough to show the full wind-up, swing,
    and follow-through. The ball is launched from the pitch config so it
    arrives at the right time.
    """
    pitch = config["pitch"]
    start_frame = config["replay_start_frame"]

    # Start the replay much earlier to show the full swing
    # The swing is around frames 275-300, so start at ~250 to show wind-up
    # The second swing is around frames 840-865, start at ~820
    swing_start = start_frame
    view_start = max(0, swing_start - 40)  # show 40 frames before the pitch starts

    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=view_start)

    # Create env - launch ball from far away initially, we'll set it manually
    env = CMUBattingEnv(
        render_mode="rgb_array",
        frame_skip=5,
        max_steps=max_steps + 100,
        pitch_x=-200,  # far away, we'll manually set the ball position later
    )
    obs, _ = env.reset()

    # Initialise humanoid from mocap
    pos, quat = ctrl.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    action = ctrl.get_action(obs)
    env.data.qpos[7:63] = action
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    ctrl.reset()

    # Compute when to launch the ball
    # The ball was launched at start_frame and arrives some frames later
    ball_start = np.array(pitch["start_pos"])
    ball_vel = np.array(pitch["velocity"])
    ball_launch_step = start_frame - view_start  # step at which to launch the ball

    frames = []
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    ball_launched = False

    for step in range(max_steps):
        if ctrl.done:
            break

        action = ctrl.get_action(obs)
        rp, rq = ctrl.get_root_state()
        env.data.qpos[0:3] = rp
        env.data.qpos[3:7] = rq
        env.data.qvel[0:6] = 0.0

        # Launch the ball at the right time
        if step == ball_launch_step and not ball_launched:
            env.data.qpos[64:67] = ball_start
            env.data.qpos[67:71] = [1, 0, 0, 0]
            env.data.qvel[63:66] = ball_vel
            env.data.qvel[66:69] = [0, 0, 0]
            ball_launched = True

        obs, _, terminated, truncated, info = env.step(action)

        # Render frame
        if cam_id >= 0:
            env._renderer.update_scene(env.data, camera=cam_id)
        else:
            env._renderer.update_scene(env.data)
        frame = env._renderer.render()
        if frame is not None:
            frames.append(frame.copy())

        # After contact, keep rendering a few more frames for the follow-through
        if terminated or truncated:
            # Render 30 more frames to show ball flying away
            for extra in range(30):
                if ctrl.done:
                    break
                action = ctrl.get_action(obs)
                rp, rq = ctrl.get_root_state()
                env.data.qpos[0:3] = rp
                env.data.qpos[3:7] = rq
                env.data.qvel[0:6] = 0.0
                obs, _, _, _, _ = env.step(action)
                if cam_id >= 0:
                    env._renderer.update_scene(env.data, camera=cam_id)
                else:
                    env._renderer.update_scene(env.data)
                frame = env._renderer.render()
                if frame is not None:
                    frames.append(frame.copy())
            break

    env.close()

    # Save GIF
    if frames:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Keep all frames for smooth animation, but cap at 200
        step_size = max(1, len(frames) // 200)
        sub_frames = frames[::step_size]
        imageio.mimwrite(str(output_path), sub_frames, duration=80, loop=0)
        print(f"  Saved {output_path} ({len(sub_frames)} frames from {len(frames)} total)")

    return frames


def render_swing_path_image(output_path):
    """Create a still image showing the humanoid at peak swing with the strike zone visible."""
    replay = CMUMocapReplay(AMC_PATH, control_dt=CONTROL_DT)

    env = CMUBattingEnv(render_mode="rgb_array", pitch_x=-200)
    env.reset()

    # Find peak frame
    all_tip = []
    for f in range(replay.n_frames):
        qpos = replay.qpos_trajectory[:, f].copy()
        env.data.qpos[:63] = qpos
        env.data.qpos[63] = 0.0
        mujoco.mj_forward(env.model, env.data)
        all_tip.append(env.data.site("bat_tip").xpos.copy())
    all_tip = np.array(all_tip)
    tip_vels = np.diff(all_tip, axis=0) / CONTROL_DT
    tip_speeds = np.linalg.norm(tip_vels, axis=1)
    peak_frame = int(np.argmax(tip_speeds))

    # Set humanoid at peak frame
    qpos = replay.qpos_trajectory[:, peak_frame].copy()
    env.data.qpos[:63] = qpos
    env.data.qpos[63] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Render from batter_view camera (uses the env's renderer at 640x480)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "batter_view")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

    env._renderer.update_scene(env.data, camera=cam_id)
    frame = env._renderer.render()
    env.close()

    if frame is not None:
        imageio.imwrite(str(output_path), frame)
        print(f"  Saved swing path image: {output_path}")


def main():
    print("=" * 70)
    print("Swing and Zone Visualization")
    print("=" * 70)

    config = load_pitch_config()
    start_frame = config["replay_start_frame"]
    print(f"\nLoaded config:")
    print(f"  Start frame: {start_frame}")
    print(f"  Ball start: {config['pitch']['start_pos']}")
    print(f"  Ball velocity: {config['pitch']['velocity']}")
    print(f"  Contact: {config.get('contact_achieved', 'unknown')}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Render from multiple camera angles ──
    cameras = [
        ("side", "cmu_swing_side.gif"),
        ("track", "cmu_swing_track.gif"),
        ("batter_view", "cmu_swing_batter_view.gif"),
        ("pitcher_view", "cmu_swing_pitcher_view.gif"),
    ]

    for cam_name, filename in cameras:
        print(f"\n  Rendering from camera: {cam_name}...")
        out_path = results_dir / filename
        render_full_swing(config, cam_name, out_path, max_steps=120)

    # ── Render swing path static image ──
    print("\n  Rendering swing path image...")
    render_swing_path_image(results_dir / "swing_path.png")

    # ── Open the best GIF ──
    best_gif = results_dir / "cmu_swing_batter_view.gif"
    print(f"\n  Opening: {best_gif}")

    print("\nDone!")
    return str(best_gif)


if __name__ == "__main__":
    gif_path = main()
    import subprocess
    subprocess.Popen(["start", "", str(gif_path)], shell=True)
