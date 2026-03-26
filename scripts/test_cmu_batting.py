"""Test the full CMU humanoid batting pipeline.

Creates the CMU batting environment, replays the mocap swing with a
ball pitched at the batter, renders and saves a GIF.
"""

import os
import sys
from pathlib import Path

import numpy as np
import imageio

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.cmu_batting_env import CMUBattingEnv
from src.motion.cmu_replay import CMUMocapReplay


def main():
    print("=" * 60)
    print("Testing Full CMU Humanoid Batting Pipeline")
    print("=" * 60)

    # --- 1. Load mocap replay ---
    amc_path = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"
    print(f"\n[1] Loading mocap replay from {amc_path.name}...")

    # The env uses frame_skip=5 with timestep=0.002, so control_dt = 0.01
    replay = CMUMocapReplay(str(amc_path), control_dt=0.01)
    print(f"  {replay}")
    print(f"  Total frames: {replay.num_frames}")

    # --- 2. Find the swing window ---
    # Subject 124 trial 07 has multiple swings. We need to find the
    # right portion. Let's analyze the motion to find the swing.
    print(f"\n[2] Analyzing motion to find swing window...")

    # Look at the arm joint velocities to find the swing
    # The right arm joints (rhumerus, rradius) should move fast during swing
    max_arm_vel = []
    for t in range(min(replay.num_frames - 1, replay.qvel_trajectory.shape[1])):
        qvel = replay.qvel_trajectory[:, t]
        # Joints 7: (root free dofs) then hinge joints in body-tree order
        # rhumerus joints are around qvel indices for the right arm
        # Just use overall joint velocity magnitude as proxy
        arm_vel = np.abs(qvel[6:]).max()  # skip root
        max_arm_vel.append(arm_vel)

    max_arm_vel = np.array(max_arm_vel)

    # Find peaks - swing should have high velocity
    threshold = np.percentile(max_arm_vel, 90)
    swing_frames = np.where(max_arm_vel > threshold)[0]

    if len(swing_frames) > 0:
        # Find the first swing
        swing_start = max(0, swing_frames[0] - 50)  # 50 frames before
        swing_end = min(replay.num_frames - 1, swing_frames[0] + 100)  # 100 after
        print(f"  Found swing activity at frame ~{swing_frames[0]}")
        print(f"  Will replay frames {swing_start} to {swing_end}")
    else:
        # Default: use middle portion
        swing_start = replay.num_frames // 4
        swing_end = min(swing_start + 200, replay.num_frames - 1)
        print(f"  No clear swing peak found, using frames {swing_start}-{swing_end}")

    # --- 3. Create environment and replay ---
    print(f"\n[3] Creating CMU batting environment...")

    # Calculate pitch timing: ball should arrive during the swing
    # The swing happens over ~0.3-0.5 seconds
    swing_duration = (swing_end - swing_start) * 0.01  # seconds
    swing_midpoint = swing_start + (swing_end - swing_start) // 2

    # We want the ball to arrive roughly when the bat is in the hitting zone
    # Ball travels from pitch_x to batter (x=0)
    pitch_x = -5.0  # closer for testing visibility
    pitch_speed = 20.0  # slower for testing
    flight_time = abs(pitch_x) / pitch_speed

    # Launch the ball so it arrives at the swing midpoint
    ball_launch_frame = swing_midpoint - int(flight_time / 0.01)
    ball_launch_frame = max(swing_start, ball_launch_frame)

    print(f"  Pitch: x={pitch_x}, speed={pitch_speed} m/s")
    print(f"  Flight time: {flight_time:.3f}s")
    print(f"  Ball launch at frame {ball_launch_frame} (relative to swing_start)")

    env = CMUBattingEnv(
        render_mode="rgb_array",
        frame_skip=5,
        max_steps=500,
        pitch_speed=pitch_speed,
        pitch_x=pitch_x,
        pitch_height=1.0,
    )

    print(f"  Env created: obs_space={env.observation_space.shape}, "
          f"act_space={env.action_space.shape}")

    # --- 4. Run replay ---
    print(f"\n[4] Running replay with rendering...")

    obs, info = env.reset()

    # First set the humanoid to the swing start pose
    qpos_start = replay.get_qpos(swing_start)
    env.set_humanoid_qpos(qpos_start)
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    frames = []
    render_every = 2
    ball_launched = False

    for step in range(swing_end - swing_start):
        current_frame = swing_start + step

        # Get target joint angles from mocap
        replay.step_idx = current_frame
        action = replay.get_action()

        # Set full humanoid qpos directly for kinematic replay
        qpos = replay.get_qpos(current_frame)
        env.set_humanoid_qpos(qpos)

        # Launch ball at the right time
        if step >= (ball_launch_frame - swing_start) and not ball_launched:
            env._launch_ball()
            ball_launched = True
            print(f"  Ball launched at step {step}")

        mujoco.mj_forward(env.model, env.data)

        # Also step physics for ball movement
        if ball_launched:
            mujoco.mj_step(env.model, env.data)
            # Re-set humanoid pose after physics step (kinematic replay)
            env.set_humanoid_qpos(qpos)
            mujoco.mj_forward(env.model, env.data)

        # Render
        if step % render_every == 0:
            img = env.render()
            if img is not None:
                frames.append(img)

        if step % 20 == 0:
            root_pos = qpos[:3]
            print(f"  Step {step}/{swing_end-swing_start}: "
                  f"root=[{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}]")

    print(f"  Rendered {len(frames)} frames")

    # --- 5. Save GIF ---
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    gif_path = results_dir / "cmu_batting_swing.gif"

    print(f"\n[5] Saving GIF to {gif_path}...")
    if len(frames) > 0:
        imageio.mimsave(str(gif_path), frames, fps=25, loop=0)
        print(f"  GIF saved! ({len(frames)} frames, {os.path.getsize(gif_path) / 1024:.0f} KB)")
    else:
        print("  ERROR: No frames rendered!")
        return

    # --- 6. Also save a swing-only GIF (no ball, better view) ---
    print(f"\n[6] Creating swing-only GIF (full motion, no ball)...")
    frames_full = []
    env2 = CMUBattingEnv(render_mode="rgb_array", frame_skip=5)
    env2.reset()

    # Remove ball from view by moving it far away
    env2.data.qpos[64:67] = [100, 100, 100]

    # Replay the full swing portion
    for t in range(swing_start, swing_end):
        qpos = replay.get_qpos(t)
        env2.set_humanoid_qpos(qpos)
        mujoco.mj_forward(env2.model, env2.data)

        if (t - swing_start) % 2 == 0:
            img = env2.render()
            if img is not None:
                frames_full.append(img)

    gif_path2 = results_dir / "cmu_swing_only.gif"
    if len(frames_full) > 0:
        imageio.mimsave(str(gif_path2), frames_full, fps=25, loop=0)
        print(f"  Saved {gif_path2.name} ({len(frames_full)} frames, "
              f"{os.path.getsize(gif_path2) / 1024:.0f} KB)")

    # --- 7. Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CMU humanoid: 56 joints, 63 qpos, 62 qvel")
    print(f"  Mocap: {replay.num_frames} frames, {replay.time[-1]:.2f}s")
    print(f"  Swing window: frames {swing_start}-{swing_end}")
    print(f"  GIFs saved:")
    print(f"    - {gif_path}")
    print(f"    - {gif_path2}")
    print(f"  Open the GIFs to verify the swing looks correct!")

    env.close()
    env2.close()


if __name__ == "__main__":
    main()
