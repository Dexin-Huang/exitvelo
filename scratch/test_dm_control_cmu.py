"""Test CMU mocap replay using dm_control's CMU humanoid.

Loads the dm_control CMU humanoid, converts our AMC file to qpos via
parse_amc.convert(), replays the motion by setting qpos each frame,
and saves a GIF to results/cmu_humanoid_swing.gif.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dm_control imports
from dm_control import mujoco as dm_mujoco
from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

# For rendering
import imageio


def main():
    print("=" * 60)
    print("Testing CMU Mocap Replay with dm_control CMU Humanoid")
    print("=" * 60)

    # --- 1. Load the CMU humanoid environment ---
    print("\n[1] Loading CMU humanoid model...")
    xml_string, assets = humanoid_CMU.get_model_and_assets()
    physics = dm_mujoco.Physics.from_xml_string(xml_string, assets)

    print(f"  Model loaded successfully")
    print(f"  nq (qpos dim): {physics.model.nq}")
    print(f"  nv (qvel dim): {physics.model.nv}")
    print(f"  nu (actuators): {physics.model.nu}")

    # Print joint names
    joint_names = [physics.model.joint(i).name for i in range(physics.model.njnt)]
    print(f"  Number of joints: {len(joint_names)}")
    print(f"  Joint names: {joint_names[:10]}... (showing first 10)")

    # Print all joint names for reference
    print("\n  Full joint list:")
    for i, name in enumerate(joint_names):
        print(f"    [{i:2d}] {name}")

    # --- 2. Load and convert AMC file ---
    amc_path = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"
    print(f"\n[2] Loading AMC file: {amc_path}")

    if not amc_path.exists():
        print(f"  ERROR: AMC file not found at {amc_path}")
        return

    # parse_amc.convert() needs: file_name, physics, timestep
    # The CMU humanoid model timestep
    control_timestep = 0.02  # 50 Hz control
    converted = parse_amc.convert(str(amc_path), physics, control_timestep)

    print(f"  Conversion successful!")
    print(f"  qpos shape: {converted.qpos.shape}")
    print(f"  qvel shape: {converted.qvel.shape}")
    print(f"  time shape: {converted.time.shape}")
    print(f"  Duration: {converted.time[-1]:.2f} seconds")
    print(f"  Number of frames: {converted.qpos.shape[1]}")

    # --- 3. Replay motion ---
    print(f"\n[3] Replaying motion and rendering frames...")

    # Determine frame range for the swing
    # The full motion is quite long; let's find the swing portion
    # Subject 124, trial 07 is a baseball swing
    n_frames = converted.qpos.shape[1]
    print(f"  Total frames available: {n_frames}")

    # Render every Nth frame to keep GIF manageable
    # At 50 Hz, rendering every 2nd frame gives 25 fps
    render_every = 2
    frames = []

    # Camera settings
    camera_id = 0  # 'back' camera

    for t in range(n_frames):
        # Set qpos from converted trajectory
        physics.data.qpos[:] = converted.qpos[:, t]

        # Also set qvel if available
        if t < converted.qvel.shape[1]:
            physics.data.qvel[:] = converted.qvel[:, t]

        # Forward kinematics
        dm_mujoco.Physics.forward(physics)

        # Render
        if t % render_every == 0:
            img = physics.render(height=480, width=640, camera_id=camera_id)
            frames.append(img)

            if t % 50 == 0:
                root_pos = converted.qpos[:3, t]
                print(f"  Frame {t}/{n_frames}: root_pos = [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")

    print(f"  Rendered {len(frames)} frames")

    # --- 4. Save GIF ---
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    gif_path = results_dir / "cmu_humanoid_swing.gif"

    print(f"\n[4] Saving GIF to {gif_path}...")
    # 25 fps (since we skip every other frame from 50 Hz)
    imageio.mimsave(str(gif_path), frames, fps=25, loop=0)
    print(f"  GIF saved! ({len(frames)} frames, {os.path.getsize(gif_path) / 1024:.0f} KB)")

    # --- 5. Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CMU humanoid: {physics.model.nq} qpos dims, {physics.model.nv} qvel dims, {physics.model.nu} actuators")
    print(f"  AMC trajectory: {n_frames} frames at {1/control_timestep:.0f} Hz = {converted.time[-1]:.2f}s")
    print(f"  GIF saved to: {gif_path}")
    print(f"  Open the GIF to verify the swing looks correct!")


if __name__ == "__main__":
    main()
