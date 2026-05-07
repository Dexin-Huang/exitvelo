"""Generate a 4-panel milestone figure for the CMU humanoid batting project.

Panels: stance, mid-swing, contact, follow-through.
Saves to results/cmu_milestone_figure.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv

AMC_PATH = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"


def load_pitch_config():
    config_path = PROJECT_ROOT / "configs" / "pitch_v1.json"
    with open(config_path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("CMU Milestone Figure Generation")
    print("=" * 70)

    config = load_pitch_config()
    start_frame = config["replay_start_frame"]
    pitch = config["pitch"]

    env = CMUBattingEnv(
        render_mode="rgb_array",
        frame_skip=5,
        max_steps=500,
        pitch_x=pitch["start_pos"][0],
        pitch_y=pitch["start_pos"][1],
        pitch_height=pitch["start_pos"][2],
        pitch_velocity=pitch["velocity"],
    )
    ctrl = CMUTrackingController(amc_path=str(AMC_PATH), start_frame=start_frame)

    obs, _ = env.reset()

    # Initialise
    pos, quat = ctrl.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    action_init = ctrl.get_action(obs)
    env.data.qpos[7:63] = action_init
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    ctrl.reset()

    # Run episode and collect all frames + bat tip speeds.
    # We continue past contact/termination for follow-through frames
    # by keeping the humanoid in kinematic replay (no env.step after contact).
    all_frames = []
    all_tip_speeds = []
    contact_step = None
    post_contact_steps = 30  # extra steps after contact for follow-through

    total_steps = 60  # enough for stance -> follow-through
    for step in range(total_steps):
        if ctrl.done:
            break

        action = ctrl.get_action(obs)
        rp, rq = ctrl.get_root_state()
        env.data.qpos[0:3] = rp
        env.data.qpos[3:7] = rq
        env.data.qvel[0:6] = 0.0

        if contact_step is not None and step > contact_step:
            # After contact, do kinematic replay only (no physics step)
            # to capture follow-through without the ball interfering
            env.data.qpos[7:63] = action
            mujoco.mj_forward(env.model, env.data)
            frame = env.render()
            if frame is not None:
                all_frames.append(frame)
            tip_vel = obs[128:131]
            tip_speed = float(np.linalg.norm(tip_vel))
            all_tip_speeds.append(tip_speed)
            continue

        obs, _, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            all_frames.append(frame)
        tip_vel = obs[128:131]
        tip_speed = float(np.linalg.norm(tip_vel))
        all_tip_speeds.append(tip_speed)

        if info.get("contact", False) and contact_step is None:
            contact_step = step
            print(f"  Contact at step {step}")

    n_frames = len(all_frames)
    n_steps = len(all_tip_speeds)

    print(f"  Collected {n_frames} frames, {n_steps} steps")

    # Pick 4 key frames
    # Stance: early in the swing (step 0 or ~5% of way through)
    # Mid-swing: just before peak bat speed
    # Contact: at contact step or peak bat speed
    # Follow-through: after contact

    tip_speeds = np.array(all_tip_speeds)
    peak_step = int(np.argmax(tip_speeds))

    if contact_step is not None:
        contact_frame_idx = contact_step
    else:
        contact_frame_idx = peak_step

    stance_idx = max(0, min(5, n_frames - 1))
    midswing_idx = max(0, min(contact_frame_idx - max(10, contact_frame_idx // 3), n_frames - 1))
    contact_idx = min(contact_frame_idx, n_frames - 1)
    followthrough_idx = min(contact_frame_idx + max(15, (n_steps - contact_frame_idx) // 2), n_frames - 1)

    # Ensure they're all different and in order
    indices = sorted(set([stance_idx, midswing_idx, contact_idx, followthrough_idx]))
    while len(indices) < 4:
        # Add intermediate frames if needed
        gap = (indices[-1] - indices[0]) // (4 - len(indices) + 1)
        new_idx = indices[0] + gap
        if new_idx not in indices and 0 <= new_idx < n_frames:
            indices.append(new_idx)
            indices = sorted(indices)
        else:
            indices.append(min(indices[-1] + 5, n_frames - 1))
            indices = sorted(set(indices))

    indices = indices[:4]

    labels = ["Stance", "Mid-Swing", "Contact", "Follow-Through"]
    print(f"  Frame indices: {indices}")

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    fig.suptitle("CMU Humanoid Batting Swing (Mocap Tracking + Root Clamping)",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, (idx, label) in enumerate(zip(indices, labels)):
        ax = axes[i]
        if idx < len(all_frames):
            ax.imshow(all_frames[idx])
        ax.set_title(f"{label}\n(step {idx})", fontsize=11)
        ax.axis("off")

    plt.tight_layout()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    fig_path = results_dir / "cmu_milestone_figure.png"
    fig.savefig(str(fig_path), bbox_inches="tight", dpi=150, pad_inches=0.1)
    plt.close(fig)

    print(f"\n  Saved figure: {fig_path}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
