"""Generate a figure for the project milestone report.

Creates a 1x4 subplot showing key moments of the batting swing:
  1. Stance (initial frame)
  2. Wind-up / mid-swing
  3. Contact / near-contact
  4. Follow-through (post-contact, from a longer non-terminating replay)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from src.env.batting_env import (
    BattingEnv, _QPOS_JOINT_START, _QPOS_JOINT_END
)
from src.controllers.tracking_controller import TrackingController

RESULTS_DIR = _PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = _PROJECT / "configs" / "pitch_v1.json"
TRAJECTORY_PATH = _PROJECT / "data" / "processed" / "swing_124_07.npz"


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    pitch = cfg["pitch"]
    replay_start = cfg.get("replay", {}).get("replay_start_mocap", 0)
    contact_step_expected = cfg.get("results", {}).get("contact_step", 18)

    # Create env WITHOUT ball (to get a longer replay for follow-through)
    env = BattingEnv(
        render_mode="rgb_array",
        frame_skip=5,
        max_steps=500,
        pitch_x=-200,  # ball far away so no contact / termination
        pitch_speed=0,
        pitch_height=0,
    )

    policy = TrackingController(
        str(TRAJECTORY_PATH),
        control_fps=100.0,
        start_frame=replay_start,
        clamp_root=True,
    )

    obs, _ = env.reset()
    policy.reset()

    # Initialise humanoid
    pos, quat = policy.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    action_init = policy.get_action(obs)
    env.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END] = action_init
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    policy.reset()

    # Collect frames for a longer replay (no ball = no early termination)
    all_frames = []
    step = 0
    max_render_steps = max(60, contact_step_expected + 30)
    while step < max_render_steps:
        action = policy.get_action(obs)
        p, q = policy.get_root_state()
        env.data.qpos[0:3] = p
        env.data.qpos[3:7] = q
        env.data.qvel[0:3] = 0.0
        env.data.qvel[3:6] = 0.0
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            all_frames.append(frame)
        step += 1
        if terminated or truncated:
            break
    env.close()

    n = len(all_frames)
    if n == 0:
        print("No frames rendered. Cannot create figure.")
        return

    # Pick key moments based on the known contact step
    cs = min(contact_step_expected, n - 1)
    stance_idx = 0
    mid_idx = max(1, cs // 2)
    contact_idx = cs
    follow_idx = min(cs + 10, n - 1)

    key_frames = [
        (all_frames[stance_idx], "Stance"),
        (all_frames[mid_idx], "Mid-swing"),
        (all_frames[contact_idx], "Contact"),
        (all_frames[follow_idx], "Follow-through"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (frame, title) in zip(axes, key_frames):
        ax.imshow(frame)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Baseball Batting -- Tracking Controller Swing Sequence",
                 fontsize=15, y=1.02)
    plt.tight_layout()

    out_path = RESULTS_DIR / "milestone_figure.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
