"""Video rendering utility for batting episodes."""

import mujoco
import numpy as np
import imageio
from pathlib import Path

try:
    from src.env.batting_env import _QPOS_JOINT_START, _QPOS_JOINT_END
except ImportError:
    pass

# Also support CMU env constants (same values)
_CMU_QPOS_JOINT_START = 7
_CMU_QPOS_JOINT_END = 63


def render_episode(env, policy, output_path, max_steps=500, clamp_root=False):
    """Roll out one episode and save a video (gif or mp4).

    Parameters
    ----------
    env : BattingEnv with render_mode='rgb_array'
    policy : policy with get_action(obs) and reset()
    output_path : destination file path (e.g. 'results/video.gif')
    max_steps : hard cap on number of env steps
    clamp_root : if True and policy has get_root_state(), clamp the
        humanoid root to the mocap trajectory each step.

    Returns
    -------
    frames : list of rgb arrays captured during the episode
    """
    frames = []
    obs, _ = env.reset()
    policy.reset()

    # Initialise humanoid pose from policy if clamping root
    if clamp_root and hasattr(policy, "get_root_state"):
        pos, quat = policy.get_root_state()
        env.data.qpos[0:3] = pos
        env.data.qpos[3:7] = quat
        env.data.qvel[0:3] = 0.0
        env.data.qvel[3:6] = 0.0
        action_init = policy.get_action(obs)
        env.data.qpos[_CMU_QPOS_JOINT_START:_CMU_QPOS_JOINT_END] = action_init
        mujoco.mj_forward(env.model, env.data)
        # Update bat position between hands
        if hasattr(policy, "update_bat"):
            policy.update_bat()
            mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        policy.reset()

    done = False
    step = 0

    while not done and step < max_steps:
        action = policy.get_action(obs)

        if clamp_root and hasattr(policy, "get_root_state"):
            pos, quat = policy.get_root_state()
            env.data.qpos[0:3] = pos
            env.data.qpos[3:7] = quat
            env.data.qvel[0:3] = 0.0
            env.data.qvel[3:6] = 0.0

        obs, _, terminated, truncated, info = env.step(action)

        # After physics step, update bat to follow hands
        if hasattr(policy, "update_bat"):
            mujoco.mj_forward(env.model, env.data)
            policy.update_bat()
            mujoco.mj_forward(env.model, env.data)

        done = terminated or truncated
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        step += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    # Try mp4 first (requires ffmpeg backend), fall back to gif
    try:
        imageio.mimwrite(output_path, frames, fps=30)
    except Exception:
        # Fall back to gif
        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        # Subsample for smaller gifs
        step_size = max(1, len(frames) // 100)
        sub_frames = frames[::step_size]
        imageio.mimwrite(gif_path, sub_frames, duration=100, loop=0)
        output_path = gif_path

    print(f"Saved video: {output_path} ({len(frames)} frames)")
    return frames
