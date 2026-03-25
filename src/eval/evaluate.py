"""Evaluation harness for batting policies."""

import mujoco
import numpy as np

try:
    from src.env.batting_env import _QPOS_JOINT_START, _QPOS_JOINT_END
except ImportError:
    pass

# Also support CMU env constants (same values)
_CMU_QPOS_JOINT_START = 7
_CMU_QPOS_JOINT_END = 63


def evaluate_policy(env, policy, n_episodes=50, seed=42, clamp_root=False):
    """Run *n_episodes* rollouts and collect contact / speed statistics.

    Parameters
    ----------
    env : BattingEnv instance
    policy : object with get_action(obs) and reset()
    n_episodes : number of episodes
    seed : base random seed
    clamp_root : if True and the policy has get_root_state(), set the
        humanoid root position/orientation from the policy each step.
        This prevents the humanoid from falling and is necessary for
        the TrackingController to produce realistic contact.

    Returns a dict with aggregate metrics and per-episode data.
    """
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset()

        # If clamping root, initialise the humanoid pose from the policy
        if clamp_root and hasattr(policy, "get_root_state"):
            pos, quat = policy.get_root_state()
            env.data.qpos[0:3] = pos
            env.data.qpos[3:7] = quat
            env.data.qvel[0:3] = 0.0
            env.data.qvel[3:6] = 0.0
            # Also set the joint targets to the first frame
            action_init = policy.get_action(obs)
            env.data.qpos[_CMU_QPOS_JOINT_START:_CMU_QPOS_JOINT_END] = action_init
            mujoco.mj_forward(env.model, env.data)
            # Update bat position between hands
            if hasattr(policy, "update_bat"):
                policy.update_bat()
                mujoco.mj_forward(env.model, env.data)
            obs = env._get_obs()
            # Reset the policy step counter (get_action incremented it)
            policy.reset()

        done = False
        ep_data = {
            "contact": False,
            "ball_speed_post": 0.0,
            "termination": "unknown",
            "steps": 0,
        }

        while not done:
            action = policy.get_action(obs)

            # Optionally clamp the humanoid root to the mocap trajectory
            if clamp_root and hasattr(policy, "get_root_state"):
                pos, quat = policy.get_root_state()
                env.data.qpos[0:3] = pos
                env.data.qpos[3:7] = quat
                env.data.qvel[0:3] = 0.0
                env.data.qvel[3:6] = 0.0

            obs, reward, terminated, truncated, info = env.step(action)

            # After physics step, update bat to follow hands
            if hasattr(policy, "update_bat"):
                mujoco.mj_forward(env.model, env.data)
                policy.update_bat()
                mujoco.mj_forward(env.model, env.data)

            done = terminated or truncated
            ep_data["steps"] += 1

            if info.get("contact", False):
                ep_data["contact"] = True
                ep_data["ball_speed_post"] = info.get("ball_speed_post", 0.0)

            ep_data["termination"] = info.get("termination_reason", "unknown")

        results.append(ep_data)

    contact_rate = float(np.mean([r["contact"] for r in results]))
    speeds = [r["ball_speed_post"] for r in results if r["contact"]]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0

    return {
        "contact_rate": contact_rate,
        "mean_ball_speed": mean_speed,
        "n_episodes": n_episodes,
        "episodes": results,
    }
