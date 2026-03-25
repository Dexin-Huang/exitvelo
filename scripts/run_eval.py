"""Evaluate random and tracking baselines on the batting environment.

Loads the frozen pitch config from configs/pitch_v1.json, runs 50 episodes
for each policy, prints a comparison table, saves results, and renders
one video per policy.
"""

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from src.env.batting_env import BattingEnv
from src.controllers.random_policy import RandomPolicy
from src.controllers.tracking_controller import TrackingController
from src.eval.evaluate import evaluate_policy
from src.eval.video import render_episode

RESULTS_DIR = _PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = _PROJECT / "configs" / "pitch_v1.json"
TRAJECTORY_PATH = _PROJECT / "data" / "processed" / "swing_124_07.npz"


def _load_pitch_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _make_env(cfg, render_mode=None):
    """Create a BattingEnv using the frozen pitch config."""
    pitch = cfg["pitch"]
    return BattingEnv(
        render_mode=render_mode,
        frame_skip=5,
        max_steps=500,
        pitch_x=pitch["start_pos"][0],
        pitch_y=pitch["start_pos"][1],
        pitch_height=pitch["start_pos"][2],
        pitch_velocity=pitch["velocity"],
    )


def main():
    cfg = _load_pitch_config()
    replay_start = cfg.get("replay", {}).get("replay_start_mocap", 0)

    print("=" * 70)
    print("Batting Evaluation — Random vs Tracking")
    print("=" * 70)
    print(f"Pitch config: {CONFIG_PATH}")
    print(f"  Ball start:    {cfg['pitch']['start_pos']}")
    print(f"  Ball velocity: {cfg['pitch']['velocity']}")
    print(f"  Replay start:  mocap frame {replay_start}")
    print()

    # ── Random Policy ──
    print("Evaluating RandomPolicy (50 episodes) ...")
    env_rand = _make_env(cfg)
    random_policy = RandomPolicy(env_rand.action_space)
    random_results = evaluate_policy(env_rand, random_policy, n_episodes=50)
    env_rand.close()
    print(f"  Contact rate: {random_results['contact_rate']:.1%}")
    print(f"  Mean ball speed: {random_results['mean_ball_speed']:.2f} m/s")

    # ── Tracking Controller ──
    print("\nEvaluating TrackingController (50 episodes) ...")
    env_track = _make_env(cfg)
    tracking_policy = TrackingController(
        str(TRAJECTORY_PATH),
        control_fps=100.0,
        start_frame=replay_start,
        clamp_root=True,
    )
    tracking_results = evaluate_policy(
        env_track, tracking_policy, n_episodes=50, clamp_root=True
    )
    env_track.close()
    print(f"  Contact rate: {tracking_results['contact_rate']:.1%}")
    print(f"  Mean ball speed: {tracking_results['mean_ball_speed']:.2f} m/s")

    # ── Comparison table ──
    print("\n" + "=" * 70)
    print(f"| {'Method':<14} | {'Contact Rate':>12} | {'Mean Ball Speed (m/s)':>21} | {'Notes':<20} |")
    print(f"|{'-'*16}|{'-'*14}|{'-'*23}|{'-'*22}|")
    print(f"| {'Random':<14} | {random_results['contact_rate']:>11.1%} | "
          f"{random_results['mean_ball_speed']:>21.2f} | {'Lower bound':<20} |")
    print(f"| {'Tracking':<14} | {tracking_results['contact_rate']:>11.1%} | "
          f"{tracking_results['mean_ball_speed']:>21.2f} | {'Reference-following':<20} |")
    print("=" * 70)

    # ── Save results ──
    results_path = RESULTS_DIR / "milestone_results.json"
    results_data = {
        "random": {
            "contact_rate": random_results["contact_rate"],
            "mean_ball_speed": random_results["mean_ball_speed"],
            "n_episodes": random_results["n_episodes"],
        },
        "tracking": {
            "contact_rate": tracking_results["contact_rate"],
            "mean_ball_speed": tracking_results["mean_ball_speed"],
            "n_episodes": tracking_results["n_episodes"],
        },
        "pitch_config": cfg["pitch"],
        "replay_start_mocap": replay_start,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Render videos ──
    print("\nRendering videos ...")
    try:
        # Random video
        env_vid = _make_env(cfg, render_mode="rgb_array")
        rp = RandomPolicy(env_vid.action_space)
        render_episode(env_vid, rp, str(RESULTS_DIR / "random_episode.mp4"),
                       max_steps=200)
        env_vid.close()

        # Tracking video
        env_vid2 = _make_env(cfg, render_mode="rgb_array")
        tp = TrackingController(
            str(TRAJECTORY_PATH), control_fps=100.0,
            start_frame=replay_start, clamp_root=True,
        )
        render_episode(env_vid2, tp, str(RESULTS_DIR / "tracking_episode.mp4"),
                       max_steps=200, clamp_root=True)
        env_vid2.close()
    except Exception as e:
        print(f"  Video rendering failed: {e}")
        print("  (This is expected if running without a display.)")

    print("\nDone!")


if __name__ == "__main__":
    main()
