"""Evaluate random and tracking policies on the CMU batting environment.

Loads pitch configuration from configs/pitch_v1.json, runs 50 episodes
each for RandomPolicy and CMUTrackingController, prints a comparison
table, saves results to results/cmu_milestone_results.json, and renders
a video for each policy.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.controllers.random_policy import RandomPolicy
from src.env.cmu_batting_env import CMUBattingEnv
from src.eval.evaluate import evaluate_policy
from src.eval.video import render_episode

AMC_PATH = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"


def load_pitch_config():
    config_path = PROJECT_ROOT / "configs" / "pitch_v1.json"
    with open(config_path) as f:
        return json.load(f)


def make_env(config, render_mode=None):
    pitch = config["pitch"]
    return CMUBattingEnv(
        render_mode=render_mode,
        frame_skip=5,
        max_steps=500,
        pitch_x=pitch["start_pos"][0],
        pitch_y=pitch["start_pos"][1],
        pitch_height=pitch["start_pos"][2],
        pitch_velocity=pitch["velocity"],
    )


def main():
    print("=" * 70)
    print("CMU Humanoid Batting Evaluation")
    print("=" * 70)

    config = load_pitch_config()
    start_frame = config["replay_start_frame"]
    print(f"\nLoaded pitch config:")
    print(f"  Start frame: {start_frame}")
    print(f"  Ball start: {config['pitch']['start_pos']}")
    print(f"  Ball velocity: {config['pitch']['velocity']}")
    print(f"  Contact achieved in sync: {config.get('contact_achieved', 'unknown')}")

    # ── Evaluate Random Policy ──
    print("\n[1] Evaluating RandomPolicy (50 episodes)...")
    env_rand = make_env(config)
    random_policy = RandomPolicy(env_rand.action_space)
    rand_results = evaluate_policy(
        env_rand, random_policy, n_episodes=50, seed=42, clamp_root=False
    )
    env_rand.close()
    print(f"  Contact rate: {rand_results['contact_rate']:.1%}")
    print(f"  Mean ball speed: {rand_results['mean_ball_speed']:.2f} m/s")

    # ── Evaluate Tracking Controller ──
    print("\n[2] Evaluating CMUTrackingController (50 episodes, root clamped)...")
    env_track = make_env(config)
    tracking_policy = CMUTrackingController(
        amc_path=str(AMC_PATH), start_frame=start_frame, env=env_track
    )
    track_results = evaluate_policy(
        env_track, tracking_policy, n_episodes=50, seed=42, clamp_root=True
    )
    env_track.close()
    print(f"  Contact rate: {track_results['contact_rate']:.1%}")
    print(f"  Mean ball speed: {track_results['mean_ball_speed']:.2f} m/s")

    # ── Comparison Table ──
    print("\n" + "=" * 70)
    print(f"{'Policy':<25} {'Contact Rate':>15} {'Mean Ball Speed':>18}")
    print("-" * 70)
    print(f"{'RandomPolicy':<25} "
          f"{rand_results['contact_rate']:>14.1%} "
          f"{rand_results['mean_ball_speed']:>15.2f} m/s")
    print(f"{'CMUTracking (clamped)':<25} "
          f"{track_results['contact_rate']:>14.1%} "
          f"{track_results['mean_ball_speed']:>15.2f} m/s")
    print("=" * 70)

    # ── Save Results ──
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results = {
        "pitch_config": config,
        "random_policy": {
            "contact_rate": rand_results["contact_rate"],
            "mean_ball_speed": rand_results["mean_ball_speed"],
            "n_episodes": rand_results["n_episodes"],
        },
        "tracking_policy": {
            "contact_rate": track_results["contact_rate"],
            "mean_ball_speed": track_results["mean_ball_speed"],
            "n_episodes": track_results["n_episodes"],
        },
    }

    results_path = results_dir / "cmu_milestone_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Render Videos ──
    print("\n[3] Rendering videos...")

    # Random policy video
    print("  Rendering random policy episode...")
    env_vid = make_env(config, render_mode="rgb_array")
    random_policy2 = RandomPolicy(env_vid.action_space)
    random_vid_path = results_dir / "cmu_random_episode.gif"
    render_episode(env_vid, random_policy2, str(random_vid_path),
                   max_steps=300, clamp_root=False)
    env_vid.close()

    # Tracking policy video
    print("  Rendering tracking policy episode...")
    env_vid2 = make_env(config, render_mode="rgb_array")
    tracking_policy2 = CMUTrackingController(
        amc_path=str(AMC_PATH), start_frame=start_frame, env=env_vid2
    )
    tracking_vid_path = results_dir / "cmu_tracking_episode.gif"
    render_episode(env_vid2, tracking_policy2, str(tracking_vid_path),
                   max_steps=300, clamp_root=True)
    env_vid2.close()

    print(f"\nVideos saved:")
    print(f"  {random_vid_path}")
    print(f"  {tracking_vid_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
