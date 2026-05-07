"""Tier 2B: PPO on the same 3-d residual action space as Tier 2A.

The env is a one-step bandit (SwingResidualBanditEnv), so PPO is just
learning a policy mu(s)=action for a constant s. This is the cleanest
direct comparison to CMA-ES: same action space, same evaluator, same
objective. PPO's value: it can use gradient info via the policy gradient,
where CMA-ES uses population statistics.

Total budget: 50,000 steps (each step = one swing, ~30 ms eval ->
~25 min wall clock on the local machine). Logs reward + best residuals
to results/tier2b_ppo/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.swing_residual_env import SwingResidualBanditEnv

OUT_DIR = PROJECT_ROOT / "results" / "tier2b_ppo"


class BestTracker(BaseCallback):
    """Logs the best residuals + carry seen across training."""
    def __init__(self):
        super().__init__()
        self.best_carry = -np.inf
        self.best_residuals = None
        self.best_step = -1
        self.history = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "carry_ft" not in info:
                continue
            carry = info["carry_ft"]
            self.history.append({
                "step": int(self.num_timesteps),
                "carry_ft": float(info["carry_ft"]),
                "exit_mph": float(info["exit_mph"]),
                "contact_rate": float(info["contact_rate"]),
                "swing_timing_s": float(info["swing_timing_s"]),
                "hip_fire_rad": float(info["hip_fire_rad"]),
                "uppercut_rad": float(info["uppercut_rad"]),
            })
            if carry > self.best_carry:
                self.best_carry = float(carry)
                self.best_residuals = {
                    "swing_timing_s": float(info["swing_timing_s"]),
                    "hip_fire_rad":   float(info["hip_fire_rad"]),
                    "uppercut_rad":   float(info["uppercut_rad"]),
                }
                self.best_step = int(self.num_timesteps)
        return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def make_env():
        return SwingResidualBanditEnv(pitch_jitter=False, seeds_per_eval=1)

    vec_env = DummyVecEnv([make_env for _ in range(4)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
    )

    tracker = BestTracker()
    print("=" * 70)
    print("Tier 2B FULL: PPO on 3-d residual bandit")
    print("=" * 70)

    total_steps = 50_000
    log_every = 1024
    so_far = 0
    while so_far < total_steps:
        chunk = min(log_every, total_steps - so_far)
        model.learn(total_timesteps=chunk, callback=tracker, reset_num_timesteps=False)
        so_far += chunk
        print(
            f"step {so_far:6d} | best_carry {tracker.best_carry:6.1f} ft | "
            f"best_step {tracker.best_step:6d} | "
            f"residuals {tracker.best_residuals}"
        )

    print()
    print("=" * 70)
    print("PPO done")
    print("=" * 70)
    print(f"  best_carry:       {tracker.best_carry:.2f} ft")
    print(f"  best_residuals:   {tracker.best_residuals}")

    (OUT_DIR / "best.json").write_text(json.dumps({
        "best_carry": tracker.best_carry,
        "best_residuals": tracker.best_residuals,
        "best_step": tracker.best_step,
        "total_steps": total_steps,
    }, indent=2))
    (OUT_DIR / "history.json").write_text(json.dumps(tracker.history, indent=2))
    model.save(str(OUT_DIR / "ppo_model"))
    print(f"  saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
