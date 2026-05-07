"""Tier 2B: one-step Gymnasium bandit env over the 3 active residuals.

Each step is one full swing through the kinematic evaluator. Action is
3-d normalized [-1, +1] (swing_timing, hip_fire, uppercut), reward is
-cma_objective_kin (so PPO maximizes), episode terminates after one
step. Same action space as Tier 2A, different optimizer.

Entry point:
  scripts/run/run_tier2b_ppo.py  -- PPO training loop
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.controllers.swing_residuals import SwingResiduals
from src.optim.kinematic_evaluator import (
    cma_objective_kin,
    evaluate_residuals_kinematic,
    kin_aggregate,
)


class SwingResidualBanditEnv(gym.Env):
    """One-step bandit env over the 3-d normalized residual space.

    Used by PPO to compare against CMA-ES. Action -> SwingResiduals ->
    kinematic rollout -> reward = -cma_objective_kin (so larger is better
    for PPO).
    """

    metadata = {"render_modes": []}

    def __init__(self, *, pitch_jitter: bool = False, seeds_per_eval: int = 1):
        super().__init__()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )
        # PPO wants a non-empty obs; we use a 1-d constant.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        self.pitch_jitter = pitch_jitter
        self.seeds_per_eval = seeds_per_eval
        self._step_seed = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_seed = int(self.np_random.integers(0, 2**31 - 1))
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        a3 = np.asarray(action, dtype=np.float64).flatten()[:3]
        a5 = np.zeros(5, dtype=np.float64)
        a5[:3] = a3
        residuals = SwingResiduals.from_normalized(a5)
        seeds = list(range(self._step_seed, self._step_seed + self.seeds_per_eval))
        results = evaluate_residuals_kinematic(
            residuals, seeds=seeds, pitch_jitter=self.pitch_jitter,
        )
        cost = cma_objective_kin(results)
        reward = -cost  # PPO maximizes; cma_objective is a cost
        agg = kin_aggregate(results)
        info = {
            "carry_ft": agg["mean_carry_ft"],
            "exit_mph": agg["mean_exit_mph"],
            "contact_rate": agg["contact_rate"],
            "swing_timing_s": residuals.swing_timing_s,
            "hip_fire_rad": residuals.hip_fire_rad,
            "uppercut_rad": residuals.uppercut_rad,
        }
        return (
            np.zeros(1, dtype=np.float32),
            float(reward),
            True,   # terminated -- one-step bandit
            False,
            info,
        )
