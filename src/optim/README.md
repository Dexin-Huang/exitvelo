# src/optim — Tier 2 swing-residual optimizers

The Tier 2 stack searches the 3 active dimensions of `SwingResiduals`
(swing_timing_s, hip_fire_rad, uppercut_rad) to maximize hit distance
on the calibrated 93 mph fastball. Two optimizers share one evaluator.

## Modules

### `kinematic_evaluator.py` — the active evaluator
The evaluator used by both Tier 2A and Tier 2B. Drives the humanoid
kinematically (sets qpos directly each control step) and propagates
the ball analytically using `src/env/contacts.py` (Nathan exit
formula + drag flight). Fast (~10 ms/rollout) and reliably produces
contacts. Exports:

- `kinematic_rollout(residuals, ...) -> KinResult` — single swing
- `evaluate_residuals_kinematic(residuals, seeds=..., pitch_jitter=...)` — multi-seed wrapper
- `cma_objective_kin(results)` — minimization objective `J = -mean(carry) + 200*miss_rate + 5*mean(min_dist on misses)`
- `kin_aggregate(results)` — summary stats for logging
- `KinResult` dataclass — per-rollout metrics (contact, exit_mph, launch_deg, carry_ft, total_ft, min_bat_ball_dist, sweet_speed_at_contact)

### `cma_runner.py` — Tier 2A driver
CMA-ES on the 3 active residuals. Pads `SwingResiduals` to 5-d with
zeros for the deferred dims. Logs each generation to `generations.csv`
and dumps the best-rollout JSON to `results/tier2a_cma/<tag>/`.

- Entry point: `scripts/run/run_tier2a_full.py`
- Population: 10, max generations: 40 (defaults)
- First 10 gens use 1 deterministic seed; later gens use 3 jittered seeds

### `swing_residual_env.py` — Tier 2B env
Gymnasium one-step bandit env over the 3 active residuals. Each
`step` runs one full kinematic swing; reward is
`-cma_objective_kin(...)`. Same action space as Tier 2A; a clean
PPO-vs-CMA-ES comparison.

- Entry point: `scripts/run/run_tier2b_ppo.py`
- Action space: `Box(-1, +1, shape=(3,))`
- Observation: 1-d constant placeholder (open-loop policy)

## How to run

```
# Tier 2A (CMA-ES)
python scripts/run/run_tier2a_full.py

# Tier 2B (PPO)
python scripts/run/run_tier2b_ppo.py
```

Both write to `results/tier2{a,b}_*/` and consume the calibrated pitch
geometry baked into `kinematic_evaluator.py`
(`DEFAULT_START_FRAME=280`, 93 mph fastball, release at x=-3.31 m).

## Design notes

- **Kinematic, not PD:** PD-controlled humanoids drift off the mocap
  during the swing and fall over. Kinematic playback gives a clean,
  realistic swing trajectory and is what produced the 76 mph milestone.
- **3 dims active, 2 deferred:** `barrel_roll_rad` and `plate_reach_m`
  require manipulating the bat-grip equality constraint; deferred to a
  follow-up. The `SwingResiduals.from_normalized` API still takes a
  5-vector — the runners just pass zeros.
- **Why bandit-style PPO:** the residual policy is open-loop. There is
  no per-step observation, so PPO sees a one-step contextual bandit
  with the same action space CMA-ES searches over.
