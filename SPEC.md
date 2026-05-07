# Exitvelo — Tier 2 Execution Spec

Source of truth for implementation. Five review checkpoints A-E. We do not
start the next stage until the previous one passes.

PLAN.md = strategy. SPEC.md = execution. Drift between the two = bug.

---

## 1. Env refactor (`src/env/cmu_batting_env.py`)

### Step 1.0: extract ball physics into `contacts.py` FIRST

**Verified ground truth:** `contacts.py` today only has
`detect_bat_ball_contact(model, data, *, bat_geom_ids, ball_geom_id)`
(`src/env/contacts.py:5`) which scans `data.ncon`. There is **no Nathan
resolver, no drag integrator, no ground-bounce code** in `contacts.py`.
That code lives ad-hoc inside `scripts/render_reel.py`:
- `Q_NATHAN = 0.18` at `render_reel.py:26`
- `ball_step_drag(pos, vel, dt)` at `render_reel.py:29`
- Nathan exit speed formula at `render_reel.py:120`
- ground-bounce + roll loop at `render_reel.py:223`

Before any of the rest, lift those functions out of `render_reel.py`
into `src/env/contacts.py` (or a new `src/env/physics.py`):
```python
def nathan_exit_velocity(bat_vel, ball_vel, q=0.18) -> ndarray
def integrate_ball_with_drag(pos, vel, dt, drag_k=...) -> (pos, vel)
def integrate_ball_until_landing(pos, vel, dt, max_t) -> traj
def carry_distance_ft(traj) -> float
```
`render_reel.py` rewires to import these. **Smoke before moving on:**
the 76 mph milestone reproduces from these new functions. Otherwise we
have a regression before we even start the env refactor.

### Replace MuJoCo contact with Nathan analytical contact

`mj_step` contact stays as telemetry only. The known failure: ball passes
the bat in ~5 steps with no contact event (`mj_step` at line 273 ->
`detect_bat_ball_contact` at line 278; verified). Fix: each step, do
swept ball-vs-bat segment detection using bat pose before/after, and on
intersection call the new `nathan_exit_velocity()` from contacts.py,
followed by the new `integrate_ball_with_drag()`.

### Reward (per step)

```
r_t = w_contact*R_contact + w_exit*R_exit + w_launch*R_launch
    + w_carry*R_carry + w_track*R_track + w_ctrl*R_ctrl + w_miss*R_miss
```

Starting weights:

| term | weight | shape |
|---|---|---|
| `w_contact` | +5.0 | 1 if contact this step else 0 |
| `w_exit` | +0.05 | `max(0, exit_mph - 40)` |
| `w_launch` | +2.0 | `exp(-((launch_deg - 28) / 18)^2)` |
| `w_carry` | +0.02 | `carry_ft` (terminal) |
| `w_track` | -0.001 | `||qpos - qpos_ref||^2` |
| `w_ctrl` | -0.0001 | `||action||^2` |
| `w_miss` | -2.0 | 1 only at end without contact |

Carry = first ground crossing after contact, before bounce, in feet.

### Termination

Replace the current `if contact: terminated = True`. New contract:
- continue stepping after contact until ball lands, exits domain, or timeout
- terminate on ground-cross, ball outside domain, humanoid fell, swing
  exhausted without contact (early-terminate to give the optimizer a
  short negative episode), or step timeout

### Pitch jitter at reset

**Verified ground truth:** today, pitch params are stored in the
constructor (`cmu_batting_env.py:64-93`) and `reset()` ignores `options`
(line 238) before calling `_launch_ball()` (line 248). So jitter requires
adding pitch sampling to `reset(options)`:
```python
def reset(self, *, seed=None, options=None):
    ...
    if self.pitch_jitter:
        self._sample_pitch(rng)
    self._launch_ball()
```

| param | range | distribution |
|---|---|---|
| pitch speed | 70-82 mph | uniform |
| plate-crossing height | 0.65-1.15 m | uniform |
| lateral plate offset | -0.25 to +0.25 m | uniform |
| release-time offset | -40 to +40 ms | uniform |

Spin/drag unchanged. Drag will live in the new contacts.py functions
(extracted from render_reel.py).

### Observation additions

**Verified ground truth:** current obs is 140 dims exactly
(`cmu_batting_env.py:46`, `:120-123`, `:348`). Layout:
joint_pos 56 + joint_vel 56 + root_pos 3 + root_quat 4 + root_lin_vel 3 +
root_ang_vel 3 + bat_tip_pos 3 + bat_tip_vel 3 + ball_pos 3 + ball_vel 3 +
ball_rel 3.

Today's obs uses `bat_tip`, not `bat_sweet`. The `bat_sweet` site
exists at `assets/mujoco/cmu_batting_scene.xml:229` (offset
`pos="0 0 0.65"` on the bat) but no sensor exposes it. Either add a
framepos/framelinvel sensor pair for `bat_sweet` (parallel to lines
426-428) or query it via `data.site("bat_sweet").xpos` directly in obs.

Append to existing 140-dim obs:
- `swing_phase` (scalar, 0->1 across the mocap clip)
- `sin(2*pi*phase)`, `cos(2*pi*phase)`
- ball pos/vel relative to bat sweet spot (replaces or supplements ball_rel)
- bat sweet-spot world pos/vel (new)
- current residual vector (5 dims)
- `has_contact` flag
- time since contact (clamped)

### Things to remove

- `reward = 0.0` literal (line 315)
- `if contact: terminated = True` (line 301)
- single hardcoded pitch (constructor must accept jitter ranges or sampler)

### Checkpoint A — env refactor

Assert (in code):
```
assert contact_count == 1                 # exactly one analytical contact
assert max(reward_log) > 0                # reward fires on contact frame
assert all(np.isfinite(ball_pos_log))     # no NaN in post-flight
assert all(np.isfinite(ball_vel_log))
exit_mph, launch_deg, carry_ft = analyze(ball_pos_log)
assert 60 < exit_mph < 95                 # in baseline regime
assert 10 < launch_deg < 50
assert carry_ft > 100
```
Smoke target: rerun the `cmu_milestone_results.json` config in the new env
and produce comparable exit/launch/carry to the analytical baseline.

---

## 2. Action space — 5 named scalars

Stored in physical units, normalized to `[-1, 1]` for the optimizer.

```python
@dataclass
class SwingResiduals:
    swing_timing_s:   float   # [-0.08, +0.08]
    hip_fire_rad:     float   # [-0.30, +0.30]
    uppercut_rad:     float   # [-0.22, +0.26]
    barrel_roll_rad:  float   # [-0.25, +0.25]
    plate_reach_m:    float   # [-0.10, +0.10]
```

### Per-scalar contract

| name | unit | range | what it does |
|---|---|---|---|
| `swing_timing` | s | ±0.08 | shifts mocap lookup time before PD target construction |
| `hip_fire` | rad | -0.30..+0.30 | adds pelvis/trunk yaw residual after retargeting |
| `uppercut` | rad | -0.22..+0.26 | trunk/lead-shoulder pitch residual (changes attack angle) |
| `barrel_roll` | rad | ±0.25 | rotates bat grip frame around handle axis, recompute right-hand IK |
| `plate_reach` | m | ±0.10 | translates bat/hand target toward/away from plate, recompute IK |

### Application order (in `cmu_tracking_controller.py`)

**Verified ground truth:** today the controller has only `get_action()`
(`cmu_tracking_controller.py:38`, `:44`) which returns 56 joint targets
straight from `replay.get_action()`. There is **no IK at runtime** — the
right hand is held to the bat handle by a MuJoCo `equality connect` in
the scene XML (`cmu_batting_scene.xml:349`). IK code only exists inside
`render_reel.py:80` for offline kinematic playback.

Also, `CMUMocapReplay.get_action()` increments `step_idx` on every call
(`cmu_replay.py:140`), which means the controller is stateful. Need
`replay.qpos_at(t)` (a pure index lookup) instead of `get_action()` for
deterministic residual application.

Refactor order:
1. Add `replay.qpos_at(t_seconds)` pure lookup, leave `get_action()` alone
2. Build a `SwingResiduals -> qpos_modifier` layer in the controller
3. For `barrel_roll` / `plate_reach`: do NOT add new IK at runtime.
   Modify the bat handle anchor (the equality connect's `anchor` attr or
   `body2` offset) at reset time, OR apply the residual as a small
   adjustment to the connect's anchor each step. This piggybacks on the
   existing constraint instead of writing IK.
```python
q_ref = replay.qpos_at(t + r.swing_timing_s).copy()
q_ref = apply_pelvis_trunk_yaw(q_ref, r.hip_fire_rad)
q_ref = apply_uppercut_pitch(q_ref, r.uppercut_rad)
adjust_bat_grip_anchor(model, data, r.barrel_roll_rad, r.plate_reach_m)
tau = pd_control(qpos, qvel, q_ref)
```

Tier 2A: all five constant per episode (open-loop).
Tier 2B: per-step targets, first-order filter + per-step rate limit.

### Checkpoint B — action space unit tests

For each scalar, set to {min, 0, max}, run a deterministic episode:
```
assert all(np.isfinite(qpos_log))            # no NaN qpos
assert all(np.isfinite(torque_log))          # no NaN torques
for j in limited_joints:
    assert qpos[j].min() >= jnt_range[j, 0]
    assert qpos[j].max() <= jnt_range[j, 1]
assert no_flipped_bodies(physics_log)         # check body orientations
assert hand_to_handle_dist < 0.05             # IK didn't break grip
```
Document expected limits: late/early whiff for `swing_timing`, over-rotation
for `hip_fire`, ground-ball/pop-up tendencies for `uppercut`, handle
misalignment for `barrel_roll`, plate-side miss for `plate_reach`. None
may produce NaN, flipped body, or broken hand-bat attachment.

---

## 3. Tier 2A — CMA-ES on the 5 named residuals

### Objective

Minimize:
```
J(theta) = -mean(carry_ft) + 25*no_contact_rate
         + 0.2*mean(|launch_deg - 28|)
         + 0.01*joint_limit_penalty
```

### Evaluation protocol

- Generations 0-10: 1 deterministic seed per candidate (fast search)
- Generations 11+: 3 jittered seeds per candidate (variance estimate)
- Total budget: ~1000 rollouts (10 gens * 10 pop * 1 seed + 30 gens * 10 pop * 3 seeds)

### CMA-ES hyperparameters

- Library: `cma` Python package (`pip install cma`)
- Dimension: 5
- Bounds: normalized `[-1, 1]`, denormalize per-scalar to physical ranges
- Initial mean: zeros
- Initial sigma: `0.35`
- Population size: `10`
- Max generations: `40`
- Restart criterion: best objective improves <1 ft over 8 generations

### Logging

Per generation -> `results/tier2a_cma/generations.csv`:
- best params (5)
- best objective
- mean params (5)
- sigma
- contact rate
- carry / exit / launch (mean over seeds for the best candidate)

Save best rollout: trajectory JSON + MP4 in `results/tier2a_cma/best/`.

### Plotting

- objective vs generation
- per-parameter trace (5 lines)
- carry/exit/launch scatter
- best ball trajectories overlaid against the 76 mph baseline

### Checkpoint C — CMA-ES converged

```
assert plateau_in_last_5_generations(best_obj_log)
assert contact_rate_at_best > 0.80         # over jittered eval seeds
assert mean_carry_at_best > mean_carry_at_zero + 20  # in feet
# Negative control: bat delayed 0.30s, ball 0.30m off plate
assert mean_carry_negative_control < 50   # near-zero
```
Best params describable in baseball terms (one-sentence each).

---

## 4. Tier 2B — PPO on the same action space

### Observation

PPO actor sees:
- mocap phase + sin/cos
- ball pos/vel relative to bat sweet spot
- bat sweet-spot pos/vel
- current residuals (5)
- previous action (5)
- `has_contact` flag

### Episode

One swing per episode: reset (with pitch jitter) -> swing -> contact or miss
-> post-contact flight -> terminate on ground/timeout.

### Action

Per-step 5-d normalized action target. Smoothed:
```python
target = denorm(action)
residuals = 0.85 * residuals + 0.15 * target
residuals = rate_limit(residuals, per_step_max=0.02)
```

### Reward

Section 1 reward + dense pre-contact shaping near expected-contact phase
(reduce bat-ball sweet-spot distance, align bat velocity with intended
field direction). Terminal carry stays dominant.

### Network

SB3 `MlpPolicy`, `net_arch=dict(pi=[128,128], vf=[128,128])`, tanh.

### Hyperparameters

| param | value |
|---|---|
| `learning_rate` | 3e-4 |
| `n_steps` | 1024 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` | 0.995 |
| `gae_lambda` | 0.95 |
| `ent_coef` | 0.01 |
| `clip_range` | 0.2 |
| smoke total_timesteps | 100,000 |
| full total_timesteps | 1-2,000,000 |

### Parallel envs

- Local 2060: 4 parallel envs (CPU-bound, dm_control)
- RunPod 4090: 16-32 parallel envs depending on vCPU count

### Curriculum

- 0-100k: no pitch jitter
- 100k-400k: half jitter
- 400k+: full jitter

Initialize residual state at zero.

### Checkpoint D — PPO trains end-to-end

```
assert nan_count == 0
assert no_env_crashes
assert smoothed_reward_curve_nondecreasing_in_last_half
assert contact_rate_final > contact_rate_baseline
assert mean_carry_final > mean_carry_zero_residual + 10  # in feet, on 32 fixed seeds
```

---

## 5. Comparison + validation pipeline

### Comparison

Same evaluator, same pitch jitter distributions, same 64 fixed seeds:

```python
methods = {
    "baseline_analytical": load_final_baseline_json(),
    "zero_residual":       ZeroPolicy(),
    "cma_es":              FixedResidualPolicy(best_cma_params),
    "ppo":                 SB3Policy(checkpoint),
}
metrics = {n: evaluate_policy(p, seeds=range(64), jitter=True)
           for n, p in methods.items()}
```

Metrics: contact rate, carry ft, exit mph, launch deg, lateral spray,
peak joint velocity, residual magnitudes, min bat-ball distance, reward.

### Validation tests

- **Negative control:** wrong timing + outside pitch -> low carry
- **Pitch jitter robustness:** carry distribution across reset ranges
- **Kinematic-chain order check:** peak angular velocity order
  pelvis -> trunk -> arm -> bat
- **Joint-velocity caps:** vs literature plausible limits
- **Statcast / Driveline comparison:** if data accessible by report freeze;
  otherwise use the 76 mph analytical baseline as internal reference and
  state the limitation in the report

### Main report figure

Four panels:
1. Ball trajectories (baseline + CMA + PPO + zero, 1 line each)
2. Carry/exit/launch bars with confidence intervals across the 64 seeds
3. Residual parameters in baseball terms (5 named bars)
4. Validation panel: negative-control failure to produce carry

### Checkpoint E — comparison + validation ready

```
assert one_script_regenerates_main_figure()
assert one_script_regenerates_metrics_table()
assert cma_and_ppo_use_identical_seeds_and_jitter()
assert all_validation_tests_emit_pass_or_explicit_caveat()
assert baseline_76mph_shown_as_reference_line()
```

---

## 6. Repo layout

Refactor `cmu_batting_env.py` in place. Refactor `contacts.py` to add
the Nathan + drag + bounce functions (extracted from `render_reel.py`).
Refactor `src/eval/evaluate.py` (already exists) to expose the
one-swing terminal-metrics protocol the runners need, instead of
creating a new `src/optim/evaluator.py`.

```
src/env/cmu_batting_env.py         <- refactored: swept contact + reward + jitter
src/env/contacts.py                <- EXTENDED: + nathan_exit_velocity,
                                                   + integrate_ball_with_drag,
                                                   + integrate_ball_until_landing,
                                                   + carry_distance_ft
src/eval/evaluate.py               <- refactored: one-swing terminal metrics
                                                  (carry, exit, launch, contact,
                                                   spray, peak_joint_vel)
src/optim/cma_runner.py            <- Tier 2A
src/optim/ppo_runner.py            <- Tier 2B
configs/task_v2.yaml               <- env, reward weights, jitter ranges
configs/cma.yaml                   <- CMA-ES search config
configs/ppo.yaml                   <- SB3 PPO config
results/tier2a_cma/
results/tier2b_ppo/
scripts/plot_tier2_results.py
```

`render_reel.py` rewires to use the new contacts.py functions instead
of inline. Untouched: `scripts/run_cmu_reference_hit.py`,
`results/final/cmu_milestone_results.json`.

---

## 7. Risks + bail-out

**Most likely failure:** contact-model validity dominates. CMA-ES and PPO
will exploit any weakness in Nathan/drag/bat-pose bookkeeping, producing
report-looking carry from physically suspect motion.

**Hidden failure mode:** residual transforms break the kinematic chain
while preserving a plausible bat trajectory. Ball improves, body sequencing
becomes indefensible.

**Bail-out if Tier 2B doesn't converge in 2 days:**
1. Freeze CMA-ES as the main quantitative result.
2. Train a small supervised residual policy from CMA-ES evaluations
   (predict residuals from pitch params). Retains a learned component.
3. Report PPO as an end-to-end attempt with 100k smoke + honest failure
   analysis.

---

## Codex grounding (verified 2026-05-06)

Re-ran with `codex exec --sandbox danger-full-access` so Codex actually
read the repo. Corrections folded into Sections 1, 2, 6 above.

Verified facts:
- sim dt = 0.002, frame_skip = 5, control_dt = 0.01
- ball qpos `64:71`, qvel `63:69`
- obs is 140-dim, layout documented in Section 1
- `bat_sweet` site exists in XML at `:229` but no sensor exposes it
- `contacts.py` has only `detect_bat_ball_contact`; Nathan + drag +
  bounce live ad-hoc in `render_reel.py` and need to be extracted
- runtime hand-to-handle is a MuJoCo equality connect at scene XML
  `:349`, not IK; controller has no IK
- `CMUMocapReplay.get_action()` mutates step_idx on every call;
  determinism requires a pure `qpos_at(t)` lookup
- env reset ignores `options`; pitch is constructor-only
- `src/eval/evaluate.py` exists and conceptually owns evaluator role

Order of work locked by these facts:
1. Extract Nathan + drag + bounce from `render_reel.py` into
   `contacts.py`. Reproduce 76 mph milestone from the new functions.
2. Refactor `cmu_batting_env.py` (swept contact, reward, no
   terminate-on-contact, pitch jitter at reset, sweet-spot obs).
3. Add `replay.qpos_at(t)` pure lookup, then plumb the 5-scalar
   residual through controller without adding runtime IK (use the
   existing equality-connect anchor adjustment).
4. Refactor `src/eval/evaluate.py` to return one-swing terminal metrics.
5. Build `src/optim/cma_runner.py` (Tier 2A).
6. Build `src/optim/ppo_runner.py` (Tier 2B).
