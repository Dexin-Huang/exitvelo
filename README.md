# Exitvelo

MuJoCo CMU-humanoid baseball-batting sim with CMA-ES and PPO optimization of
swing residuals. Goal: discover swing-timing interventions that improve
batted-ball distance.

COMS6998E (Deep Learning for Robotic Manipulation) course project, due
2026-05-11.

## Start here

Read in this order. ~60 min to be productive.

1. `GOAL.md` (10 min) - north star, research loop, evidence standards,
   current truth vs long-term ambition.
2. `PLAN.md` (10 min) — strategy, tier tree, what's done, what's killed,
   what's blocked.
3. `SPEC.md` (15 min) — execution spec for the env refactor + optimizers,
   five review checkpoints A-E, Codex-verified ground truth on the env
   internals.
4. `REFERENCES.md` (5 min) — papers + repos the project leans on
   (vid2player3d, ASAP, RFC, MoCapAct, Nathan 2003, OpenBiomechanics, ...)
   with one-line "what / why" each.
5. `results/tier2_comparison.png` — headline figure: CMA-ES vs PPO vs
   zero-residual baseline on the 3-d action space.
6. `src/optim/cma_runner.py` (5 min) — Tier 2A loop. Thin wrapper around
   `cma` + the kinematic evaluator.
7. `src/optim/kinematic_evaluator.py` (10 min) — the rollout pipeline that
   both CMA-ES and PPO call into. Project memory (top of file) explains why
   the rollout is kinematic-humanoid + analytical-ball instead of full
   physics.

## Setup

```
uv sync
```

A working venv already lives at `.venv/`. On Windows the project's Python is
`.venv/Scripts/python.exe` — use it directly so `import mujoco` and `cma`
resolve. Mocap data is downloaded once via
`python scripts/io/download_cmu_data.py` (writes
`data/raw/cmu_subject_124/124.asf`, `124_07.amc`, `124_08.amc`).

For RunPod / GPU setup (vectorized PPO at scale, future hybrid physics
work, MJX ports), see `RUNPOD.md` — it covers pod selection, headless
MuJoCo rendering, persistent storage, and per-job cost estimates.

## Reproduce key results

Three commands. Run from the repo root with the project Python.

```
.venv/Scripts/python.exe scripts/smoke/smoke_extracted_physics.py   # 3 s sanity check
.venv/Scripts/python.exe scripts/run/run_tier2a_full.py             # ~3 min  CMA-ES
.venv/Scripts/python.exe scripts/run/run_tier2b_ppo.py              # ~25 min PPO
```

| script | what it does |
|---|---|
| `scripts/smoke/smoke_extracted_physics.py` | reproduces the 76 mph milestone from the extracted Nathan + drag + bounce in `src/env/contacts.py` |
| `scripts/run/run_tier2a_full.py` | CMA-ES on 3 active residuals, popsize 10, 40 gens. Writes `results/tier2a_cma/full_v1/` |
| `scripts/run/run_tier2b_ppo.py` | PPO on the same 3-d action space, 50k env steps. Writes `results/tier2b_ppo/` |
| `scripts/plot/plot_tier2_comparison.py` | regenerates `results/tier2_comparison.png` |

## Open question — hybrid physics

The current Tier 2A and Tier 2B pipeline is **kinematic humanoid +
analytical Nathan ball**. The humanoid qpos is set directly each step
from the (residual-modified) mocap reference; the bat sweet-spot velocity
comes from `mj_forward` on that pose; the ball uses
`nathan_exit_velocity()` + `integrate_ball_with_drag()` from
`src/env/contacts.py`. This is fast (~10 ms/rollout) and produces
reproducible carry, but it isn't physical.

Hybrid physics — kinematic body + a real `mj_step` impulse on the bat-ball
contact — was attempted and is broken. The bat is held to the right hand
by a MuJoCo `equality connect` (`assets/mujoco/cmu_batting_scene.xml:349`),
and stepping that constraint alongside ball contact destabilizes the
solver (project memory: ball velocity is damped to zero within ~5 steps
after contact). The `physics_evaluator.py` module that hosted this attempt
has been removed; `scripts/smoke/smoke_physics_rollout.py` still imports
it as a tombstone. See PLAN.md "Tier 3" and SPEC.md §7 for the failure
mode.

Ideal end-state: full physics RL on an actuated humanoid (DeepMimic / ASAP
style). Pragmatic intermediate: hybrid (kinematic body + physical
bat-ball impulse). The hybrid is broken at the rhand-bat equality
constraint and that is the highest-leverage place for outside help.

## File layout

```
GOAL.md                     north star + research loop + evidence standards
PLAN.md, SPEC.md            strategy + execution spec (read these first)
src/env/                    cmu_batting_env.py, contacts.py (Nathan + drag + bounce)
src/motion/                 cmu_replay.py (AMC parser, qpos_at(t) lookup)
src/controllers/            cmu_tracking_controller.py, swing_residuals.py
src/optim/                  kinematic_evaluator.py, cma_runner.py,
                            swing_residual_env.py (PPO bandit env)
src/eval/                   evaluate.py, video.py
assets/mujoco/              cmu_batting_scene.xml (scene, plate, bat equality)
assets/meshes/              baseball_bat.{obj,stl} (Model 271)
configs/pitch_v1.json       one preset 93 mph pitch + bat target + replay frame
data/raw/cmu_subject_124/   downloaded mocap (gitignored)
scripts/smoke/              smoke tests (env contact, residuals, physics)
scripts/run/                CMA-ES, PPO, reference replay, eval sweep
scripts/io/                 download_cmu_data, render_reel, milestone figure
scripts/plot/               tier2 comparison + tier2a summary plots
scripts/calibrate/          baseline-contact + pitch-geometry calibration
results/final/              76 mph milestone (mp4, figure, json)
results/tier2a_cma/full_v1/ CMA-ES best.json, generations.csv, history.json
results/tier2b_ppo/         PPO best.json, history.json, ppo_model.zip
results/tier2_comparison.png    headline figure
results/legacy/             every iteration GIF/PNG documenting prior work
```

## Status

| tier | what | state |
|---|---|---|
| 0 | Mocap replay → 76 mph, 31°, 295 ft (`results/final/`) | done |
| 1 | OAT sensitivity sweep | killed (PLAN.md) |
| 2A | CMA-ES on 3 named residuals → 184 ft, 64 mph | done (`results/tier2a_cma/full_v1/`) |
| 2B | PPO on the same 3-d action space → 184 ft, 64 mph | done (`results/tier2b_ppo/`) |
| 2 (5-d) | PPO on the full 5-d named action space + pitch jitter | deferred (SPEC.md §4) |
| 3 | Full residual joint-torque policy + ablations | deferred (PLAN.md) |
| 4-5 | AMP style preservation, multi-batter, coaching cues | deferred (PLAN.md) |
| hybrid | Kinematic body + physical bat-ball impulse | broken (see "Open question") |

## Coordinate conventions

- `+Y`: pitcher → batter (ball flies in `+Y`)
- `+X`: lateral
- `+Z`: up
- `nq=71`, `nv=69`, `nu=56`. `bat_grip` hinge at `qpos[60]` is unactuated.
  Ball free joint at `qpos[64:71]`, `qvel[63:69]`.
