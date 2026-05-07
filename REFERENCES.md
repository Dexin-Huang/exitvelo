# Reference papers + repos

Pointers a collaborator needs to read or run alongside this repo. Grouped
by what they contribute. Verified URLs as of 2026-05-06.

## Methodology — what shapes our optimizer + env design

### vid2player3d (NVIDIA, SIGGRAPH 2023)
- Physically simulated tennis: high-level RL on a 32-dim MotionVAE
  latent + residual root/DoF, low-level imitation realizes the motion.
- The single closest sports-swing precedent in the literature. **The
  reward + termination + analytical-flight pattern we tried to port for
  the hybrid physics path comes from here.** Specifically:
  - `physics_mvae_controller.py:521` `compute_reward_return` — pre-contact
    `exp(-k_pos · |ball - racket|²) · exp(-k_phase · (phase - π)²)`,
    post-contact landing-target bonus.
  - `physics_mvae_controller.py:408` `_compute_reset` — does NOT terminate
    on contact; keeps stepping for post-flight scoring.
  - `humanoid_smpl_im_mvae.py:414` — racket COR=1.0, friction=0.8.
- Code (Isaac Gym, SMPL): https://github.com/nv-tlabs/vid2player3d
- Project page: https://research.nvidia.com/labs/toronto-ai/vid2player3d/

### ASAP — Aligning Simulation and Real-World Physics (LeCAR-Lab, RSS 2025)
- Two-stage humanoid skill learning: (1) phase-based motion tracker via
  PPO; (2) delta-action model that perturbs stage-1 outputs to close the
  sim-to-real gap.
- **The architectural template for our residual-on-mocap framing**, even
  though we ended up with CMA-ES / one-step PPO instead of full
  closed-loop policy gradients.
- Paper: https://arxiv.org/pdf/2502.01143
- Code (Isaac Gym + IsaacLab + Genesis + MuJoCo sim2sim):
  https://github.com/LeCAR-Lab/ASAP
- **Local checkout** at
  `../../asap/` (same project root, not in this repo). Audited in our
  Code Report — see `../../code-report-asap/codex_repo_audit.md`.

### RFC — Residual Force Control (Khrylx, NeurIPS 2020)
- Learn a residual virtual force on top of a PD-tracked mocap. The
  canonical "small correction on a kinematic reference" paper.
- The math reference for our `SwingResiduals` framing. Code is on legacy
  `mujoco-py` and not directly portable, so we ported the *idea*, not
  the implementation.
- Paper / repo: https://github.com/Khrylx/RFC

### DeepMimic (Peng et al., SIGGRAPH 2018) and MimicKit (Peng, 2025)
- The original "imitate mocap with RL on a physically-actuated character"
  paper. MimicKit is Peng's 2025 successor that consolidates DeepMimic +
  AMP + ASE + ADD + LCP under one Apache-2.0 codebase.
- The "north star" architecture for full physics RL (which we did not
  ship — see PLAN.md "hybrid physics" entry).
- DeepMimic: https://github.com/xbpeng/DeepMimic
- MimicKit (2025): https://github.com/xbpeng/MimicKit

## Backbone — same simulator + humanoid as us

### MoCapAct (Microsoft, 2022)
- Pretrained low-level skill experts that track ~3 hours of CMU mocap
  on the **same dm_control CMU humanoid we use**, plus a hierarchical
  controller for downstream tasks.
- Closest drop-in starting point if we ever rebuild Tier 2 on top of an
  actual learned tracker rather than kinematic playback.
- https://github.com/microsoft/MoCapAct

### dm_control CMU humanoid
- The 56-DoF skeleton our env sits on. Joint names + qpos layout
  documented inline in `src/env/cmu_batting_env.py` and
  `src/motion/cmu_replay.py`.
- https://github.com/google-deepmind/dm_control

## Physics — the contact + flight model

### Nathan, A. M. (2003). "Characterizing the Performance of Baseball Bats"
- *American Journal of Physics* 71(2). The 1D bat-ball collision model
  `v_exit = (1 + q) · v_bat + q · v_pitch` with q ≈ 0.18 at the sweet
  spot. Implemented in `src/env/contacts.py` as `nathan_exit_speed` /
  `nathan_exit_velocity`.
- Author site (free copies of all his papers):
  https://baseball.physics.illinois.edu/

### Bridson (2015). "Fluid Simulation for Computer Graphics" (CRC Press)
- Used for the air-drag model on the batted-ball flight integrator
  (`integrate_ball_with_drag`): `a_drag = -0.5 · ρ · Cd · A / m · |v| · v`
  with the standard baseball constants.

## Validation — real-world data we'd compare against

### MLB Statcast — exit velocity + launch angle distributions
- Public batted-ball metrics for every MLB at-bat. Use to bound the
  plausibility of our exit_mph / launch_deg outputs.
- https://baseballsavant.mlb.com/statcast_search

### OpenBiomechanics Project (Driveline Baseball)
- Open-data mocap of professional + amateur swings, with bat speed,
  pelvis / trunk angular velocity, and kinetic-chain timing.
- The right comparison set for "did our intervention move the swing
  closer to a top-tier human swing or further from it?"
- https://www.openbiomechanics.org/

### Welch et al. (1995); MacWilliams et al. (1998); Escamilla et al. (2009)
- Classic kinesiology papers on the kinetic-chain order in baseball
  swings: pelvis → trunk → lead arm → hands → bat.
- Use as falsifier: if our optimized residual reverses this order, the
  result is a sim artifact (see SPEC.md "Falsifiers" section).

## Project paper-review trail (for context)

Our presentation paper for COMS6998E was ASAP. Audit notes and slides
live at `../../code-report-asap/`. Other reviews submitted:
- π₀ (VLA Flow Model)
- π₀.₆⋆ (Self-Improving VLA)
- Ctrl-World (Generative World Model)
- Large Video Planner

These don't bear directly on the batting project but are the wider
context for what we considered as backbones / inspiration.

## Cloud + tooling references

### RunPod
- Where we'd train PPO / DeepMimic-style at scale. RTX 4090 community
  pod is ~$0.34/hr (May 2026). Not used yet because the kinematic
  pipeline runs in 2-3 minutes locally.
- https://www.runpod.io/

### MuJoCo Playground (mujoco_playground / MJX)
- The right place to look if we ever port the env to GPU JAX for
  large-batch training.
- https://playground.mujoco.org/
