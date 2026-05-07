# Exitvelo — Plan

## North star

RL as a hypothesis generator over low-dimensional, named swing-timing
interventions. Validate by feasibility, literature, and counterfactual tests.

If we can't describe what the policy learned in 5–10 baseball terms, the
result is animation, not coaching.

This is a **Deep Learning for Robotic Manipulation** project. RL is the
headline.

## Tree

```
TIER 0 ─ Baseline (DONE)
         Mocap replay → 76 mph, 31°, 295 ft. Nathan contact + air drag.
         Single hand-tuned config (one lucky pitch + start frame).

TIER 2A ─ CMA-ES on 3 named residuals (DONE 2026-05-06)
         baseline 177.7 ft -> optimum 183.8 ft (+6.1 ft, +3.4%)
         exit 62.4 -> 63.8 mph; bat speed 17.4 -> 17.9 m/s (+3.2%).
         Dominant intervention: hip_fire = -12.5 deg
         (early pelvis-trunk separation).
         Saved: results/tier2a_cma/full_v1/best.json

TIER 2B ─ PPO on the same 3-d action space (DONE 2026-05-06)
         50k env steps, one-step bandit env.
         optimum 184.05 ft (matches CMA-ES within noise)
         hip_fire saturated at bound (-17 deg); uppercut +5.9 deg
         CMA-ES vs PPO converge to ~same carry but different residual
         configurations -> ridge/manifold of equivalent solutions.
         Saved: results/tier2b_ppo/best.json

TIER 2 ─ PPO on a constrained named action space  ← MAIN DELIVERABLE
         Action: 4-5 named params (stride, pelvis/trunk/wrist phase
         offsets, bat attack angle, etc.).
         Reward = exit velocity − energy − miss penalty.
         Pitch randomization across timing/location/speed jitter.
         Story: "RL discovers a multi-parameter combination that holds
         up under pitch jitter, intervention readable in baseball terms."
         GATE: training converges, intervention describable in <=5
         baseball terms, survives a negative-control sanity check.
         If we ship anything, it's this.

TIER 3 ─ Full residual joint-torque policy (stretch)
         RFC-style residual on PD-tracker.
         Mandatory ablations: fixed-gain, energy-matched, pitch jitter,
         body-group, torque-rate.
         Counterfactual: delete intervention from RL, add to baseline.
         GATE: policy survives >=3 of 5 ablations.
         If Tier 2 lands clean, this is "we unlocked the full action
         space and the same intervention re-emerged (or a different one,
         with caveats)."

TIER 4 ─ Style preservation via AMP (stretch)
         Discriminator on mocap reference. Compare with/without AMP.

TIER 5 ─ Stretch
         Pitch type variation. Comparison vs Driveline / OpenBiomechanics.
         Multi-batter robustness. Coaching cue extraction.
```

## Operating rule

Climb until Thursday. Build slides with whatever tier we reached. Same
again for the report on Sunday. No further scheduling — just go.

Floor for ship: a working Tier 2 policy with one interpretable
intervention and the negative-control sanity check passed.

## Tier 1 (sensitivity sweep) was killed

Codex review concluded fixed-pitch OAT is structurally broken: bat-ball
contact is a near-measure-zero timing event; perturbing the swing on a
fixed pitch mostly measures phase miss, not sensitivity. All-zero
contacts in the screening confirmed it. Tier 1 motivation gets folded
into Tier 2's action-space justification (cite the kinematic-chain
literature directly: Welch / MacWilliams / Escamilla).

## Tier 2 prep — env is NOT RL-ready as it stands

Critical work before training:

1. **Baseline-contact smoke test.** Reproduce a known-contact reference
   and assert it contacts in the same env/controller path RL will use.
2. **Env refactor.** Currently:
   - `reward = 0.0` literal
   - terminates on contact (kills post-flight signal)
   - single hardcoded pitch, no jitter
   - no dense pre-contact shaping
   Fix all four before training.
3. **Define Tier 2 action space** as a Gym Box of the named params.
4. **Smoke-train PPO** locally for a few thousand steps end-to-end.
5. **RunPod 4090** for the real run.

## Anti-patterns Codex flagged

- Don't shift body groups in raw qpos channels — breaks the spine.
  Use smooth time warps with overlap windows OR perturb low-dim swing
  features (hip-trunk separation, hand-path, bat-plane angle).
- Don't use integrated bat-tip KE as the energy proxy — it constrains
  the output channel you're optimizing.
- Don't use velocity² × inertia — scaled CMU humanoid inertias are
  unreliable.
- Don't claim "RL discovered technique" without: counterfactual both
  directions, parameter sweep around the intervention, body-group
  ablations, pitch-jitter robustness.
- Don't optimize exit velocity alone — penalize fouls, mishits, extreme
  launch angles, brittle contact.

## Falsifiers (any of these = no story)

- RL reverses normal kinetic-chain order (pelvis → trunk → arm)
- Segment angular velocities exceed reported human ranges
- Improvement comes from wrist/arm snap with no pelvis/trunk contribution
- Bat speed up but contact quality brittle to pitch jitter
- Same "technique" only works for one exact pitch

## Tech notes

- Backbone: MoCapAct (microsoft/MoCapAct) — same dm_control + CMU stack.
- Method ref: RFC (Khrylx/RFC) — residual-on-mocap math.
- Recipe ref: ASAP (LeCAR-Lab/ASAP) — two-stage training template.
- Cloud: RunPod RTX 4090 community pod (~$0.34/hr) when ready.
- Validation data: MLB Statcast (exit vel, launch angle), Driveline /
  OpenBiomechanics (bat speed, pelvis/trunk angular velocities,
  sequencing), Welch / MacWilliams / Escamilla swing studies.
