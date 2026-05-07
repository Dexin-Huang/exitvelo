# Exitvelo Goal Document

## 0. One-Sentence North Star

Can reinforcement learning and simulation optimize human physical performance by starting from human motion, searching beyond human intuition through massive simulated practice, and returning discoveries as techniques that real humans can understand, test, and learn?

## 0.1 Current Pivot

The next research loop should pivot from "kinematic residual proof" to the
actual DeepMimic-style recipe:

1. Imitation phase:
   train or reuse a physical tracking policy that can reproduce the CMU
   batting swing in MuJoCo without directly setting `qpos[t] = motion[t]`.
   Reward is primarily "stay close to the reference motion": joint-angle
   error, body/site position error, contact-window timing, uprightness, and
   bounded control effort. Output is a policy that can physically hold the
   body through the swing.

2. Task phase:
   warm-start from the imitation tracker, swap in or add a batting objective,
   and train residual actions for contact quality, exit velocity, and carry.
   The tracker supplies the human-level starting behavior; the task optimizer
   searches for better technique around it.

3. Style preservation:
   optionally add an adversarial style prior, motion-latent prior, or explicit
   tracking penalty so the optimizer cannot win by inventing a non-human
   contortion. A better swing must still look and measure like a plausible
   human swing.

This is now the operational target. The kinematic pipeline remains valuable
because it defines the action vocabulary, contact metrics, validation gates,
and report evidence. The missing piece is the physical tracker.

## 0.2 Active Autonomous-Loop Directive

If this document is used to restart an autonomous research loop, the loop
should treat the tracker-first pivot as the active objective.

The immediate goal is not to improve the existing kinematic residual numbers
by a few more feet. The immediate goal is to create the first physically
controlled version of the swing:

> CMU human batting animation -> imitation tracker -> physically generated
> swing rollout -> Exitvelo batting evaluator -> residual/task optimization.

This means the next loop should begin from the MoCapAct probe, not from a new
full-physics RL implementation. Full physics task RL from a blank policy is
too many problems at once: animation tracking, humanoid balance, bat control,
sparse contact, reward shaping, and simulator exploit prevention. The correct
sequence is to reuse or train a tracker first, then optimize around it.

### 0.2.1 First Concrete Action

The first concrete action was to unblock and run the Linux MoCapAct probe.

Current state as of 2026-05-07:

- The real Linux RunPod probe has been run.
- The official dm_control/MoCapAct CMU HDF5 does not contain `CMU_124_07` or
  `CMU_124_08`.
- `CMU_124_06` and `CMU_016_22` loaded and stepped, proving the Linux
  MoCapAct install/control path.
- The `CMU_124_06` fallback adapter now exports structurally into Exitvelo,
  but it is far from the batting target and produces no contact.
- A new custom data bridge now exists:
  `scripts/io/export_cmu_amc_to_mocapact_hdf5.py` converts tracked raw
  `data/raw/cmu_subject_124/124_07.amc` into a MoCapAct-compatible HDF5.
- The generated local target file
  `results/mocapact_custom_hdf5/CMU_124_07_from_amc_30hz.h5` has 361 steps at
  `dt=0.03`, passes `HDF5TrajectoryLoader` validation, and resets
  `MultiClipMocapTracking` on the target batting window with initial tracking
  error `0.0`.
- Therefore the active next action is no longer "find whether the official
  HDF5 has the target"; it is "run the Linux custom-`mocap_path` branch, then
  train a `CMU_124_07` clip expert if the custom path loads on Linux."

Current intended pod-side command:

```bash
git clone https://github.com/Dexin-Huang/exitvelo.git
cd exitvelo
BUILD_CUSTOM_MOCAP_HDF5=1 bash scripts/runpod/mocapact_probe.sh
```

The probe should test these candidate windows by default:

| candidate | window | purpose |
|---|---:|---|
| `CMU_124_07` custom HDF5 | `87-120` | exact Exitvelo batting contact target, mapped from 100 Hz frames `260-360` |
| `CMU_124_07` official HDF5 | `260-360` | historical failed probe; keep as evidence of official data absence |
| `CMU_124_08` | `260-360` | secondary local subject-124 raw clip |
| `CMU_124_06` | `0-189` | nearby subject-124 MoCapAct split fallback |
| `CMU_016_22` | `0-82` | known MoCapAct install/control clip |

Before launching a paid pod, verify that the public clone contains the current
probe payload and the custom HDF5 converter. For the custom `CMU_124_07`
branch, prefer the public-clone path; the old minimal upload path is not enough
unless it also carries the converter and tracked raw AMC input. Do not silently
launch a stale public clone.

Paid RunPod work is allowed conceptually because the research question needs
Linux/GPU execution, but it must be attached to a bounded experiment card:
command, expected runtime, success criteria, output files, and cleanup plan.
Use the cheapest sufficient GPU for probes and imitation debugging; use an
A100 only when the bottleneck is actual training throughput rather than setup,
packaging, or a missing adapter.

### 0.2.2 Branching Logic

After the probe, branch by evidence:

1. `CMU_124_07` loads and steps:
   adapt the exact target rollout into Exitvelo first.

2. `CMU_124_08` loads and steps:
   treat subject-124 loading as viable, then decide whether to retarget,
   supply custom `CMU_124_07` data, or use `124_08` as a temporary swing
   baseline.

3. `CMU_124_06` loads and steps:
   validate the adapter/intake path on the nearby subject-124 fallback while
   debugging why the exact target clip is absent or broken.

4. `CMU_016_22` loads and steps but subject-124 fails:
   MoCapAct itself is probably installed correctly; the blocker is batting
   clip coverage or data formatting.

5. No candidate loads:
   debug Linux dependencies, MoCapAct installation, and dm_control assets
   before doing any batting-specific RL.

6. A candidate loads but cannot track the swing well enough:
   preserve the report, then train a custom imitation tracker on the shortest
   viable contact-window snippet. Do not jump to unrestricted full-body task
   RL.

### 0.2.3 Tracker Training Recipe If Reuse Fails

If a custom tracker is needed, keep it deliberately small:

- reference: `CMU_124_07` around the batting contact window, or the nearest
  subject-124 fallback if the exact clip is unavailable
- policy objective: physically reproduce the reference swing
- imitation reward: joint angle tracking, root pose tracking, body/site
  position tracking, contact-window phase timing, uprightness, and bounded
  action effort
- initialization: reference-state initialization sampled through the clip
- curriculum: stance hold -> slow swing -> full-speed swing -> contact window
- first controller: PD target actions or residual target offsets, not raw
  unconstrained torques
- first success metric: non-falling rollout through the contact window with a
  bat sweet-spot trajectory close enough for the existing evaluator

Only after this tracker exists should the loop switch to task reward:
contact, exit velocity, carry distance, launch-angle sanity, pre-contact
miss-distance shaping, and robustness to small pitch/timing perturbations.

### 0.2.4 What Counts As Progress In The Next Loop

The next loop is successful if it produces one of these evidence-backed states:

- MoCapAct exact-target success:
  a `CMU_124_07` physical rollout is copied back, validated, and classified.

- MoCapAct fallback success:
  a nearby/control rollout proves the infrastructure, and the remaining
  blocker is narrowed to target-clip coverage.

- Custom tracker success:
  a trained or partially trained imitation policy holds the swing through the
  contact window and exports a usable rollout.

- Productive failure:
  the loop identifies a precise blocker, such as missing MoCapAct clip data,
  incompatible dm_control assets, failed tracking reward, adapter mismatch, or
  contact-evaluator mismatch, and records the next bounded experiment.

The loop is not successful merely because a pod launched or a training job
ran. It must come back with a report, artifacts, and a branch decision.

### 0.2.5 Required Notes While Running Autonomously

Keep the project legible while moving fast:

- append major actions, commands, results, and branch decisions to
  `AUTONOMY_LOG.md`
- update `MOCAPACT_PROBE.md` after any real probe result
- refresh `results/final/project_checkpoint.json` after meaningful state
  changes
- never turn a failed probe into a vague "RL is hard" note; classify the
  failure as install, data, tracking, adapter, contact, reward, or compute
- record when paid compute starts, why it starts, what it is expected to
  produce, and how it is cleaned up

The north star stays ambitious, but each loop must leave behind a smaller
piece of evidence that is true under review.

## 1. The Big Idea

Most robotics work asks:

> Can humans teach robots how to move?

This project asks the reverse:

> Can robots, simulations, and RL help humans discover better ways to move?

The long-term vision is "chess for physical skill." In chess, computers began by imitating human play, then surpassed human search, then changed how humans understand openings, tactics, and positional ideas. The analogous question for physical activity is:

> If a simulated body can start from human technique, run far more trials than any human could, and optimize under realistic constraints, can it discover refinements that become learnable coaching cues?

Baseball batting is the first test domain because it is:

- measurable: exit velocity, launch angle, carry distance, contact rate
- time-critical: tiny changes in timing matter
- biomechanical: performance depends on kinetic-chain sequencing
- familiar: results can be described in baseball language
- difficult: contact is sparse, fast, and physically unforgiving

The project is not merely "train a batter robot." The project is to build a loop where human motion is the prior, simulation is the search engine, RL or optimization proposes interventions, and biomechanical review turns those interventions into human-facing hypotheses.

## 2. Philosophical Frame

### 2.1 Human Skill As A Search Problem

Elite physical technique can be viewed as a point in a large space of possible movements. Human athletes search this space through practice, coaching, feedback, injury constraints, cultural knowledge, and biological limits.

That search is powerful but slow:

- one athlete gets only thousands or millions of lifetime repetitions, not billions
- experiments are expensive because fatigue, injury, and time matter
- coaching language can preserve tradition but also encode blind spots
- feedback is noisy because many variables change at once
- some counterfactuals cannot be tested safely or repeatedly in the real world

Simulation changes the search budget. A simulated agent can try millions of variations, fail cheaply, and produce a map of what helps, what hurts, and which changes interact.

The scientific question is whether that search can produce knowledge that survives the trip back to humans.

### 2.2 The Chess Analogy

The analogy is not that physical activity is exactly like chess. Chess has perfect information and exact rules. Human movement has noisy bodies, imperfect simulation, physiology, injury, perception, and environment.

The useful analogy is the loop:

1. Human experts provide the initial distribution.
2. A machine searches far beyond normal human experience.
3. The machine finds patterns that initially look strange or non-obvious.
4. Humans analyze, compress, and name those patterns.
5. The patterns become teachable.
6. Human practice changes because the search changed the theory.

For batting, the equivalent of a chess engine novelty might be:

- a slightly earlier pelvis-trunk separation pattern
- a different attack-angle adjustment under pitch jitter
- a timing compensation that preserves contact quality
- a swing-plane change that trades small bat speed for much better collision geometry
- a cue like "hold the torso closed longer, then release earlier through contact"

The project succeeds only if the discovered intervention can be expressed in baseball or biomechanics terms, not only as a vector of simulator parameters.

## 3. Core Research Question

Can an RL/simulation system discover physically plausible, interpretable improvements to a human-derived baseball swing that increase batting performance and can be translated into human-learnable coaching hypotheses?

This breaks into five subquestions:

1. Can we build a reliable simulated batting pipeline from human motion?
2. Can we define an action space that changes technique in interpretable human terms?
3. Can optimization or RL improve performance over the human-motion baseline?
4. Can the improvement survive validation tests that rule out simulator artifacts?
5. Can the optimized intervention be converted into a clear, testable human coaching hypothesis?

## 4. Current State

### 4.1 Repository State

The active project is `batting-project`.

Key files:

- `README.md`: operational project overview
- `PLAN.md`: strategy, tier tree, what is done, what is killed, what is deferred
- `SPEC.md`: execution spec and verified implementation facts
- `src/motion/cmu_replay.py`: CMU Subject 124 mocap replay
- `src/controllers/cmu_tracking_controller.py`: mocap tracking with named residuals
- `src/controllers/swing_residuals.py`: interpretable residual action space
- `src/env/contacts.py`: analytical Nathan exit velocity plus drag and bounce
- `src/optim/kinematic_evaluator.py`: active rollout evaluator for Tier 2
- `src/optim/cma_runner.py`: CMA-ES optimizer
- `src/optim/swing_residual_env.py`: PPO one-step bandit environment
- `scripts/run/run_tier2a_full.py`: full CMA-ES run
- `scripts/run/run_tier2b_ppo.py`: PPO run
- `scripts/plot/plot_tier2_comparison.py`: main comparison figure

### 4.2 Milestone Result

The milestone objective was to prove the pipeline:

- CMU Subject 124 swing replayed on a MuJoCo humanoid
- bat attached to the hand
- pitched ball synchronized to the swing
- contact produced under a non-random tracking baseline
- random policy fails as expected
- saved figure, video, and quantitative result

That milestone is effectively complete.

Saved milestone numbers:

- tracking policy contact rate: 100 percent over the saved fixed eval
- random policy contact rate: 0 percent
- exit velocity: about 76.5 mph
- launch angle: about 31.6 degrees

This result proves that the environment, motion source, contact pipeline, and reporting artifacts exist.

### 4.3 Current Tier 2 Result

The current final-project path is not full humanoid physics RL. It is:

- kinematic humanoid replay
- analytical ball propagation
- Nathan-style bat-ball exit model
- low-dimensional swing residual optimization
- CMA-ES and PPO comparison on the same residual space

Active residual dimensions:

- `swing_timing_s`
- `hip_fire_rad`
- `uppercut_rad`

Deferred residual dimensions:

- `barrel_roll_rad`
- `plate_reach_m`

Current optimization results:

- zero-residual baseline is a calibrated human-motion swing
- CMA-ES reaches about 183.8 ft carry
- PPO reaches about 184.1 ft carry
- both methods find approximately the same performance level
- dominant intervention is negative `hip_fire_rad`, interpreted as earlier or stronger pelvis-trunk timing adjustment

The current result is useful because it is interpretable and reproducible. It is limited because the humanoid is kinematic, not dynamically controlled.

### 4.4 Current Technical Truth

The honest technical state is:

- We can replay human motion.
- We can perturb that motion with named residuals.
- We can evaluate bat sweet-spot motion against a pitched ball.
- We can compute post-contact ball flight analytically.
- We can optimize residuals with CMA-ES.
- We can run PPO as a fair optimizer comparison over the same search space.
- We cannot yet claim full physics RL on an actuated humanoid.
- We cannot yet claim that the discovered technique would transfer to a real human.
- We can claim an early prototype of the larger loop: human prior -> simulated search -> interpretable intervention -> validation agenda.

### 4.5 Current Pivot Truth

The validated local system has answered the warmup question:

> Can a human swing animation be turned into an interpretable simulator-side
> technique search problem?

The answer is yes, with the explicit limitation that the body is still
kinematic. The next question is sharper:

> Can the same human swing become a physically controlled motion, and can the
> existing residual/search logic improve that physical motion?

That makes the tracker the central missing component.

Current tracker-readiness facts:

- MoCapAct is the best first reuse target because it already works in the
  CMU/dm_control humanoid ecosystem.
- The public Exitvelo repo is cloneable on a clean machine, and
  `scripts/runpod/mocapact_probe.sh` now supports the pod-side clone path.
- Local RunPod CLI and Python SDK readiness exists, but paid pod creation
  should still be treated as an explicit execution gate.
- The first probe is binary: can MoCapAct load/track `CMU_124_07` or a close
  subject-124 fallback through the batting contact window?
- If yes, we skip most custom imitation training and adapt the rollout into
  Exitvelo.
- If no, we start cheap custom imitation training on the same clip instead of
  jumping to full physics RL from scratch.

The near-term research claim should become:

> Human swing as imitation target -> physical tracker holds the motion ->
> residual/task optimization improves the held swing for batting metrics.

## 5. Desired End State

### 5.1 Course-Project End State

For the course project, the target is a defensible final report and presentation showing:

1. A working simulated batting pipeline initialized from human mocap.
2. A named residual action space grounded in baseball biomechanics.
3. CMA-ES and PPO optimization over that action space.
4. Quantitative improvement over zero-residual baseline.
5. Evidence that the optimized residual is interpretable.
6. Validation tests or explicit caveats around simulator limitations.
7. A clear argument that the project is a first step toward robot-assisted human technique discovery.

The course-project win condition is not "we solved batting." It is:

> We demonstrated a concrete loop where human motion is optimized in simulation and the result can be discussed as a human technique hypothesis.

### 5.2 Research-Program End State

The long-term target is a general system for physical skill discovery:

1. Start with human motion data.
2. Build or fit a simulated performer.
3. Define constraints that preserve human plausibility.
4. Use RL or optimization to search for performance improvements.
5. Detect whether improvements are real, robust, and non-artifactual.
6. Translate successful changes into coaching-level concepts.
7. Test those concepts with human athletes or higher-fidelity biomechanical tools.
8. Add the validated discoveries back into the system as new priors.

For baseball, the eventual system would answer questions like:

- What timing changes improve exit velocity without increasing miss rate?
- Which interventions survive pitch-speed and pitch-location variation?
- Which swing changes are robust across body types?
- Which changes preserve plausible joint velocities and kinetic-chain order?
- Can the system generate coaching cues that improve human practice?

### 5.3 The Ultimate North-Star Claim

The strongest future claim would be:

> Starting from human batting motion, a simulation/RL system discovered a physically plausible technique change that improved simulated performance, survived robustness and biomechanics checks, and produced a coaching cue that measurably improved real human or high-fidelity biomechanical performance.

The current project does not need to prove that full claim. It needs to make the first version of the loop real.

### 5.4 Next-Loop End State

The next autonomous loop should aim for the first physical version of the
pipeline:

1. Launch or rehearse the Linux MoCapAct probe.
2. Determine whether a free MoCapAct tracker can hold the batting clip.
3. If the free tracker works, export its rollout and compare it to the
   kinematic swing.
4. If the free tracker does not work, train a small custom imitation tracker
   on `CMU_124_07` or the nearest viable subject-124 fallback.
5. Feed the physical-tracker rollout into Exitvelo's existing bat sweet-spot
   and contact evaluator.
6. Run the current residual/table/search machinery on top of the tracker
   output, not on raw kinematic `qpos`.
7. Evaluate whether the physical residual improves contact/carry while
   preserving human-looking motion.

Near-term success is not a 500 m swing. Near-term success is a physical body
that can swing through the contact window and produce a trusted baseline for
residual optimization.

## 6. The Loop

The project should operate as a repeated research loop:

1. Research
2. Checkpoint
3. Review
4. Synthesize
5. Brainstorm
6. Experiment
7. Evaluate
8. Archive
9. Repeat

This loop is the engine of the project.

### 6.1 Research

Purpose:

- Understand the human-performance domain.
- Identify plausible technique variables.
- Avoid optimizing nonsense.
- Anchor claims in biomechanics, sports science, and simulation literature.

Inputs:

- baseball swing biomechanics literature
- kinetic-chain sequencing studies
- bat-ball collision models
- Statcast-style metrics
- Driveline/OpenBiomechanics-style measurements if available
- RL imitation literature such as DeepMimic, AMP, RFC, ASAP, MoCapAct
- observed behavior of the current simulator

Research questions:

- What does a plausible swing sequence look like?
- What are normal ranges for pelvis, trunk, arm, and bat velocities?
- What swing changes are coaches already known to use?
- Which variables are likely trainable by humans?
- Which variables are likely simulator artifacts?
- What metrics represent real performance rather than merely exploiting the model?

Outputs:

- candidate intervention list
- validation thresholds
- falsifiers
- paper/report citations
- updated action-space design

### 6.2 Checkpoint

Purpose:

- Freeze a narrow question before experimenting.
- Prevent open-ended wandering.
- Define what would count as success or failure.

Every experiment should have a checkpoint card:

- name
- hypothesis
- exact code path
- config
- metric
- success threshold
- failure threshold
- expected runtime
- output files
- review date

Example:

Name:

- `tier2_hip_fire_sweep_v1`

Hypothesis:

- Negative `hip_fire_rad` improves carry by increasing bat sweet-spot velocity near contact without breaking contact timing.

Metric:

- carry feet
- exit mph
- sweet-spot speed
- contact rate
- min bat-ball distance

Success:

- a smooth region of improved carry exists near the CMA/PPO optimum
- contact rate remains 100 percent on fixed pitch

Failure:

- optimum is a one-point spike
- improvement vanishes under small pitch jitter
- intervention creates implausible body motion

### 6.3 Review

Purpose:

- Treat results skeptically.
- Find artifacts before writing claims.
- Separate "the optimizer found a number" from "we learned something."

Review questions:

- Did the experiment actually run the intended evaluator?
- Were the seeds fixed or logged?
- Did the optimizer compare against the correct baseline?
- Is the improvement larger than noise?
- Is the result robust to small parameter changes?
- Is the result robust to pitch jitter?
- Does the motion remain plausible?
- Does the explanation match the measured kinematics?
- Is this an RL result, an optimization result, or merely a calibration artifact?

Review outputs:

- pass
- fail
- inconclusive
- follow-up experiment
- caveat for report

### 6.4 Synthesize

Purpose:

- Convert raw experiment logs into a coherent understanding.
- Look across experiments rather than one run at a time.

Synthesis questions:

- What pattern appears repeatedly?
- Which intervention dimensions matter?
- Which dimensions are redundant?
- Do CMA-ES and PPO find the same family of solutions?
- Is there a ridge or manifold of equivalent swings?
- Does the result support a baseball-language interpretation?
- What is the smallest sentence that explains the finding?

Potential synthesis statements:

- "Both optimizers converge to similar carry but with different residual settings, suggesting a ridge of equivalent timing solutions."
- "The most consistent improvement comes from earlier pelvis-trunk timing, not from pure uppercut."
- "The optimized swing increases sweet-spot speed by about 3 percent, enough to improve carry under the analytical flight model."
- "The result is interpretable but still kinematic, so it should be framed as a technique hypothesis rather than a validated physical prescription."

### 6.5 Brainstorm

Purpose:

- Generate the next useful experiment based on synthesis.
- Keep creativity disciplined by evidence.

Brainstorm categories:

- improve action space
- improve evaluator fidelity
- add validation
- add visualization
- add negative control
- add human-facing interpretation
- add robustness
- add paper-ready figure

Good brainstorm prompts:

- What would make this result more credible?
- What would falsify our preferred story?
- What single plot would make the result easier to understand?
- What would a skeptical reviewer attack first?
- What would a coach ask?
- What would make the intervention more learnable?
- What can we do in 30 minutes, 3 hours, or 1 day?

### 6.6 Experiment

Purpose:

- Execute one bounded test.
- Produce reproducible artifacts.

Experiment rules:

- one primary hypothesis per experiment
- named script or notebook
- saved config
- saved outputs
- fixed or logged seeds
- before/after baseline
- no manual result editing
- no claim without artifact

Experiment outputs:

- JSON metrics
- CSV history
- plot
- optional video
- short written interpretation

### 6.7 Evaluate

Purpose:

- Decide whether the experiment changes project belief.

Evaluation categories:

- Supports current thesis
- Weakly supports current thesis
- Neutral
- Contradicts current thesis
- Reveals a bug
- Reveals a limitation
- Suggests a better question

Evaluation should answer:

- What did we learn?
- What changed?
- What remains uncertain?
- What should we do next?

### 6.8 Archive

Purpose:

- Preserve evidence and prevent repeated confusion.

Every meaningful experiment should be archived with:

- date
- command
- git status if relevant
- config
- output path
- result summary
- verdict
- next action

Possible archive location:

- `results/experiment_log.md`
- `results/<experiment_name>/summary.md`
- `PLAN.md` for high-level state changes

## 7. Research Tiers

### Tier 0: Pipeline Proof

Question:

- Can we create a working human-motion batting pipeline?

Status:

- Done.

Evidence:

- CMU replay works.
- Bat/ball scene exists.
- Fixed pitch can be synchronized.
- Tracking baseline hits.
- Random baseline fails.
- 76 mph milestone artifact exists.

Claim:

- The environment and motion pipeline are real enough to support final-project experiments.

### Tier 1: Naive Sensitivity

Question:

- Can one-at-a-time residual sweeps reveal useful swing sensitivities?

Status:

- Killed.

Reason:

- Fixed-pitch contact is too sparse and timing-sensitive. One-at-a-time perturbations mostly measure whether the bat misses the ball, not whether the technique is better.

Lesson:

- The action space and evaluator must preserve contact and search structured interventions, not arbitrary perturbations.

### Tier 2A: CMA-ES Over Named Residuals

Question:

- Can black-box optimization improve carry using interpretable swing residuals?

Status:

- Done for 3 active dimensions.

Evidence:

- CMA-ES improves carry to about 183.8 ft.
- Contact rate remains 100 percent in the best recorded rollout.
- Best residuals are interpretable.

Claim:

- Low-dimensional technique search can improve the human-motion baseline in the current evaluator.

Limit:

- This is not RL.
- This is not full physics.
- This uses a fixed or tightly jittered pitch.

### Tier 2B: PPO Over Same Residuals

Question:

- Can PPO find similar interventions when given the same action space and evaluator?

Status:

- Done as a one-step bandit PPO comparison.

Evidence:

- PPO reaches about 184.1 ft best carry.
- PPO agrees with CMA-ES on the existence of an improvement.
- PPO finds a somewhat different residual configuration, suggesting a solution ridge.

Claim:

- RL-style optimization can participate in the same technique-discovery loop, but the current PPO setup is closer to policy-gradient black-box search than closed-loop motor control.

Limit:

- Observation is a constant placeholder.
- Episode is one step.
- PPO is not controlling a dynamic humanoid over time.

### Tier 2C: Validation And Interpretation

Question:

- Is the discovered residual a credible technique hypothesis or just an evaluator artifact?

Status:

- Needed next.

Candidate checks:

- residual sweep around optimum
- zero-out each residual dimension
- compare CMA and PPO optima
- pitch jitter evaluation
- negative controls
- kinematic-chain plausibility
- joint velocity sanity
- contact-quality analysis
- visualization of body and bat motion

Desired output:

- a report-ready validation panel
- a one-paragraph human-readable interpretation
- explicit caveats

### Tier 3: Hybrid Physics

Question:

- Can we keep the mocap body kinematic but use physical MuJoCo impulse for bat-ball contact?

Status:

- Broken or deferred.

Known blocker:

- Bat is attached via equality constraint, and stepping ball contact alongside that constraint destabilizes or damps the ball.

Why it matters:

- Analytical ball physics is useful, but hybrid physical contact would make the result more credible.

Possible future work:

- redesign bat attachment
- weld bat differently
- make bat part of hand body
- use a contact proxy
- tune solver/contact parameters
- run a simplified bat-only contact model first

### Tier 4: Dynamic Humanoid RL

Question:

- Can an actuated humanoid track human motion while optimizing batting performance?

Status:

- Active next research branch, but staged. The immediate work is physical
  tracking/imitation, not unconstrained full-body task RL.

Approach:

- first try MoCapAct pretrained or quickly trainable CMU clip experts
- if reuse fails, train a custom imitation tracker on the batting clip
- then add residual correction or task fine-tuning on top of the tracker
- preserve style with tracking penalties, adversarial priors, or motion
  latent priors
- keep energy, torque, joint-limit, and pitch-randomization constraints

Why it matters:

- This is the real version of "the robot starts from human level and improves technique."

Risk:

- The full problem is still hard; the tracker-first decomposition keeps the
  first experiment binary and interpretable.

### Tier 5: Human Translation

Question:

- Can optimized simulator interventions become learnable human coaching cues?

Status:

- Future research.

Path:

- compress residuals into baseball language
- identify measurable body markers
- create drills or cues
- test with video, wearable, markerless pose, or human coach review
- compare before/after human performance

This is where the project becomes more than simulation.

## 8. Evidence Standards

### 8.1 What Counts As A Real Result

A result is credible only if:

- it is reproducible from a command or script
- it beats a named baseline
- the metric is meaningful
- seeds and configs are known
- the result survives at least one perturbation or validation test
- the motion is visually and biomechanically plausible
- the conclusion is proportional to the evidence

### 8.2 What Does Not Count

The following are not enough:

- one lucky rollout
- a beautiful video with no metrics
- a metric improvement with no baseline
- an optimizer best value with no validation
- a simulator exploit
- a residual vector with no human interpretation
- a claim of RL discovery when PPO is only acting as a one-step optimizer

### 8.3 Required Claims And Required Evidence

Claim:

- "The pipeline works."

Evidence:

- milestone replay, contact, random vs tracking result

Claim:

- "Optimization improves performance."

Evidence:

- zero-residual baseline vs CMA/PPO carry and exit metrics

Claim:

- "RL finds a similar solution."

Evidence:

- PPO run on the same evaluator and action space as CMA-ES

Claim:

- "The intervention is interpretable."

Evidence:

- named residuals, sweep plots, zero-out ablations, visual explanation

Claim:

- "The technique may transfer to humans."

Evidence:

- not currently proven; must be framed as a hypothesis

Claim:

- "The system discovers human-learnable technique."

Evidence:

- future work unless backed by human or high-fidelity validation

## 9. Falsifiers

The project should actively look for reasons the story might be false.

A result should be rejected or caveated if:

- improvement disappears under small pitch jitter
- optimized swing only works for one exact pitch
- contact is produced by proximity threshold abuse
- improvement comes from a non-human body configuration
- joint angles exceed plausible limits
- joint velocities exceed plausible ranges
- kinetic-chain order becomes backwards or nonsensical
- bat speed increases but contact quality becomes brittle
- PPO and CMA only find isolated spikes rather than stable regions
- the result depends on hard-coded launch angle more than swing change
- the intervention cannot be expressed in human terms

## 10. Human-Learnability Filter

Every proposed technique should pass a human-learnability filter.

A useful intervention should be:

- observable
- nameable
- trainable
- measurable
- robust
- safe enough to test
- connected to existing coaching or biomechanics language

Examples of potentially learnable variables:

- swing timing
- pelvis-trunk separation timing
- attack angle
- hand path
- plate coverage
- stride timing
- barrel entry path

Examples of less useful variables:

- arbitrary qpos channel manipulation
- exploiting bat attachment constraints
- impossible joint speeds
- exact frame-level timing that no human can reproduce
- changes that only work because the pitch is frozen

## 11. Project Vocabulary

Use precise language.

Human prior:

- The CMU mocap swing. The system starts from a real human motion rather than random exploration.

Technique residual:

- A low-dimensional, named perturbation to the human swing.

Simulation search:

- The optimizer or RL agent exploring technique residuals.

Technique hypothesis:

- A human-readable explanation of an optimized residual.

Validation:

- Tests that determine whether the improvement is robust and plausible.

Simulator artifact:

- A result caused by model weakness rather than a real movement principle.

Human translation:

- The process of turning a simulator result into a cue, drill, or measurable coaching idea.

## 12. Main Narrative For The Final Report

The final report should tell this story:

1. Physical skill can be viewed as a search problem over technique.
2. Human motion provides a strong prior, but humans cannot exhaustively search all counterfactuals.
3. Simulation and RL offer a way to cheaply search possible variations.
4. We built a baseball batting pipeline from CMU human mocap.
5. We created a named residual action space for interpretable swing changes.
6. We compared CMA-ES and PPO on the same residual search problem.
7. Both methods improved carry over the baseline and converged to related interventions.
8. The strongest intervention suggests a timing change in pelvis-trunk sequencing.
9. The result is an early demonstration of a human-performance optimization loop.
10. The current limitation is fidelity: the active evaluator is kinematic plus analytical ball flight, not full dynamic humanoid physics.

The report should be ambitious in framing but conservative in claims.

## 13. Main Figure Set

Ideal final figures:

1. System diagram:
   human mocap -> simulated swing -> residual optimizer/RL -> ball metrics -> technique hypothesis

2. Pipeline proof:
   milestone swing frames or video stills showing humanoid, bat, ball, contact

3. Quantitative comparison:
   zero residual vs CMA-ES vs PPO for carry, exit velocity, bat speed

4. Residual interpretation:
   bar chart of optimized residuals with physical units and bounds

5. Search progress:
   CMA-ES best carry over generations and PPO best carry over steps

6. Validation:
   sweep or ablation showing the intervention is not a one-point accident

7. Caveat figure if needed:
   current kinematic/analytical pipeline vs desired full-physics future

## 14. Immediate Next Checkpoints

### Checkpoint 1: Freeze The Kinematic Warmup

Goal:

- Preserve the current validated kinematic residual result as the warmup
  evidence package.

Current claim:

- "We demonstrate a first human-performance optimization loop for baseball
  batting: starting from CMU human mocap, we optimize interpretable swing
  residuals with CMA-ES/PPO-style search, improving simulated carry while
  producing a human-readable timing hypothesis."

Pass condition:

- `scripts/run/reproduce_final_artifacts.py --skip-validation` passes.
- The bundle, report, checkpoint, RunPod readiness, and smoke gates remain
  internally consistent.

### Checkpoint 2: Run The MoCapAct Probe

Goal:

- Answer the binary tracker-reuse question.

Question:

- Can MoCapAct load, step, train, or reuse an expert for the batting contact
  window around `CMU_124_07`, `CMU_124_08`, `CMU_124_06`, or `CMU_016_22`?

Preferred command on a Linux pod:

```bash
git clone https://github.com/Dexin-Huang/exitvelo.git
cd exitvelo
bash scripts/runpod/mocapact_probe.sh
```

Pass condition:

- `results/runpod_mocapact_probe/mocapact_probe_report.json` exists.
- At least one candidate writes a valid `*_rollout.json`.
- Local intake and decision smokes classify the result without importing
  MoCapAct locally.

### Checkpoint 3: Decide Free Tracker Or Custom Tracker

Goal:

- Choose the physical tracking path based on probe evidence, not preference.

Decision table:

- `CMU_124_07` works: use the exact target rollout first.
- `CMU_124_08` works: use it to validate subject-124 loading, then retarget
  or supply custom `CMU_124_07` data.
- `CMU_124_06` works: validate the adapter path on a nearby subject-124
  fallback while debugging the target clip.
- `CMU_016_22` works only: MoCapAct install is valid, but batting/subject-124
  data coverage is the blocker.
- no candidate works: stop batting work and debug Linux/MoCapAct dependency
  installation.

Pass condition:

- The next branch is recorded in `MOCAPACT_PROBE.md`,
  `results/analysis/mocapact_probe_decision/summary.json`, and
  `AUTONOMY_LOG.md`.

### Checkpoint 4: Adapt Tracker Output Into Exitvelo

Goal:

- Make the physical-tracked motion usable by the existing evaluator.

Required adapters:

- validate rollout schema
- export a proxy trajectory with body/site paths
- compare proxy hand/root/bat sweet-spot paths against kinematic replay
- evaluate batting contact with the existing analytical evaluator

Pass condition:

- tracked swing error is bounded enough to be a credible baseline
- contact-window timing is interpretable
- any mismatch is measured instead of hidden

### Checkpoint 5: If Needed, Train A Custom Imitation Tracker

Goal:

- Build the missing tracker directly if MoCapAct cannot provide one.

Training recipe:

- reference motion: `CMU_124_07` around the contact window
- reward: joint angle tracking, body/site position tracking, root pose,
  contact-window timing, uprightness, and bounded action effort
- initialization: reference-state initialization near sampled clip phases
- curriculum: stance -> slow swing -> full-speed swing -> contact window
- first action space: PD targets or residual target offsets, not raw
  unconstrained torques

Pass condition:

- the humanoid holds the swing through the contact window without falling
- bat sweet-spot trajectory is close enough to evaluate
- training logs and checkpoints are copied back and summarized

### Checkpoint 6: Optimize The Physical Tracker

Goal:

- Replace kinematic residuals with residuals on top of the physical tracker.

Task reward:

- contact
- exit velocity
- carry distance
- launch-angle sanity
- bat sweet-spot speed near contact
- miss-distance shaping before contact

Guardrails:

- imitation/style penalty
- joint-limit and torque penalties
- plausible bat speed caps
- pitch-jitter validation
- reject non-human contortions even if they score

Pass condition:

- physical residual policy beats the physical tracking baseline
- improvement survives at least a small pitch/timing perturbation
- the learned change can be named in baseball language

## 15. Open Questions

Scientific:

- What makes a simulator-discovered movement insight credible?
- How much physical fidelity is required before a technique hypothesis is useful?
- Can a low-dimensional residual space capture meaningful human technique?
- When does optimization become coaching insight rather than curve fitting?

Technical:

- Can MoCapAct provide a free tracker for `CMU_124_07` or a close fallback?
- If not, what is the smallest custom imitation tracker that can hold the
  batting contact window?
- How close must the physical tracker be to the kinematic swing before the
  residual evaluator is meaningful?
- Should residual optimization act on named swing variables, tracker action
  deltas, phase offsets, or a learned motion latent?
- Can hybrid physical bat-ball contact be stabilized?
- Can the bat attachment be redesigned to support real contact impulses?
- Can PPO become closed-loop rather than one-step bandit?
- Can pitch jitter be expanded without losing all contact?
- Can the full 5-D residual space be wired safely?

Human translation:

- What would a coach call negative `hip_fire_rad`?
- Is the intervention visible in video?
- Can it be measured with markerless pose?
- Would a human be able to practice it?
- What drill would test it?

## 16. Risks

### 16.1 Overclaiming

Risk:

- Presenting the current result as solved human technique discovery.

Mitigation:

- Say "technique hypothesis" rather than "validated coaching cue."
- Be explicit that the current evaluator is kinematic and analytical.

### 16.2 Simulator Exploitation

Risk:

- Optimizer exploits contact threshold, fixed launch angle, or simplified ball physics.

Mitigation:

- run sweeps, negative controls, jitter, and visual review.

### 16.3 Weak RL Story

Risk:

- PPO is currently a one-step bandit, so reviewers may reject it as real closed-loop RL.

Mitigation:

- frame PPO honestly as a policy-gradient optimizer over the same residual search space.
- present the broader RL program as future work.
- emphasize that the course result compares two optimization methods within the technique-discovery loop.

### 16.4 Transfer Gap

Risk:

- Simulated intervention may not transfer to humans.

Mitigation:

- explicitly separate "simulated discovery" from "human validation."
- propose a concrete human-validation path.

### 16.5 Physical Fidelity

Risk:

- Kinematic replay hides feasibility constraints.

Mitigation:

- include this as the central limitation.
- define hybrid/full physics as the next major research tier.

## 17. Success Definitions

### Minimum Success

- Final report shows a working human-mocap batting pipeline.
- CMA-ES improves a named residual swing over baseline.
- PPO comparison exists.
- Result is explained honestly with limitations.

### Strong Success

- CMA-ES and PPO both improve baseline.
- Main residual intervention is validated by sweep or ablation.
- Figure set clearly communicates the loop.
- Caveats are explicit and technically credible.

### Excellent Success

- Intervention survives pitch jitter.
- Negative control passes.
- Human-readable cue is compelling.
- Report convincingly frames the work as the first step toward robot-assisted human technique discovery.

### Next-Loop Success

- MoCapAct probe produces a valid tracker rollout, or custom imitation
  training reaches a non-falling swing through the contact window.
- Physical tracker output is adapted into the Exitvelo evaluator.
- The physical-tracked baseline can be compared against kinematic replay with
  measured body/site error.
- Residual or task optimization runs on top of the tracker instead of directly
  editing kinematic `qpos`.
- The result is reported as physical-tracker evidence only if the body remains
  plausible and the improvement survives basic perturbation checks.

### Long-Term Success

- Full dynamic humanoid RL starts from human motion and discovers robust, physically plausible improvements.
- Discoveries transfer into human training or high-fidelity biomechanical validation.

## 18. The Final Thesis

The final thesis should be:

> Human physical performance can be treated as an optimization problem initialized by human expertise. In this project, we demonstrate a first version of that idea for baseball batting: a CMU human swing is replayed in simulation, perturbed through named technique residuals, optimized by CMA-ES and PPO, and translated back into a candidate baseball technique hypothesis. The current system is not yet a full physics coaching engine, but it establishes the loop needed to build one.

The next-loop thesis should be:

> The kinematic system proves the warmup loop. The next step is to convert the
> CMU swing animation into a physically tracked policy, then optimize residuals
> on top of that tracker so the discovered swing change is produced by a
> controlled body rather than by direct pose playback.

## 19. Decision Rules

When deciding what to do next, use these rules:

1. Prefer experiments that strengthen the central thesis.
2. Prefer validation over another optimizer run once a result exists.
3. Prefer interpretable residuals over high-dimensional opaque control.
4. Prefer honest caveats over fragile claims.
5. Prefer one reproducible figure over five unorganized logs.
6. Prefer human-language synthesis over raw parameter dumps.
7. Prefer falsification tests before finalizing claims.
8. Prefer a reused MoCapAct tracker over custom imitation training if the
   probe shows it can hold the batting clip.
9. Prefer custom imitation tracking over full task RL from scratch if reuse
   fails.
10. Prefer tracker-plus-residual optimization over giving PPO unrestricted
   full-body control.

## 20. One-Page Version

We are trying to answer whether simulation and RL can optimize human physical technique rather than only teaching robots to imitate humans.

Baseball batting is the test case. We start from a real CMU human swing, replay it in MuJoCo, attach a bat, synchronize a pitch, and measure ball-flight outcomes. Instead of asking an RL agent to discover batting from scratch, we ask it to search a small space of named, human-interpretable residuals around a real swing.

The current system has already proven the pipeline and produced a Tier 2 result. CMA-ES and PPO both improve simulated carry over the zero-residual swing in a kinematic-plus-analytical evaluator. The main intervention appears to involve pelvis-trunk timing, especially negative `hip_fire_rad`.

The key limitation is fidelity. The active result is not yet full physics RL. It is a fast, interpretable search system using kinematic humanoid replay and analytical ball flight. Therefore the right claim is not "we discovered a validated coaching cue." The right claim is "we built the first version of a loop that can turn human motion into simulated search and simulated search into human-facing technique hypotheses."

The next work should focus on the physical tracker pivot. The immediate
question is whether MoCapAct can give us a free or cheap tracker for the CMU
batting clip. If yes, we adapt that rollout into Exitvelo and optimize
residuals on top of it. If no, we train a small custom imitation tracker on
the same clip. The long-term goal is a general engine for physical skill
discovery: start from human level, simulate many more trials than humans can
perform, find better technique, and return it to humans in a learnable form.

## 21. Course Alignment

This project fits **Deep Learning for Robotic Manipulation** when framed as:

> A physics-based humanoid manipulation task in which a simulated embodied agent uses human demonstration and deep RL/optimization to improve bat-ball striking performance.

The course-facing technical problem is not only baseball. It is:

- contact-rich dynamic manipulation
- learning from human demonstration
- humanoid control
- action-space design
- reinforcement learning under sparse rewards
- simulation fidelity and sim-to-real-style caveats
- task reward vs imitation reward tradeoff
- interpretation of learned policies

### 21.1 Why This Is Robotic Manipulation

The robot is not manipulating a small object with a gripper. It is manipulating:

- its own body
- a bat as a tool
- a fast incoming ball through impact

This is still manipulation because the agent must control an object state through physical interaction. The manipulated outcome is the post-contact baseball trajectory:

- exit velocity
- launch angle
- carry distance
- spray direction
- contact quality

The bat is an extended tool. The humanoid must coordinate body motion to deliver the tool to a small region in space and time. That makes the task harder, not less relevant.

### 21.2 Why This Is Deep Learning

The long-term system uses deep learning in several places:

- imitation policies that track human motion
- PPO or similar policy-gradient RL for task optimization
- learned residual policies around human motion
- learned latent action spaces from human motion fragments
- possible learned predictors for ball arrival/contact windows
- possible learned style priors or discriminators

The current project has a smaller proof:

- PPO is already used as a policy-gradient optimizer over named residuals.
- The next stage is a deep motion-tracking policy.
- The final research target is full RL over a physics-based humanoid.

For the course, the safe phrasing is:

> We demonstrate the first stage of a deep-RL robotic manipulation pipeline: starting from human demonstration, we define an interpretable residual action space and optimize a contact-rich striking objective in simulation.

### 21.3 Why This Is Not Just Animation

Animation would be:

- replay a nice-looking swing
- show a video
- stop there

This project goes beyond animation because it asks:

- what variables can be changed?
- what objective improves?
- what baseline is beaten?
- what optimizer found the change?
- what validation rules out artifacts?
- what human-language technique hypothesis emerges?

The central distinction:

> Animation makes motion look plausible. This project uses motion as a starting point for performance search.

## 22. Why Not Full Physical RL Immediately?

The intuitive plan is:

> Take a human animation, put a robot at that state, and let RL run until it discovers a better swing.

That is the correct long-term vision, but it is missing a necessary middle
layer. The robot cannot simply "start from the animation" unless it first has
a controller that can physically reproduce the animation. The correct version
is:

1. train or reuse a physical imitation tracker for the animation
2. warm-start task optimization from that tracker
3. preserve style so the task optimizer cannot abandon human motion

That is why DeepMimic, AMP, vid2player3d, ASAP, RFC, and MoCapAct matter:
they all treat motion data as a prior or tracker, not as magic actuator
commands. Immediate full-physics RL is risky because several hard problems
stack on top of each other.

### 22.1 Animation Is Not Control

Human mocap provides pose trajectories:

- pelvis position
- joint angles
- body orientation
- timing

It does not provide:

- motor torques
- actuator commands
- stabilizing feedback
- contact forces
- balance corrections
- physically feasible bat-hand interaction

A kinematic replay can set `qpos` directly and look good. A physical humanoid cannot teleport its joints to the next mocap frame. It must produce forces that cause the motion while obeying dynamics.

This is why the full problem is much harder than "start from this animation."

### 22.2 Human Mocap May Not Be Feasible For The Simulator Body

The CMU subject and the simulated humanoid differ in:

- body proportions
- joint limits
- mass distribution
- muscle/actuator strength
- foot-ground contact
- balance dynamics
- hand geometry
- bat attachment

Even if the original human motion is physically valid for the human, it may not be physically valid for the MuJoCo humanoid.

That mismatch creates two common failures:

- the humanoid falls while trying to track the swing
- the controller learns ugly compensations that no longer look human

This is the same class of problem addressed by residual-force and sim-to-real papers.

### 22.3 Batting Has A Sparse, Brittle Reward

In batting, tiny errors matter:

- a few centimeters of bat-ball miss
- a few milliseconds of timing error
- a small launch-angle error
- a weak contact instead of square contact

Full RL from scratch would see mostly:

- misses
- falls
- unstable contacts
- no useful reward gradient

The agent would not reliably learn "better batting." It would more likely learn:

- to exploit contact thresholds
- to move the bat unnaturally
- to generate unstable solver impulses
- to overfit one exact pitch
- to sacrifice human plausibility for reward

### 22.4 Full Physics Combines Too Many Unknowns

Immediate full physics would require all of these to work at once:

1. mocap retargeting
2. stable humanoid tracking
3. bat attachment
4. bat-ball contact
5. reward design
6. exploration
7. pitch timing
8. policy architecture
9. compute budget
10. validation

If it fails, we would not know which subsystem caused the failure.

The staged path isolates uncertainty:

- first prove the swing and contact metrics
- then prove optimization over interpretable residuals
- then prove physical tracking
- then prove physical/hybrid contact
- then combine them

### 22.5 The Correct Principle

The correct principle is:

> Do not give RL more freedom than the evidence can interpret.

At the beginning, RL should only control things we can name and validate. Full-body torque control is a later stage, after we know what a meaningful improvement looks like.

## 23. Concrete Technical Path

This is the path from current state to the north star.

### 23.1 Stage A: Kinematic Technique Search

Status:

- active and mostly complete

Question:

- Can named residuals around human mocap improve batted-ball metrics?

System:

- CMU mocap replay
- kinematic humanoid
- bat sweet-spot velocity
- analytical Nathan contact
- drag/bounce ball flight
- CMA-ES and PPO over residuals

Why it matters:

- proves the technique-search concept
- keeps the action space interpretable
- gives us a result that can be explained in baseball language

Main risks:

- simplified contact model
- hard-coded launch-angle assumptions
- fixed or narrow pitch distribution
- kinematic body hides feasibility constraints

Required next validation:

- residual sweeps
- ablations
- pitch jitter
- negative control
- visual biomechanical review

Report claim if this is all we ship:

> We built and validated a first-stage human-performance optimization loop using kinematic replay and interpretable residual search.

### 23.2 Stage B: Physical Motion Tracking

Status:

- immediate next technical step

Question:

- Can a torque/position-control humanoid physically track the CMU batting
  swing without kinematic teleportation?

System:

- same CMU swing
- MuJoCo humanoid
- no optimization for ball distance yet
- imitation/tracking reward only:
  - joint angle error
  - body/site position error
  - root pose error
  - contact-window phase/timing error
  - uprightness and non-fall bonus
  - bounded action/energy penalty
- reference-state initialization across swing phases
- early termination on fall or extreme pose error
- PD target actions or residual target offsets before raw torques

Path 1, free tracker:

- Use MoCapAct pretrained CMU experts or clip-expert infrastructure.
- Probe `CMU_124_07` first, then `CMU_124_08`, `CMU_124_06`, and
  `CMU_016_22`.
- If a valid target/fallback rollout exists, export it immediately and adapt
  it into Exitvelo.
- This can skip most of the custom imitation-training phase.

Path 2, cheap tracker:

- Train a custom imitation policy on `CMU_124_07` or the closest subject-124
  fallback.
- Expected target budget: hours on a 4090/A6000/A100-class pod, not days,
  if the clip window is short and the action space is constrained.
- Use the same adapter gates as the MoCapAct path.

Success:

- humanoid remains upright
- bat stays attached
- swing resembles reference
- bat sweet-spot trajectory is close enough to kinematic replay
- tracking error is bounded
- exported rollout passes schema/hash/intake checks
- adapter comparison reports body/site error instead of hiding mismatch

Failure modes:

- humanoid falls
- swing is too weak
- bat path drifts away from ball
- arm or torso motion becomes unnatural
- training is too slow
- target clip is absent from MoCapAct split files
- subject-124 data loads but no expert exists
- dependency installation succeeds but the tracker cannot hold the contact
  window

Adjustment if failure:

- shorten the clip around contact
- train only stance-to-contact first
- increase imitation reward
- use reference-state initialization
- reduce action frequency
- use stronger PD targets
- freeze lower body temporarily
- use MoCapAct-style expert tracking if available
- retarget through a nearby subject-124 fallback before insisting on the exact
  clip
- train at slow speed, then increase speed
- use residual force/control ideas if pure tracking cannot satisfy dynamics

Report claim if this succeeds:

> We can turn the human batting animation into a physically controlled humanoid swing.

### 23.3 Stage C: Residual Physical RL

Status:

- target after physical tracking works

Question:

- Can RL improve batting while staying near the human swing?

System:

- physical tracking controller
- residual policy on top
- small named residuals, tracker-output deltas, or low-dimensional latent
  action
- task reward for contact, exit velocity, and carry
- imitation/style penalty so the swing stays human-looking
- energy and joint-limit penalties

Important design choice:

- RL should not start with full joint freedom.
- RL should adjust technique around the human prior.
- The first task policy should be a residual on top of the tracker, not a
  blank policy learning all motor control and batting at once.

Candidate action spaces:

1. named residuals:
   - timing
   - hip-fire
   - uppercut
   - barrel roll
   - plate reach

2. body-group residuals:
   - pelvis phase
   - trunk phase
   - lead arm phase
   - wrist/barrel phase

3. latent motor primitives:
   - learned from tracking policies or mocap fragments

4. tracker residual actions:
   - additive offsets to the tracker action
   - phase-conditioned deltas
   - residual-force style corrections for dynamics mismatch

Success:

- improved carry/contact over physical tracking baseline
- motion remains plausible
- improvement survives pitch jitter
- learned change can be named
- physical body produces the motion rather than direct `qpos` playback
- style/tracking metric degrades only within a predeclared tolerance

Adjustment if failure:

- reduce action dimension
- add curriculum
- increase dense pre-contact shaping
- train fixed pitch first
- warm-start from CMA/PPO kinematic residuals
- use supervised imitation of best CMA residuals before PPO
- use dense pre-contact bat-ball miss-distance shaping
- freeze most body groups and unlock degrees of freedom gradually
- fall back to CMA-ES on residual tracker parameters before PPO if gradient
  RL is too noisy

Report claim if this succeeds:

> RL improves a physically controlled human-derived swing while preserving interpretable technique structure.

### 23.4 Stage D: Hybrid Contact

Status:

- broken/deferred

Question:

- Can we keep the body mostly controlled while making bat-ball impact physical?

System:

- physical or kinematic body
- real MuJoCo bat-ball contact impulse
- improved bat attachment
- solver/contact tuning

Known blocker:

- current bat equality constraint destabilizes or damps contact.

Approach:

1. Make a simplified bat-only contact scene.
2. Verify ball rebound under a moving rigid bat.
3. Add hand/bat attachment without full humanoid.
4. Add kinematic humanoid.
5. Add physical humanoid.

Adjustment if failure:

- keep analytical contact for report
- use physical contact only as future-work analysis
- redesign bat as part of hand body
- weld bat to hand with simpler inertial properties
- tune contact margins/solver impedance
- use a proxy collision geom for the bat barrel

Report claim if this does not succeed:

> Full physical contact is the main fidelity limitation and the highest-priority next engineering target.

### 23.5 Stage E: Full Physics RL

Status:

- long-term north-star implementation

Question:

- Can a dynamic humanoid policy, initialized from human motion, discover robust and learnable batting improvements?

System:

- physical humanoid
- physical or validated hybrid contact
- imitation/style prior
- task reward
- pitch randomization
- action/energy limits
- robustness tests
- biomechanics validation

Training curriculum:

1. imitate stance
2. imitate swing without ball
3. swing at fixed ghost ball
4. contact fixed pitch
5. optimize fixed pitch
6. add timing jitter
7. add location jitter
8. add speed jitter
9. add multiple pitch types
10. optimize for performance and robustness

Success:

- beats physical tracking baseline
- robust over pitch distribution
- no obvious simulator exploit
- intervention remains human-readable

Long-term claim if this succeeds:

> Starting from human skill, a physics-based RL system discovered a superior batting technique and returned it as a human-learnable hypothesis.

### 23.6 Concrete Mapping From Current Code To The Pivot

| Current system | Tracker-first target |
|---|---|
| `qpos[t] = motion[t]` kinematic playback | imitation tracker physically holds the motion |
| CMA-ES/PPO residuals applied to replayed `qpos` | CMA-ES/PPO residuals applied to tracker outputs, tracker targets, or phase-conditioned deltas |
| analytical contact from kinematic bat sweet-spot | same evaluator first, then hybrid/physical contact once tracker works |
| fixed warmup gain around the human swing | physical baseline swing, then residual/task gain over that baseline |
| human interpretation from residual vector | human interpretation from a physically produced swing difference |

This mapping keeps the work incremental. The existing evaluator, residual
tables, pitch jitter audits, learned descriptor policies, and report gates do
not get thrown away. They become the measurement and validation layer for the
physical tracker branch.

## 24. The Dream Scenario And The Scientific Standard

The intuitive dream scenario is:

1. We start with a human animation.
2. The baseline swing hits a solid ball.
3. The agent practices for a huge simulated budget.
4. The optimized swing hits massive home runs.
5. We inspect the difference.
6. The difference compresses into a coaching cue.

Example:

> Baseline animation: normal contact.
> Optimized policy: much farther carry.
> Human interpretation: add a toe tap, delay torso release, increase pelvis-trunk separation, or change barrel entry timing.

That is exactly the north star.

But the scientific standard is:

> It only counts if the improvement is plausible, robust, measurable, and explainable.

### 24.1 What A "500 m Bomb" Would Need To Prove

If a simulated swing suddenly hits 500 m, we should be skeptical first.

Questions:

- Did the bat speed become impossible?
- Did the ball get an impossible exit velocity?
- Did launch angle come from a hard-coded value?
- Did contact happen because of an inflated proximity threshold?
- Did the bat tunnel through the ball?
- Did the humanoid use non-human joint positions?
- Did the agent exploit the bat attachment?
- Did the result vanish with pitch jitter?
- Did air drag or bounce logic inflate distance?

If those checks pass, then the result becomes interesting.

If they fail, the result is still useful as a debugging signal:

> The optimizer found a weakness in the simulator.

### 24.2 From Giant Performance Jump To Coaching Cue

The desired translation pipeline:

1. Identify optimized residuals.
2. Compare optimized swing to baseline.
3. Measure body-segment timing.
4. Measure bat path and sweet-spot speed.
5. Measure contact point and launch vector.
6. Convert differences into baseball terms.
7. Reject differences that are not human-learnable.
8. Propose a cue or drill.

Possible cue forms:

- "start the pelvis earlier"
- "hold the torso closed longer"
- "toe tap later to delay commitment"
- "enter the zone flatter"
- "increase upward barrel path only after contact window"
- "reach less across the plate to keep barrel speed"
- "shift timing, not raw effort"

The output should not be:

- "set qpos[21] to -0.219"

The output should be:

- "the optimized swing behaves like earlier pelvis-trunk separation before contact."

## 25. Scenario Playbook

This section is for adjusting on the fly.

### 25.0 Scenario: MoCapAct Tracker Probe Works

Symptoms:

- Linux pod installs MoCapAct cleanly.
- At least one target or fallback clip loads and steps.
- A rollout JSON passes local schema/hash/intake validation.
- Proxy trajectory comparison is within declared thresholds or the mismatch is
  small enough to debug.

Action:

- make MoCapAct the physical-tracker base
- adapt the rollout into Exitvelo's evaluator
- compare physical-tracked baseline against kinematic replay
- run fixed-pitch batting evaluation on the tracker output
- then run residual search on tracker outputs, not direct `qpos`

Claim:

- the animation-to-physical-tracker bridge is working
- the project has moved from kinematic warmup to physical-control evidence

Do not:

- claim task RL improvement until residual/task optimization beats the
  physical tracking baseline.

### 25.0b Scenario: MoCapAct Probe Does Not Give A Usable Tracker

Symptoms:

- install works but `CMU_124_07` is absent or cannot step
- only non-batting control clips work
- rollout exists but tracking error is too high for batting evaluation
- no pretrained expert exists for the contact window

Action:

- preserve the probe result as evidence
- train a small custom imitation tracker on the shortest viable contact-window
  snippet
- use reference-state initialization and constrained PD target actions
- keep the existing adapter/intake/report gates

Claim:

- reuse was tested first, and custom imitation tracking is now justified

Do not:

- jump to full task RL from scratch.

### 25.1 Scenario: Current CMA/PPO Result Looks Strong

Symptoms:

- baseline vs optimized difference is clear
- sweeps show a stable improvement region
- negative controls fail as expected
- pitch jitter preserves some gain
- visual motion remains plausible

Action:

- freeze the result
- build final figures
- write the report around Stage A
- discuss full physics as future work

Claim:

- strong course-project result
- first-stage technique-discovery loop

Do not:

- spend final days chasing full physics if the report is not packaged

### 25.2 Scenario: Result Is A One-Point Spike

Symptoms:

- only one residual vector works
- nearby values fail
- CMA and PPO do not agree
- slight pitch jitter kills the effect

Action:

- downgrade the claim
- present the optimizer result as brittle
- emphasize sparse-contact difficulty
- use the failure to motivate robust residual spaces

Next experiment:

- add pitch jitter during evaluation
- smooth reward around contact quality
- search lower-dimensional residuals
- plot min bat-ball distance

Claim:

- the pipeline works, but the first optimized intervention is not robust enough to call a technique hypothesis

### 25.3 Scenario: PPO Underperforms CMA-ES

Symptoms:

- CMA improves
- PPO does not find the optimum
- PPO has noisy learning curve

Action:

- make CMA the primary quantitative optimizer
- frame PPO as an attempted RL comparison
- explain that one-step bandit PPO is not the ideal architecture

Next experiment:

- warm-start PPO around CMA solution
- use supervised residual policy from CMA samples
- reduce action bounds
- increase reward normalization
- train longer only if time allows

Claim:

- black-box optimization demonstrated the loop; PPO remains an initial RL attempt

### 25.4 Scenario: PPO Matches CMA-ES

Symptoms:

- both reach similar carry
- residual values differ
- performance similar

Action:

- make this a key result
- argue that both methods identify a performance ridge
- compare residual configurations
- show sweep/ablation

Claim:

- independent optimizers find related high-performing technique regions

### 25.5 Scenario: Full Physical Tracking Fails

Symptoms:

- humanoid falls
- bat path diverges
- controller cannot reproduce swing
- training too slow

Action:

- do not block final report
- document failure under Tier 3/Tier 4
- keep Stage A as final result

Technical adjustments:

- train shorter clip
- use reference-state initialization
- use stronger PD targets
- freeze root or lower body temporarily
- train stance/contact segment only
- use curriculum from slow motion to full speed
- use MoCapAct experts or tracking infrastructure

Claim:

- full physics is the correct next stage, but the current deliverable remains the validated residual-search loop

### 25.6 Scenario: Physical Contact Fails

Symptoms:

- ball tunnels through bat
- solver damps ball velocity
- bat constraint injects artifacts
- ball explodes or sticks

Action:

- keep analytical contact for final project
- show physical contact as an explicit limitation
- isolate contact in a tiny scene

Technical adjustments:

- simplify bat geometry
- tune contact margin/solref/solimp
- remove equality constraint
- weld bat into hand body
- increase simulation frequency
- test bat-only collision before humanoid

Claim:

- analytical contact is a pragmatic evaluator, not the final fidelity target

### 25.7 Scenario: Optimizer Exploits Simulator

Symptoms:

- impossible bat speed
- non-human posture
- huge distance from weird contact
- optimized motion looks broken
- improvement depends on launch-angle constant

Action:

- reject the result as a technique claim
- keep it as a simulator-artifact finding
- add constraints or penalties

Technical adjustments:

- cap joint velocities
- cap residual magnitudes
- add kinematic-chain check
- add bat speed plausibility thresholds
- add contact-quality constraints
- add style penalty
- add robust pitch distribution

Claim:

- the experiment identified why validation is necessary

### 25.8 Scenario: Results Are Modest But Credible

Symptoms:

- improvement is 3-5 percent
- motion plausible
- intervention interpretable
- robustness acceptable

Action:

- prefer this over a huge suspect improvement
- emphasize credibility and interpretability
- connect to human-performance optimization

Claim:

- small but defensible simulated improvement around human skill

This may be the best scientific outcome for the course.

### 25.9 Scenario: The Best Intervention Is Not Human-Learnable

Symptoms:

- residual improves metrics
- but cannot be described as a drill/cue
- or requires impossible precision

Action:

- do not call it coaching insight
- call it an optimized simulator residual
- search for coarser/named variables

Technical adjustments:

- constrain action space to coachable variables
- use body-group timing parameters
- build qualitative visual comparison
- ask whether a coach could observe it

Claim:

- the result improves performance but fails the human-translation filter

### 25.10 Scenario: We Run Out Of Time

Action priority:

1. preserve a reproducible result
2. produce main figure
3. write honest limitations
4. include validation if possible
5. leave full physics as next step

Do not:

- chase a fragile new result at the expense of final report quality

Final claim:

- first-stage demonstration of the loop, with a concrete roadmap to full physics RL

## 26. Adjustment Rules

Use these rules when deciding what to do under uncertainty.

### 26.1 If The Result Is Not Reproducible

Then:

- stop adding complexity
- rerun saved commands
- verify seeds
- check output files
- compare against baseline

Do not:

- write claims around unreproducible numbers

### 26.2 If The Result Is Not Interpretable

Then:

- add ablations
- add residual plots
- add baseline-vs-optimized overlays
- reduce action dimension
- translate into body-segment timing

Do not:

- claim technique discovery from opaque parameters

### 26.3 If The Result Is Not Robust

Then:

- add jitter
- report fixed-pitch result honestly
- say robustness is future work
- avoid broad coaching language

Do not:

- imply transfer to real batting

### 26.4 If The Physics Is Suspect

Then:

- isolate the physics subsystem
- compare to analytical baseline
- add caveat figure/table
- frame as fidelity limitation

Do not:

- bury the limitation

### 26.5 If Full RL Is Too Slow

Then:

- use CMA-ES as the main result
- use PPO as comparison or attempted RL
- train a supervised residual predictor from optimizer samples if needed
- keep full RL in roadmap

Do not:

- spend all remaining time waiting for a weak training run

### 26.6 If A Reviewer Says "This Is Not Real RL"

Answer:

- The current PPO experiment is not full closed-loop humanoid RL.
- It is a policy-gradient optimizer over a named residual action space.
- The course project demonstrates the first stage of the larger RL pipeline.
- The full closed-loop stage is explicitly defined as future work.

Then point to:

- DeepMimic/MoCapAct/RFC/ASAP/LATENT/PACE as the intended full-physics lineage.

### 26.7 If A Reviewer Says "This Is Just Baseball"

Answer:

- Baseball is the test domain.
- The general problem is contact-rich tool manipulation from human demonstration.
- The method pattern is domain-general:
  human motion prior -> simulation search -> performance metric -> interpretable residual -> validation.

### 26.8 If A Reviewer Says "This Does Not Teach Humans Yet"

Answer:

- Correct.
- The current output is a technique hypothesis, not a validated coaching prescription.
- Human validation is the next research layer.
- The contribution is building the loop that can generate such hypotheses.

## 27. Reference Paper Map

These papers define the technical neighborhood of the project.

### 27.1 DeepMimic

Paper:

- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
- https://arxiv.org/abs/1804.02717

Why it matters:

- establishes the core recipe of using reference motions plus RL to train physics-based character skills
- combines imitation objectives with task objectives
- shows that motion clips can initialize rich dynamic behavior

Relevance to Exitvelo:

- our long-term full-physics path is DeepMimic-like: imitate the human swing, then add a batting objective

### 27.2 AMP

Paper/project:

- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- https://xbpeng.github.io/projects/AMP/

Why it matters:

- uses motion datasets as style priors rather than explicit frame-by-frame tracking only
- lets task reward drive behavior while the motion prior preserves natural style

Relevance to Exitvelo:

- useful if we move beyond one swing clip and want the optimized swing to remain human-like without tracking one exact motion

### 27.2b vid2player3d

Paper/project:

- vid2player3d: Learning Controllable Tennis Skills from Broadcast Videos
  / physically simulated tennis characters from video and motion priors
- Code: https://github.com/nv-tlabs/vid2player3d
- Project page: https://research.nvidia.com/labs/toronto-ai/vid2player3d/

Why it matters:

- represents the "motion prior plus residual control" family in a sports
  striking task
- uses a MotionVAE-style high-level action plus residual control so task
  optimization can sit on top of human-like movement instead of controlling
  every joint from scratch
- its reward/termination pattern is useful for batting: pre-contact
  ball-tool proximity shaping, post-contact flight scoring, and no immediate
  termination at contact

Relevance to Exitvelo:

- supports the idea that batting residuals can eventually live on top of a
  tracker or motion latent
- useful if named residuals become too restrictive but raw full-body actions
  are too unconstrained
- public-code limitation: use architecture/reward ideas first, not direct
  checkpoint reuse

### 27.3 MoCapAct

Paper:

- MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control
- https://arxiv.org/abs/2208.07363

Why it matters:

- directly uses CMU humanoid-style mocap in `dm_control`
- highlights that mocap is kinematic and does not directly provide actions
- provides expert policies and rollouts for humanoid control research

Relevance to Exitvelo:

- this is the closest infrastructure match to our CMU humanoid stack
- it supports the claim that converting mocap to physical control is a real research problem, not a trivial implementation detail

### 27.4 RFC

Paper:

- Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis
- https://arxiv.org/abs/2006.07364

Why it matters:

- explicitly addresses dynamics mismatch between human motion and simulated humanoid bodies
- adds residual forces so policies can imitate motions that are otherwise hard or impossible

Relevance to Exitvelo:

- explains why our physical-tracking stage may fail
- suggests residual correction as a bridge between kinematic replay and physical control

### 27.5 ASAP

Paper:

- ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills
- https://arxiv.org/abs/2502.01143

Why it matters:

- uses human motion data, simulation tracking, and residual/delta action models to handle dynamics mismatch
- provides a modern staged approach to agile humanoid skills

Relevance to Exitvelo:

- supports our staged plan:
  human motion -> tracking policy -> residual correction -> more realistic control

### 27.6 LATENT Tennis

Paper:

- Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data
- https://arxiv.org/abs/2603.12686

Why it matters:

- very close conceptually to our sports-skill framing
- uses imperfect human tennis motion fragments as priors
- learns humanoid tennis behavior that can strike incoming balls while preserving natural style

Relevance to Exitvelo:

- validates the idea that sports manipulation can start from imperfect human motion priors
- supports our analogy between tennis/racket sports and baseball batting

### 27.7 PACE Table Tennis

Paper:

- PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis
- https://arxiv.org/abs/2509.21690

Why it matters:

- trains humanoid table-tennis behavior with predictive signals and dense physics-guided rewards
- addresses rapid perception, timing, footwork, and striking

Relevance to Exitvelo:

- shows that fast ball-striking humanoid tasks need prediction and dense shaping
- supports adding ball-arrival prediction and pre-contact shaping to future batting RL

## 28. How To Talk About The Project

### 28.1 Short Technical Pitch

> We use human motion capture to initialize a simulated humanoid batting swing, then optimize interpretable residuals with CMA-ES and PPO to improve bat-ball outcomes. The project studies whether robotic simulation can search for human-learnable technique improvements in a contact-rich manipulation task.

### 28.2 Short Course Pitch

> This is a deep-RL robotic manipulation project: a humanoid uses a bat as a tool to manipulate a fast-moving ball. We initialize from human demonstration and optimize a striking objective in simulation.

### 28.3 Short North-Star Pitch

> Instead of humans teaching robots, can robots start from human skill, search beyond human trial budgets, and return better technique back to humans?

### 28.4 Honest Limitation Sentence

> Our current successful evaluator is kinematic plus analytical ball physics, so the result should be read as an interpretable technique hypothesis, not as a validated full-physics coaching prescription.

### 28.5 Strong Final Sentence

> Exitvelo is a first step toward an engine for physical skill discovery: human motion provides the prior, simulation provides the search budget, RL/optimization proposes interventions, and biomechanical review translates them into human-facing hypotheses.

## 29. Final Operating Principle

When the project runs into problems, do not ask:

> How do we preserve the most ambitious claim?

Ask:

> What claim is still true under the evidence we have?

Then build the report around that claim.

The hierarchy of claims is:

1. We built the batting simulation pipeline.
2. We replayed human motion and synchronized contact.
3. We optimized named residuals around human motion.
4. We found an interpretable simulated improvement.
5. We validated the improvement against simple falsifiers.
6. We trained a physical tracking policy.
7. We optimized a physical humanoid policy.
8. We discovered a human-learnable technique.
9. We validated that technique with humans.

The course project needs to land solidly on claims 1-5. Claims 6-9 are the roadmap.
