# Exitvelo Goal Document

## 0. One-Sentence North Star

Can reinforcement learning and simulation optimize human physical performance by starting from human motion, searching beyond human intuition through massive simulated practice, and returning discoveries as techniques that real humans can understand, test, and learn?

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

- Long-term.

Approach:

- imitation pretraining
- residual force control
- PPO or similar fine-tuning
- style preservation
- energy and torque constraints
- pitch randomization curriculum

Why it matters:

- This is the real version of "the robot starts from human level and improves technique."

Risk:

- The learning problem is much harder than the current course-project timeline allows.

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

### Checkpoint 1: Lock The Story

Goal:

- Decide the exact final-project claim.

Candidate claim:

- "We demonstrate a first human-performance optimization loop for baseball batting: starting from CMU human mocap, we optimize interpretable swing residuals with CMA-ES and PPO, improving simulated carry while producing a human-readable timing hypothesis."

Pass condition:

- Everyone agrees this is the claim used in slides and report.

### Checkpoint 2: Reproduce The Headline Figure

Goal:

- Confirm `results/tier2_comparison.png` regenerates from saved artifacts.

Pass condition:

- One command regenerates the plot.
- Metrics match README numbers.

### Checkpoint 3: Validate The Main Intervention

Goal:

- Determine whether negative `hip_fire_rad` is a stable effect.

Experiments:

- sweep `hip_fire_rad`
- zero out CMA/PPO residual dimensions
- compare fixed pitch and jittered pitch

Pass condition:

- There is a stable region of improvement, not only a single lucky optimum.

### Checkpoint 4: Generate Human Interpretation

Goal:

- Translate residuals into baseball language.

Output:

- one paragraph
- one diagram or annotated plot
- one caveat sentence

Pass condition:

- A reader can understand what the simulated intervention means without reading code.

### Checkpoint 5: Add Negative Control

Goal:

- Show the system does not produce success under obviously bad timing/location.

Pass condition:

- wrong timing or outside pitch has low carry/contact.

### Checkpoint 6: Final Report Package

Goal:

- Freeze all evidence needed for final submission.

Pass condition:

- figures, tables, commands, metrics, and caveats are complete.

## 15. Open Questions

Scientific:

- What makes a simulator-discovered movement insight credible?
- How much physical fidelity is required before a technique hypothesis is useful?
- Can a low-dimensional residual space capture meaningful human technique?
- When does optimization become coaching insight rather than curve fitting?

Technical:

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

### Long-Term Success

- Full dynamic humanoid RL starts from human motion and discovers robust, physically plausible improvements.
- Discoveries transfer into human training or high-fidelity biomechanical validation.

## 18. The Final Thesis

The final thesis should be:

> Human physical performance can be treated as an optimization problem initialized by human expertise. In this project, we demonstrate a first version of that idea for baseball batting: a CMU human swing is replayed in simulation, perturbed through named technique residuals, optimized by CMA-ES and PPO, and translated back into a candidate baseball technique hypothesis. The current system is not yet a full physics coaching engine, but it establishes the loop needed to build one.

## 19. Decision Rules

When deciding what to do next, use these rules:

1. Prefer experiments that strengthen the central thesis.
2. Prefer validation over another optimizer run once a result exists.
3. Prefer interpretable residuals over high-dimensional opaque control.
4. Prefer honest caveats over fragile claims.
5. Prefer one reproducible figure over five unorganized logs.
6. Prefer human-language synthesis over raw parameter dumps.
7. Prefer falsification tests before finalizing claims.

## 20. One-Page Version

We are trying to answer whether simulation and RL can optimize human physical technique rather than only teaching robots to imitate humans.

Baseball batting is the test case. We start from a real CMU human swing, replay it in MuJoCo, attach a bat, synchronize a pitch, and measure ball-flight outcomes. Instead of asking an RL agent to discover batting from scratch, we ask it to search a small space of named, human-interpretable residuals around a real swing.

The current system has already proven the pipeline and produced a Tier 2 result. CMA-ES and PPO both improve simulated carry over the zero-residual swing in a kinematic-plus-analytical evaluator. The main intervention appears to involve pelvis-trunk timing, especially negative `hip_fire_rad`.

The key limitation is fidelity. The active result is not yet full physics RL. It is a fast, interpretable search system using kinematic humanoid replay and analytical ball flight. Therefore the right claim is not "we discovered a validated coaching cue." The right claim is "we built the first version of a loop that can turn human motion into simulated search and simulated search into human-facing technique hypotheses."

The next work should focus on validation: sweeps, ablations, negative controls, pitch jitter, kinematic-chain sanity, and clear translation of the optimized residual into baseball language. The long-term goal is a general engine for physical skill discovery: start from human level, simulate many more trials than humans can perform, find better technique, and return it to humans in a learnable form.

