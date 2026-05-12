# Stage 2 Physical Tee-Ball Distance Search Result

Date: 2026-05-11

## Objective

Move from a visually approved massful bat/contact rollout to the first physical
distance optimization loop:

1. Keep the MoCapAct same-body imitation tracker and approved residual prior.
2. Weld the original Exitvelo bat asset to the CMU left hand with massful
   collision primitives.
3. Place a held/released free baseball at the tee sweet spot.
4. Use CEM to optimize residual actions for physical post-contact carry while
   keeping the body close to the mocap imitation.

This is not yet a full PPO/RL distance training run. It is the first working
physical objective search over the learned tracker.

## Implementation

New script:

`scripts/runpod/search_mocapact_physical_tball_cem.py`

RunPod run:

`/workspace/exitvelo/results/mocapact_residual_ppo/physical_tball_distance_stride1_pop12_iter4`

Local artifacts:

- `results/downloads/CMU_124_07_physical_tball_distance_pop12_iter4.mp4`
- `results/downloads/CMU_124_07_physical_tball_distance_pop12_iter4_summary.json`
- `results/downloads/CMU_124_07_physical_tball_distance_pop12_iter4_best_residual.npy`

Key scene settings:

- Mocap source: `/workspace/exitvelo/results/mocapact_custom/cmu_124_07_stride1.h5`
- Initial residual: `virtual_bat_tball_speedkick3860_twohand_pop16_iter2/best_residual.npy`
- Bat mesh: `/workspace/exitvelo/assets/meshes/baseball_bat.stl`
- Bat attachment: `parent_hand=lhand`
- Bat local position: `[0.20, -0.047, 0.0]`
- Bat local Euler degrees: `[210.0, -90.0, -10.0]`
- Baseball mode: free body, held at tee until step 36
- Tee position: `[-0.2822901626, -0.1131119678, 1.2160991329]`
- CEM: population 12, elites 4, iterations 4, knots 12

## Result

Baseline physical tee-ball contact:

- Frames: 95
- First bat-ball contact: step 35
- Contact frames: 5
- Launch sample step: 45
- Launch velocity: `[-1.3023, -3.4570, -2.7477] m/s`
- Exit speed: `10.30 mph`
- No-drag carry estimate: `2.98 ft`
- Launch angle: `-36.64 deg`

Best CEM candidate:

- Frames: 95
- First bat-ball contact: step 35
- Contact frames: 6
- Launch sample step: 45
- Launch velocity: `[-1.3666, -4.7841, -2.2139] m/s`
- Exit speed: `12.18 mph`
- No-drag carry estimate: `4.62 ft`
- Launch angle: `-23.99 deg`

Delta:

- Exit speed: `+1.88 mph`
- Carry: `+1.64 ft` (`+55%`)
- Launch angle: `+12.65 deg` less downward
- Contact timing: unchanged at step 35
- Pose fidelity: essentially unchanged
  - body abs mean: `0.16721 -> 0.16713`
  - end-effector L2 mean: `0.44068 -> 0.43960`

## Interpretation

This validates the next-stage loop: we can use the physically tracked mocap
swing as a prior, attach the real massful bat, place a tee ball, and optimize a
residual for a task objective measured from MuJoCo contact.

The current improvement is modest because the optimizer is still constrained by
a small residual around a tracker that was not trained with the bat/ball task.
The best candidate mostly improves distance by making the launch less steeply
downward and increasing horizontal ball speed.

## Next Step

Use this result as the seed for a stronger distance loop:

1. Run a wider CEM continuation from the best residual.
2. Add an explicit upward/launch-angle curriculum so the ball leaves the bat
   instead of being driven downward.
3. If CEM plateaus, convert the same physical objective into residual PPO with
   the best CEM residual as warm start.
4. Only after distance improves in tee-ball mode, consider adding two-hand grip
   constraints or a trainable bat-hand interface.

## Continuation Result

After the first pass, a continuation search was run from the best residual with
an explicit launch-shape objective:

- Seed residual:
  `physical_tball_distance_stride1_pop12_iter4/best_residual.npy`
- RunPod run:
  `/workspace/exitvelo/results/mocapact_residual_ppo/physical_tball_distance_stride1_cont_pop14_iter4_launchfloor`
- CEM: population 14, elites 4, iterations 4
- Extra scoring:
  - penalize downward vertical launch speed
  - prefer launch angle above `-10 deg`
  - keep carry and horizontal speed as primary distance terms

Local continuation artifacts:

- `results/downloads/CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor.mp4`
- `results/downloads/CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor_summary.json`
- `results/downloads/CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor_best_residual.npy`

Continuation baseline, seeded from the first pass:

- Frames: 95
- First bat-ball contact: step 35
- Launch sample step: 45
- Exit speed: `12.18 mph`
- No-drag carry estimate: `4.62 ft`
- Launch angle: `-23.99 deg`

Continuation best:

- Frames: 95
- First bat-ball contact: step 35
- Contact frames: 10
- Launch sample step: 39
- Launch velocity: `[-0.2412, -4.5092, 0.0699] m/s`
- Exit speed: `10.10 mph`
- No-drag carry estimate: `7.35 ft`
- Launch angle: `+0.89 deg`
- Pose fidelity remains close:
  - body abs mean: `0.16713 -> 0.16765`
  - end-effector L2 mean: `0.43960 -> 0.43948`

Total improvement versus the original physical tee-ball baseline:

- Carry: `2.98 ft -> 7.35 ft` (`+147%`)
- Launch angle: `-36.64 deg -> +0.89 deg`
- Contact timing: unchanged at step 35
- Contact frames: `5 -> 10`
- Exit speed: `10.30 mph -> 10.10 mph`

Interpretation:

The continuation found a more distance-correct impact: not faster, but much
flatter and slightly upward. This is the first clear proof that the physical
tee-ball optimizer can change the learned motion in the direction required for
carry distance while preserving the mocap-tracked swing. The obvious next
training target is to recover/increase exit speed while keeping the launch angle
near positive.

## 100 ft Target Gap

The historical kinematic pipeline is much stronger than the physical tracker:

- Kinematic fixed-pitch baseline:
  - carry: `177.66 ft`
  - exit speed: `62.35 mph`
  - bat sweet speed at contact: `17.35 m/s`
  - pitch model: calibrated `93 mph` inbound pitch with analytical Nathan exit
    velocity and fixed `28 deg` launch
- Current physical tee-ball best:
  - carry: `7.35 ft`
  - exit speed: `10.10 mph`
  - bat sweet speed at contact: about `5.2 m/s`
  - ball model: free tee ball with MuJoCo contact

Approximate no-drag speed needed for `100 ft` tee-ball carry from the current
ball height:

- `~40 mph` exit at `30 deg`
- `~46 mph` exit at `20 deg`
- `~60 mph` exit at `10 deg`

With the old `93 mph` pitched-ball Nathan model, a `10 m/s` bat speed would
project to roughly `110 ft` at `28 deg`; the old `17.35 m/s` kinematic bat
speed projects well beyond that. Therefore the current blocker is not simply
the distance objective. The physical tracker is carrying only about one third
of the kinematic sweet-spot speed into contact.

## Speed-Recovery Continuation

RunPod run:

`/workspace/exitvelo/results/mocapact_residual_ppo/physical_tball_distance_stride1_speedrecover_pop18_iter5`

Local artifacts:

- `results/downloads/CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5.mp4`
- `results/downloads/CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5_summary.json`
- `results/downloads/CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5_best_residual.npy`

This run added a sweet-spot speed term with a soft `10 m/s` target, loosened
the prior, and used larger CEM noise.

Best speed-recovery result:

- Frames: 95
- First bat-ball contact: step 35
- Exit speed: `11.41 mph`
- Carry: `7.01 ft`
- Launch angle: `-7.17 deg`
- Bat sweet speed: `5.19 m/s`
- Bat speed shortfall versus `10 m/s` target: `4.81 m/s`

Trace conclusion:

- Iteration best bat speed stayed between `5.18` and `5.20 m/s`.
- CEM could trade launch angle and ball speed inside the same envelope.
- It did not find a path toward `10 m/s`, let alone the old `17.35 m/s`
  kinematic speed.

Decision:

More residual CEM over this tracker is unlikely to reach `100 ft`. The next
straight path is to fix the physical imitation layer itself:

1. Build a bat-speed audit that compares target kinematic bat sites against the
   MoCapAct physical bat sites frame by frame.
2. Train or fine-tune the imitation tracker on the contact window with explicit
   bat-site velocity rewards, not just body pose tracking.
3. Keep the ball out during this retraining until the physical bat sweet speed
   reaches at least `10 m/s`.
4. Reintroduce tee-ball distance once the physical tracker has enough barrel
   speed.
5. Reintroduce the pitched-ball setup only after tee-ball impact is speed- and
   launch-correct.
