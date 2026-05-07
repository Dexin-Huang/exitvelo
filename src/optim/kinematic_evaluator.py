"""Kinematic-mode rollout used by Tier 2A (CMA-ES) and Tier 2B (PPO).

Drives the humanoid kinematically (sets qpos directly), propagates
the ball analytically with src/env/contacts.py drag, and resolves
contact via the Nathan formula. The active evaluator across both
optimizers.

Entry points:
  scripts/run/run_tier2a_full.py  -- CMA-ES (uses cma_objective_kin)
  scripts/run/run_tier2b_ppo.py   -- PPO (uses SwingResidualBanditEnv)

Project memory note: PD-controlled humanoid drifts off the mocap during
the swing and falls over. Setting qpos directly (kinematic playback)
produces a smooth, realistic swing trajectory at the cost of being
non-physical. This is what produced the 76 mph milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.controllers.swing_residuals import SwingResiduals
from src.env.contacts import (
    GROUND_Z,
    integrate_ball_with_drag,
    integrate_ball_trajectory,
    nathan_exit_velocity,
    carry_distance_ft,
    total_distance_ft,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AMC = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")

# Calibrated defaults (scripts/calibrate_pitch_geometry.py).
# Zero residuals at this start_frame + pitch geometry produces a clean
# contact in the kinematic pipeline: 62 mph exit, 28 deg launch, 235 ft total.
DEFAULT_START_FRAME = 280
DEFAULT_PITCH_RELEASE_X = -3.31
DEFAULT_PITCH_RELEASE_Y = -0.95
DEFAULT_PITCH_HEIGHT = 1.85
# Ball must arrive at the bat sweet spot at t=0.09s. With release height
# 1.85m and target 1.35m, gravity drop alone is 0.04m so we need a small
# downward initial vz to land on the spot:
DEFAULT_PITCH_VZ = -5.14
DEFAULT_PITCH_SPEED_MS = 41.6  # 93 mph fastball

_XML_PATH = PROJECT_ROOT / "assets" / "mujoco" / "cmu_batting_scene.xml"

_BALL_QPOS_START = 64

# Swept-contact threshold matches env: ball radius + bat half-width + ~2cm
# tolerance for control-step quantization.
_SWEPT_CONTACT_RADIUS = 0.06


@dataclass
class KinResult:
    contact: bool
    exit_mph: float
    launch_deg: float
    carry_ft: float
    total_ft: float
    min_bat_ball_dist: float
    contact_frame: int
    steps: int
    sweet_speed_at_contact: float


def kinematic_rollout(
    residuals: SwingResiduals,
    *,
    amc_path: str = DEFAULT_AMC,
    start_frame: int = DEFAULT_START_FRAME,
    pitch_release_x: float = DEFAULT_PITCH_RELEASE_X,
    pitch_release_y: float = DEFAULT_PITCH_RELEASE_Y,
    pitch_release_z: float = DEFAULT_PITCH_HEIGHT,
    pitch_velocity: tuple[float, float, float] = (DEFAULT_PITCH_SPEED_MS, 0.0, DEFAULT_PITCH_VZ),
    n_steps: int = 100,
    control_dt: float = 0.01,
) -> KinResult:
    """Run one kinematic swing under a fixed pitch.

    The humanoid qpos is set directly from the mocap (with residual
    transforms applied). The ball is propagated analytically. Contact
    is detected by swept proximity between bat sweet spot and ball.
    """
    # Lazy-load the MuJoCo model. We don't need an env, just the model
    # for sites and forward kinematics.
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = mujoco.MjData(model)

    sweet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bat_sweet")

    ctrl = CMUTrackingController(
        amc_path=amc_path, start_frame=start_frame,
        residuals=residuals, control_dt=control_dt,
    )

    # Place ball at release point with the given velocity
    ball_pos = np.array([pitch_release_x, pitch_release_y, pitch_release_z], dtype=np.float64)
    ball_vel = np.array(pitch_velocity, dtype=np.float64)

    min_dist = np.inf
    contact_frame = -1
    sweet_speed_at_contact = 0.0
    exit_velocity = np.zeros(3)
    exit_mph = 0.0
    launch_deg = 0.0
    carry_ft = 0.0
    total_ft = 0.0
    contact_made = False

    prev_sweet = None

    for step in range(n_steps):
        if ctrl.done:
            break
        # Drive the humanoid kinematically: write the full qpos directly.
        action = ctrl.get_action()
        pos, quat = ctrl.get_root_state()
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
        data.qpos[7:63] = action
        data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3] = ball_pos
        mujoco.mj_forward(model, data)
        sweet_pos = data.site_xpos[sweet_id].copy()

        # Track minimum distance and check contact at the control frame.
        d = float(np.linalg.norm(ball_pos - sweet_pos))
        if d < min_dist:
            min_dist = d

        if not contact_made and d < _SWEPT_CONTACT_RADIUS:
            if prev_sweet is None:
                sweet_vel = np.zeros(3)
            else:
                sweet_vel = (sweet_pos - prev_sweet) / control_dt
            sweet_speed = float(np.linalg.norm(sweet_vel))
            if sweet_speed >= 4.0:  # bat-speed gate (~9 mph)
                contact_made = True
                contact_frame = step
                sweet_speed_at_contact = sweet_speed
                exit_velocity = nathan_exit_velocity(sweet_vel, ball_vel)
                exit_mph = float(np.linalg.norm(exit_velocity)) * 2.237
                horiz = float(np.linalg.norm(exit_velocity[:2]))
                launch_deg = (
                    float(np.degrees(np.arctan2(exit_velocity[2], horiz)))
                    if horiz > 0 else 0.0
                )
                traj, _, _ = integrate_ball_trajectory(sweet_pos.copy(), exit_velocity.copy())
                carry_ft = carry_distance_ft(traj, contact_xy=sweet_pos[:2])
                total_ft = total_distance_ft(traj)
                break

        # Propagate ball one control step. Drag is small over 0.01s so a
        # single Euler step is fine.
        ball_pos, ball_vel = integrate_ball_with_drag(ball_pos, ball_vel, control_dt)
        if ball_pos[0] > 2.0 or ball_pos[2] <= GROUND_Z:
            break
        prev_sweet = sweet_pos

    return KinResult(
        contact=contact_made,
        exit_mph=exit_mph,
        launch_deg=launch_deg,
        carry_ft=carry_ft,
        total_ft=total_ft,
        min_bat_ball_dist=float(min_dist) if min_dist != np.inf else 1.0,
        contact_frame=int(contact_frame),
        steps=int(step + 1),
        sweet_speed_at_contact=float(sweet_speed_at_contact),
    )


def evaluate_residuals_kinematic(
    residuals: SwingResiduals,
    *,
    seeds: list[int] | int = (0,),
    pitch_jitter: bool = False,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> list[KinResult]:
    """Run one kinematic rollout per seed and return per-seed result.

    With pitch_jitter=True, draws speed/height/lateral/release-time from
    a fixed distribution per seed (deterministic given the seed).
    """
    if isinstance(seeds, int):
        seeds = [seeds]

    out: list[KinResult] = []
    for seed in seeds:
        kw = dict(kwargs)
        if pitch_jitter:
            r = np.random.default_rng(int(seed))
            # Jitter AROUND the calibrated 93 mph fastball geometry so the
            # ball still lands near the bat sweet spot. Tightened from
            # the original 70-82 mph range (which never produced contact).
            speed = float(r.uniform(39.6, 43.6))    # ~88-98 mph fastball
            y_off = float(r.uniform(-0.10, 0.10))   # ~half plate width
            rt_off = float(r.uniform(-0.02, 0.02))  # +/- 20 ms release timing
            kw["pitch_velocity"] = (speed, 0.0, DEFAULT_PITCH_VZ)
            kw["pitch_release_x"] = DEFAULT_PITCH_RELEASE_X + speed * rt_off
            kw["pitch_release_y"] = DEFAULT_PITCH_RELEASE_Y + y_off
            kw["pitch_release_z"] = DEFAULT_PITCH_HEIGHT
        out.append(kinematic_rollout(residuals=residuals, **kw))
    return out


def cma_objective_kin(
    results: list[KinResult],
    *,
    w_no_contact: float = 200.0,
    w_min_dist: float = 5.0,
) -> float:
    """CMA-ES (minimization) score over a list of rollouts.

    J = -mean(carry_ft) + w_no_contact * no_contact_rate
        + w_min_dist * mean(min_bat_ball_dist on misses)

    Notes:
    - Launch term dropped: nathan_exit_velocity hard-codes 28 deg launch
      in the kinematic pipeline, so |launch - 28| is identically 0.
    - w_no_contact bumped from 25 to 200: at 25, a 5% miss rate cost
      only 1.25 ft, which is a smaller penalty than CMA-ES sample noise.
      At 200 the optimizer pays 10 ft per 5% miss, comparable to per-seed
      carry variance.
    """
    if not results:
        return 0.0
    n = len(results)
    contacts = [r for r in results if r.contact]
    misses = [r for r in results if not r.contact]
    mean_carry = float(np.mean([r.carry_ft for r in results]))
    no_contact_rate = float(len(misses) / n)
    j = -mean_carry + w_no_contact * no_contact_rate
    if misses:
        j += w_min_dist * float(np.mean([r.min_bat_ball_dist for r in misses]))
    return float(j)


def kin_aggregate(results: list[KinResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "mean_carry_ft":   float(np.mean([r.carry_ft for r in results])),
        "mean_exit_mph":   float(np.mean([r.exit_mph for r in results])),
        "mean_launch_deg": float(np.mean([r.launch_deg for r in results])),
        "contact_rate":    float(np.mean([r.contact for r in results])),
        "no_contact_rate": float(1.0 - np.mean([r.contact for r in results])),
        "mean_min_dist":   float(np.mean([r.min_bat_ball_dist for r in results])),
        "mean_sweet_speed_at_contact": float(np.mean(
            [r.sweet_speed_at_contact for r in results if r.contact]
        )) if any(r.contact for r in results) else 0.0,
    }


def kin_asdict_list(results: list[KinResult]) -> list[dict]:
    return [asdict(r) for r in results]
