"""Bat-ball contact detection (MuJoCo) + analytical Nathan/drag/bounce physics.

The MuJoCo `mj_step` contact path frequently misses bat-ball collisions
in this scene because the ball passes through the bat's swept volume in
fewer steps than the contact solver tolerates. The analytical Nathan
formula here is the one that reproduces the project milestone (76 mph
exit, 31 deg launch, ~295 ft).

API:
    detect_bat_ball_contact(model, data, ...)   MuJoCo contact array scan
    nathan_exit_speed(bat_speed, pitch_speed)   1D scalar exit speed
    nathan_exit_velocity(bat_vel, ball_vel)     vector exit velocity
    integrate_ball_with_drag(pos, vel, dt)      one drag step
    integrate_ball_trajectory(pos, vel, ...)    until ground or settle
    carry_distance_ft(traj, contact_xy)         feet, first ground cross
"""

from __future__ import annotations

import numpy as np
import mujoco

# ---------------------------------------------------------------- constants

G = np.array([0.0, 0.0, -9.81])
"""Gravity, m/s^2, z-up."""

# Air drag on a baseball
RHO = 1.2
"""Air density at sea level, kg/m^3."""
CD = 0.33
"""Drag coefficient for a baseball."""
R_BALL = 0.037
"""Ball radius, m (regulation MLB ~0.0365 m)."""
A_BALL = np.pi * R_BALL**2
"""Cross-sectional area, m^2."""
M_BALL = 0.145
"""Ball mass, kg."""
DRAG_K = 0.5 * RHO * CD * A_BALL / M_BALL
"""Drag acceleration coefficient: a_drag = -DRAG_K * |v| * v."""

# Nathan (2003) collision
Q_NATHAN = 0.18
"""Bat-ball coefficient of restitution at the sweet spot.
Reference: Nathan, A.M. (2003), 'Characterizing the Performance of
Baseball Bats', Am. J. Phys. 71."""

LAUNCH_ANGLE_DEG = 28.0
"""Default launch angle adjustment for the 1D->3D mapping. The Nathan
1D model gives only an exit speed magnitude along the bat's direction;
real bats produce a launch angle from sweet-spot geometry. We tilt
the exit direction up by this angle as the project's analytical
calibration produced 31 deg launch with this default."""

GROUND_Z = R_BALL
"""Ball-center z-coordinate at rest on the ground, m."""

BOUNCE_COR = 0.35
"""Coefficient of restitution on ground bounce."""
BOUNCE_MIN_VZ = 0.3
"""Below this |v_z| at impact, treat as roll instead of bounce."""
ROLL_FRICTION = 0.995
"""Per-substep multiplier on horizontal velocity while rolling."""
BOUNCE_LATERAL_DAMPING = 0.80
"""Per-bounce multiplier on horizontal velocity components."""

SETTLE_SPEED = 0.15
"""Stop integration after this speed is reached post-bounce."""


# ----------------------------------------------------------- MuJoCo contact

def detect_bat_ball_contact(model, data, *, bat_geom_ids=None, ball_geom_id=None):
    """Check MuJoCo contact array for bat-ball collision pairs.

    Returns (is_contact, contact_info_dict_or_None). Info has 'pos',
    'dist', 'frame'.
    """
    if bat_geom_ids is None:
        bat_geom_ids = set()
        for name in ['bat_barrel', 'bat_handle', 'bat_taper', 'bat_end', 'bat_knob']:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                bat_geom_ids.add(gid)
    if ball_geom_id is None:
        ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in bat_geom_ids and g2 == ball_geom_id) or \
           (g2 in bat_geom_ids and g1 == ball_geom_id):
            return True, {
                'pos': contact.pos.copy(),
                'dist': contact.dist,
                'frame': contact.frame.copy(),
            }
    return False, None


# ----------------------------------------------------------- Nathan exit

def nathan_exit_speed(bat_speed: float, pitch_speed: float, q: float = Q_NATHAN) -> float:
    """1D Nathan collision: scalar exit speed.

    v_exit = (1 + q) * v_bat + q * v_pitch

    Both inputs are positive scalars (speeds, not velocities).
    """
    return (1.0 + q) * bat_speed + q * pitch_speed


def nathan_exit_velocity(
    bat_vel: np.ndarray,
    ball_vel: np.ndarray,
    *,
    q: float = Q_NATHAN,
    launch_deg: float = LAUNCH_ANGLE_DEG,
) -> np.ndarray:
    """3D ball velocity right after contact.

    Speed is the Nathan 1D scalar. Direction is the bat's instantaneous
    direction tilted up by launch_deg in the vertical plane that contains
    the horizontal projection of the bat direction. If the bat is
    near-stationary, returns zero.
    """
    bat_vel = np.asarray(bat_vel, dtype=np.float64)
    ball_vel = np.asarray(ball_vel, dtype=np.float64)
    bat_speed = float(np.linalg.norm(bat_vel))
    if bat_speed < 1e-6:
        return np.zeros(3, dtype=np.float64)
    pitch_speed = float(np.linalg.norm(ball_vel))
    exit_speed = nathan_exit_speed(bat_speed, pitch_speed, q=q)
    bat_dir = bat_vel / bat_speed
    exit_dir = bat_dir.copy()
    horizontal = float(np.linalg.norm(exit_dir[:2]))
    exit_dir[2] = np.tan(np.radians(launch_deg)) * horizontal
    norm = float(np.linalg.norm(exit_dir))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return exit_dir / norm * exit_speed


# ----------------------------------------------------------- ball flight

def integrate_ball_with_drag(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One semi-implicit Euler step under gravity + quadratic drag.

    Updates velocity first then position so the position uses the new
    velocity (matches the milestone analytical pipeline).
    """
    speed = float(np.linalg.norm(vel))
    drag = -DRAG_K * speed * vel if speed > 0 else np.zeros(3)
    vel_new = vel + (G + drag) * dt
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


def integrate_ball_trajectory(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    dt: float = 0.001,
    max_t: float = 10.0,
    bounce: bool = True,
    ground_z: float = GROUND_Z,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Integrate ball physics until it settles or max_t.

    Returns:
        traj: (N, 3) position history
        vels: (N, 3) velocity history (parallel to traj)
        bounces: int, number of bounces detected
    """
    p, v = pos.copy().astype(np.float64), vel.copy().astype(np.float64)
    traj = [p.copy()]
    vels = [v.copy()]
    bounces = 0
    n_steps = int(max_t / dt)
    for _ in range(n_steps):
        p, v = integrate_ball_with_drag(p, v, dt)
        if p[2] <= ground_z:
            p[2] = ground_z
            if not bounce:
                traj.append(p.copy()); vels.append(v.copy())
                break
            if v[2] < -BOUNCE_MIN_VZ:
                bounces += 1
                v[2] = -v[2] * BOUNCE_COR
                v[0] *= BOUNCE_LATERAL_DAMPING
                v[1] *= BOUNCE_LATERAL_DAMPING
            else:
                v[2] = 0.0
                v[0] *= ROLL_FRICTION
                v[1] *= ROLL_FRICTION
        traj.append(p.copy()); vels.append(v.copy())
        if bounces > 0 and float(np.linalg.norm(v)) < SETTLE_SPEED:
            break
    return np.asarray(traj), np.asarray(vels), bounces


def carry_distance_ft(
    traj: np.ndarray,
    contact_xy: np.ndarray | None = None,
    *,
    ground_z: float = GROUND_Z,
) -> float:
    """Carry distance in feet: contact_xy to first ground crossing.

    If the ball never landed in the trajectory, return distance to the
    final position.
    """
    if contact_xy is None:
        contact_xy = traj[0, :2]
    below = np.where(traj[:, 2] <= ground_z + 1e-3)[0]
    if len(below) == 0:
        landing = traj[-1, :2]
    else:
        landing = traj[below[0], :2]
    dist_m = float(np.linalg.norm(landing - contact_xy))
    return dist_m * 3.28084


def total_distance_ft(traj: np.ndarray) -> float:
    """Total horizontal distance from contact to ball settle (ft).
    Use this for the 'including bounces and roll' metric, e.g. ~295 ft
    in the milestone reel.
    """
    contact_xy = traj[0, :2]
    final_xy = traj[-1, :2]
    return float(np.linalg.norm(final_xy - contact_xy)) * 3.28084
