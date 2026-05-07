"""Find a pitch geometry that lands the ball on the bat sweet spot
during the swing. Used to set the kinematic evaluator's default pitch.

Algorithm:
  1. Run the kinematic swing (no ball) for N frames; record sweet pos.
  2. Find the contact frame f* = argmax forward velocity along +x.
  3. Place the ball at bat_target = sweet[f*] at simulation time
     t_target = f* * control_dt.
  4. Choose a pitch_speed; back-solve the release point:
        pitch_release_pos = bat_target - velocity * t_flight
     with gravity correction so vertical motion lines up.
  5. Verify by kinematic_rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.controllers.swing_residuals import SwingResiduals
from src.optim.kinematic_evaluator import (
    DEFAULT_AMC,
    kinematic_rollout,
)

_XML_PATH = PROJECT_ROOT / "assets" / "mujoco" / "cmu_batting_scene.xml"


def trace_bat_trajectory(start_frame: int, n_steps: int = 80, control_dt: float = 0.01):
    """Run the kinematic swing and return per-step sweet-spot pos."""
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    data = mujoco.MjData(model)
    sweet_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bat_sweet")
    ctrl = CMUTrackingController(
        amc_path=DEFAULT_AMC, start_frame=start_frame,
        residuals=SwingResiduals(), control_dt=control_dt,
    )
    sweets = []
    for _ in range(n_steps):
        if ctrl.done:
            break
        action = ctrl.get_action()
        pos, quat = ctrl.get_root_state()
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
        data.qpos[7:63] = action
        mujoco.mj_forward(model, data)
        sweets.append(data.site_xpos[sweet_id].copy())
    return np.array(sweets)


def find_pitch_geometry(
    start_frame: int,
    *,
    pitch_speed_ms: float = 41.6,
    control_dt: float = 0.01,
):
    """Return (target_xyz, contact_frame, release_xyz, ball_vel) such that
    a ball launched at release_xyz with ball_vel arrives at target_xyz
    at simulation time contact_frame*control_dt."""
    sweets = trace_bat_trajectory(start_frame, control_dt=control_dt)
    if len(sweets) < 3:
        raise RuntimeError("not enough frames in mocap to trace bat")

    # Forward velocity per frame (toward +x = into the field)
    vx = np.gradient(sweets[:, 0]) / control_dt
    f_star = int(np.argmax(vx))

    target = sweets[f_star]
    t_target = f_star * control_dt
    g = 9.81

    # Choose flight time and pitch direction along +x so the ball reaches target.
    # We pick a release point at z = release_height and back-solve initial
    # velocity so the ball passes through `target` at time t_flight.
    # Use the same t_target for flight time so pitch is released at sim t=0.
    release_height = 1.85  # standard MLB release point
    if t_target < 0.05:
        raise RuntimeError(
            f"contact frame {f_star} is too early (t={t_target:.3f}s); "
            f"try a smaller start_frame."
        )

    # Pitch starts at sim t=0. We want pos(t_target) = target.
    # pos(t) = release + v0*t + 0.5*g*t^2 (with g_z = -9.81)
    # so:
    #   v0_x = (target.x - release.x) / t_target  -> with release fixed by speed
    #   v0_y = (target.y - release.y) / t_target
    #   v0_z = (target.z - release.z) / t_target + 0.5*g*t_target

    # Given pitch_speed_ms is the magnitude of the horizontal velocity,
    # set release.x so v0_x = pitch_speed_ms exactly.
    release_x = target[0] - pitch_speed_ms * t_target
    release_y = target[1]            # straight in laterally
    release_z = release_height
    v0_x = pitch_speed_ms
    v0_y = (target[1] - release_y) / t_target  # = 0
    v0_z = (target[2] - release_z) / t_target + 0.5 * g * t_target
    return {
        "start_frame": start_frame,
        "contact_frame": f_star,
        "t_contact_s": t_target,
        "target_xyz": target.tolist(),
        "release_xyz": [release_x, release_y, release_z],
        "ball_velocity": [v0_x, v0_y, v0_z],
        "pitch_speed_ms": pitch_speed_ms,
    }


def main():
    print("=" * 70)
    print("Pitch geometry calibration")
    print("=" * 70)
    candidates = [280, 300, 320, 340, 360, 380, 400, 420, 440]
    rows = []
    for sf in candidates:
        try:
            g = find_pitch_geometry(sf)
        except RuntimeError as e:
            print(f"  sf={sf}: {e}")
            continue
        result = kinematic_rollout(
            residuals=SwingResiduals(),
            start_frame=sf,
            pitch_release_x=g["release_xyz"][0],
            pitch_release_y=g["release_xyz"][1],
            pitch_release_z=g["release_xyz"][2],
            pitch_velocity=tuple(g["ball_velocity"]),
        )
        tag = "HIT" if result.contact else "MISS"
        print(
            f"  sf={sf:3d}  cf={g['contact_frame']:3d}  "
            f"t_contact={g['t_contact_s']:.3f}s  "
            f"target=({g['target_xyz'][0]:+.2f},{g['target_xyz'][1]:+.2f},{g['target_xyz'][2]:+.2f})  "
            f"release_x={g['release_xyz'][0]:+.2f}  "
            f"v0=({g['ball_velocity'][0]:+5.1f},{g['ball_velocity'][1]:+5.1f},{g['ball_velocity'][2]:+5.1f})  "
            f"{tag} min_d={result.min_bat_ball_dist:.3f} "
            f"exit={result.exit_mph:5.1f}mph carry={result.carry_ft:5.1f}ft "
            f"sweet_spd={result.sweet_speed_at_contact:.1f}m/s"
        )
        rows.append((sf, g, result))

    hits = [r for r in rows if r[2].contact]
    print()
    print(f"Hits: {len(hits)}/{len(rows)}")
    if hits:
        best = max(hits, key=lambda r: r[2].carry_ft)
        sf, g, res = best
        print()
        print("BEST CALIBRATION:")
        print(f"  start_frame = {sf}")
        print(f"  pitch_release_xyz = {g['release_xyz']}")
        print(f"  pitch_velocity    = {g['ball_velocity']}")
        print(f"  exit={res.exit_mph:.1f} mph, launch={res.launch_deg:.1f}°, "
              f"carry={res.carry_ft:.1f} ft, total={res.total_ft:.1f} ft, "
              f"sweet_speed={res.sweet_speed_at_contact:.2f} m/s")


if __name__ == "__main__":
    main()
