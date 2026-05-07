"""Tier 1: One-At-a-Time (OAT) sensitivity screening of swing perturbations.

Picks the action space for Tier 2 RL by ranking which body-group / timing
perturbations move exit velocity the most under fixed pitch and contact.

Pipeline:
  1. Load the saved pitch_v1 config (fixed 93 mph fastball + best start frame).
  2. For each named perturbation in {time_scale, start_frame_offset,
     legs_amp, spine_amp, larm_amp, rarm_amp}, sweep across a small set of
     deltas, run one PD-controlled trial, log (contact, exit_vel, launch).
  3. Output JSON + a sensitivity ranking (max |d_exit| per param).

This is pure motivation for Tier 2. Not a primary result.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controllers.cmu_tracking_controller import CMUTrackingController
from src.env.cmu_batting_env import CMUBattingEnv
from src.motion.cmu_replay import CMUMocapReplay

# ── Config ─────────────────────────────────────────────────────────────────
AMC_PATH = str(PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc")
PITCH_CONFIG = PROJECT_ROOT / "results" / "final" / "cmu_milestone_results.json"
OUT_PATH = PROJECT_ROOT / "results" / "tier1_oat_sensitivity.json"

# qpos slices per body group (after the 7-dim root)
GROUPS = {
    "legs":  slice(7, 21),    # lfemur, ltibia, lfoot, ltoes + r mirror
    "spine": slice(21, 30),   # lowerback, upperback, thorax
    "larm":  slice(39, 45),   # lclavicle, lhumerus, lradius
    "rarm":  slice(51, 57),   # rclavicle, rhumerus, rradius
}

# OAT deltas per param
SWEEP = {
    "time_scale":         [0.90, 0.95, 1.05, 1.10],   # multiplicative on dt
    "start_frame_offset": [-8, -4, +4, +8],            # frames
    "legs_amp":           [0.90, 0.95, 1.05, 1.10],   # scale (q-mean) around mean
    "spine_amp":          [0.90, 0.95, 1.05, 1.10],
    "larm_amp":           [0.90, 0.95, 1.05, 1.10],
    "rarm_amp":           [0.90, 0.95, 1.05, 1.10],
}

# ── Perturbation functions ─────────────────────────────────────────────────

def perturb_amp_inplace(qpos_traj: np.ndarray, body_slice: slice, alpha: float) -> None:
    """Scale joint angles in `body_slice` around their per-joint temporal mean.
    Preserves continuity at body group boundaries (mean is unchanged) but
    expands or contracts the swing's ROM in that group.
    """
    q = qpos_traj[body_slice, :]
    mean = q.mean(axis=1, keepdims=True)
    qpos_traj[body_slice, :] = mean + alpha * (q - mean)


def perturb_time_scale_inplace(replay: CMUMocapReplay, alpha: float) -> None:
    """Resample the qpos trajectory by linear interpolation along time so the
    swing plays out alpha times slower (alpha > 1) or faster (alpha < 1).
    The control_dt stays the same; we just stretch the trajectory's duration.
    """
    T = replay.qpos_trajectory.shape[1]
    new_T = max(2, int(round(T * alpha)))
    src_t = np.linspace(0.0, 1.0, T)
    dst_t = np.linspace(0.0, 1.0, new_T)
    new_qpos = np.empty((replay.qpos_trajectory.shape[0], new_T))
    for d in range(replay.qpos_trajectory.shape[0]):
        new_qpos[d] = np.interp(dst_t, src_t, replay.qpos_trajectory[d])
    replay.qpos_trajectory = new_qpos
    replay.n_frames = new_T


# ── Trial runner ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    contact: bool
    exit_vel_ms: float
    launch_deg: float
    peak_bat_speed: float


def run_trial(
    pitch_cfg: dict,
    perturbation: Callable[[CMUMocapReplay], None] | None = None,
    sf_offset: int = 0,
    n_steps: int = 200,
) -> TrialResult:
    """Run one PD-controlled swing with the given pitch + perturbation.
    Returns contact + outcome metrics. Mostly mirrors test_pitch() in
    run_cmu_reference_hit.py, but lets us mutate the mocap trajectory first.
    """
    sf = pitch_cfg["replay_start_frame"] + sf_offset
    ball_start = np.array(pitch_cfg["pitch"]["start_pos"])
    ball_vel = np.array(pitch_cfg["pitch"]["velocity"])

    ctrl = CMUTrackingController(amc_path=AMC_PATH, start_frame=sf)
    if perturbation is not None:
        perturbation(ctrl.replay)
        # if the perturbation extended the trajectory, sf may need clamping
        if sf >= ctrl.replay.n_frames:
            return TrialResult(False, 0.0, 0.0, 0.0)
        ctrl.replay.step_idx = sf

    env = CMUBattingEnv(
        render_mode=None,
        pitch_x=float(ball_start[0]),
        pitch_y=float(ball_start[1]),
        pitch_height=float(ball_start[2]),
        pitch_velocity=ball_vel.tolist(),
    )
    obs, _ = env.reset()

    pos, quat = ctrl.get_root_state()
    env.data.qpos[0:3] = pos
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    action = ctrl.get_action(obs)
    env.data.qpos[7:63] = action
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    ctrl.reset()

    peak_bat_speed = 0.0
    for _ in range(n_steps):
        if ctrl.done:
            break
        action = ctrl.get_action(obs)
        rp, rq = ctrl.get_root_state()
        env.data.qpos[0:3] = rp
        env.data.qpos[3:7] = rq
        env.data.qvel[0:6] = 0.0
        obs, _, term, trunc, info = env.step(action)
        peak_bat_speed = max(peak_bat_speed, float(np.linalg.norm(obs[128:131])))
        if info.get("contact", False):
            sp = info.get("ball_speed_post", 0.0)
            bvel = env.data.qvel[63:66].copy()
            hz = float(np.linalg.norm(bvel[:2]))
            la = float(np.degrees(np.arctan2(bvel[2], hz))) if hz > 0 else 0.0
            env.close()
            return TrialResult(True, float(sp), la, peak_bat_speed)
        if term or trunc:
            break
    env.close()
    return TrialResult(False, 0.0, 0.0, peak_bat_speed)


# ── Main: OAT sweep ────────────────────────────────────────────────────────

def main():
    pitch_cfg = json.loads(PITCH_CONFIG.read_text())["pitch_config"] if "pitch_config" in json.loads(PITCH_CONFIG.read_text()) else json.loads(PITCH_CONFIG.read_text())

    print("=" * 70)
    print("Tier 1: OAT Sensitivity Screening")
    print("=" * 70)
    print(f"  Pitch config: {PITCH_CONFIG}")
    print(f"  Replay start frame (baseline): {pitch_cfg['replay_start_frame']}")

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n[baseline]")
    base = run_trial(pitch_cfg)
    print(f"  contact={base.contact}  exit={base.exit_vel_ms:.2f} m/s  "
          f"launch={base.launch_deg:.1f}deg  peak_bat={base.peak_bat_speed:.2f} m/s")
    if not base.contact:
        print("  WARNING: baseline did not make contact — sweep results will be noisy.")

    log = {"baseline": vars(base), "perturbations": {}}

    # ── Perturbation sweeps ──────────────────────────────────────────
    for param, deltas in SWEEP.items():
        print(f"\n[{param}]")
        log["perturbations"][param] = []
        for delta in deltas:
            if param == "time_scale":
                perturb = lambda r, a=delta: perturb_time_scale_inplace(r, a)
                sf_off = 0
            elif param == "start_frame_offset":
                perturb = None
                sf_off = int(delta)
            else:
                grp_name = param.replace("_amp", "")
                slc = GROUPS[grp_name]
                perturb = lambda r, s=slc, a=delta: perturb_amp_inplace(r.qpos_trajectory, s, a)
                sf_off = 0

            res = run_trial(pitch_cfg, perturbation=perturb, sf_offset=sf_off)
            d_exit = res.exit_vel_ms - base.exit_vel_ms if res.contact and base.contact else 0.0
            print(f"  delta={delta:+.2f}  contact={res.contact}  "
                  f"exit={res.exit_vel_ms:6.2f} m/s  d_exit={d_exit:+.2f}  "
                  f"launch={res.launch_deg:5.1f}deg  peak_bat={res.peak_bat_speed:.2f}")
            log["perturbations"][param].append({
                "delta": delta,
                "contact": res.contact,
                "exit_vel_ms": res.exit_vel_ms,
                "d_exit_vs_baseline": d_exit,
                "launch_deg": res.launch_deg,
                "peak_bat_speed": res.peak_bat_speed,
            })

    # ── Sensitivity ranking ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Sensitivity ranking (max |d_exit| over the sweep, contact-only)")
    print("=" * 70)
    ranking = []
    for param, results in log["perturbations"].items():
        contact_results = [r for r in results if r["contact"]]
        if not contact_results:
            ranking.append((param, 0.0, 0))
            continue
        max_abs_d = max(abs(r["d_exit_vs_baseline"]) for r in contact_results)
        ranking.append((param, max_abs_d, len(contact_results)))
    ranking.sort(key=lambda x: -x[1])
    for param, max_d, nc in ranking:
        print(f"  {param:25s}  max|d_exit|={max_d:5.2f} m/s   contacts={nc}/{len(SWEEP[param])}")

    log["ranking"] = [{"param": p, "max_abs_d_exit_ms": d, "n_contacts": n} for p, d, n in ranking]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(log, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
