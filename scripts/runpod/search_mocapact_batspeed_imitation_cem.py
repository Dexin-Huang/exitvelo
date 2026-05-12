"""No-ball bat-speed imitation search for the MoCapAct physical tracker.

The physical tee-ball CEM can tune launch/contact, but it plateaued around
5.2 m/s bat-sweet speed.  The old kinematic evaluator reaches ~17.35 m/s at
the same conceptual contact moment.  This script removes the ball and asks a
narrower question:

Can an open-loop residual make the massful welded bat move faster in the
contact window while the MoCapAct body remains close to the tracked swing?

It builds a raw kinematic target from the original AMC + cmu_batting_scene.xml,
aligns the kinematic contact frame to the physical contact step, and scores the
physical bat-site speed profile against that target.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_mocapact_old_bat_asset import (
    OLD_BAT_POS,
    OLD_BAT_SITES,
    _install_old_bat_walker,
    _make_env,
    _site_xpos,
)
from search_mocapact_residual_bias_cem import _residual_at_step
from search_mocapact_virtual_tball_cem import _load_initial_residual, _pose_metrics
from train_mocapact_multiclip_residual_ppo import (
    DEFAULT_OUT_DIR,
    DEFAULT_POLICY,
    _json_safe,
    _metric_snapshot,
    _write_json,
)


MPH_PER_MPS = 2.2369362920544
DEFAULT_PHYSICAL_MOCAP = Path(
    "/workspace/exitvelo/results/mocapact_custom/cmu_124_07_stride1.h5"
)
DEFAULT_INITIAL_RESIDUAL = Path(
    "/workspace/exitvelo/results/mocapact_residual_ppo/"
    "physical_tball_distance_stride1_cont_pop14_iter4_launchfloor/best_residual.npy"
)
DEFAULT_BAT_EULER_DEG = (210.0, -90.0, -10.0)
DEFAULT_AMC = Path("/workspace/exitvelo/data/raw/cmu_subject_124/124_07.amc")
DEFAULT_KINEMATIC_XML = Path("/workspace/exitvelo/assets/mujoco/cmu_batting_scene.xml")


def _parse_step_ranges(value: str) -> set[int]:
    steps: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending step range: {part!r}")
            steps.update(range(start, end + 1))
        else:
            steps.add(int(part))
    return steps


def _velocity(points: np.ndarray, *, dt: float = 0.01) -> np.ndarray:
    out = np.zeros_like(points, dtype=np.float64)
    if len(points) < 2:
        return out
    out[1:] = (points[1:] - points[:-1]) / float(dt)
    out[0] = out[1]
    return out


def _speed(points: np.ndarray, *, dt: float = 0.01) -> np.ndarray:
    return np.linalg.norm(_velocity(points, dt=dt), axis=1)


def _kinematic_target(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    from src.controllers.cmu_tracking_controller import CMUTrackingController
    from src.controllers.swing_residuals import SwingResiduals

    model = mujoco.MjModel.from_xml_path(str(args.kinematic_xml))
    data = mujoco.MjData(model)
    site_ids = {
        site: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        for site in OLD_BAT_SITES
    }
    missing = [site for site, site_id in site_ids.items() if site_id < 0]
    if missing:
        raise RuntimeError(f"kinematic XML missing bat sites: {missing}")

    ctrl = CMUTrackingController(
        amc_path=str(args.amc_path),
        start_frame=int(args.kinematic_start_frame),
        residuals=SwingResiduals(),
        control_dt=0.01,
    )
    sites: dict[str, list[list[float]]] = {site: [] for site in OLD_BAT_SITES}
    for _ in range(int(args.kinematic_steps)):
        if ctrl.done:
            break
        action = ctrl.get_action()
        pos, quat = ctrl.get_root_state()
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
        data.qpos[7:63] = action
        mujoco.mj_forward(model, data)
        for site, site_id in site_ids.items():
            sites[site].append(np.asarray(data.site_xpos[site_id]).tolist())

    arrays = {site: np.asarray(values, dtype=np.float64) for site, values in sites.items()}
    speeds = {site: _speed(values).tolist() for site, values in arrays.items()}
    summary = {
        "amc_path": str(args.amc_path),
        "kinematic_xml": str(args.kinematic_xml),
        "kinematic_start_frame": int(args.kinematic_start_frame),
        "kinematic_steps": len(next(iter(sites.values()))),
        "kinematic_contact_step": int(args.kinematic_contact_step),
        "physical_contact_step": int(args.physical_contact_step),
        "sites": {site: values.tolist() for site, values in arrays.items()},
        "speeds_mps": speeds,
    }
    for site, values in arrays.items():
        speed = np.asarray(speeds[site], dtype=np.float64)
        summary[f"{site}_peak_speed_mps"] = float(np.max(speed)) if len(speed) else 0.0
        summary[f"{site}_peak_step"] = int(np.argmax(speed)) if len(speed) else None
        idx = min(max(0, int(args.kinematic_contact_step)), len(speed) - 1)
        summary[f"{site}_contact_speed_mps"] = float(speed[idx]) if len(speed) else 0.0
    return summary


def _install_scene(args: argparse.Namespace) -> None:
    _install_old_bat_walker(
        args.bat_mesh,
        attach_mode="fixed_body",
        parent_hand=args.parent_hand,
        bat_pos=tuple(float(value) for value in args.bat_pos),
        bat_euler_rad=tuple(math.radians(float(value)) for value in args.bat_euler_deg),
        physical_contact=True,
        two_hand_connect=False,
        two_hand_anchor=tuple(float(value) for value in args.two_hand_anchor),
        tiny_mass=float(args.tiny_mass),
        include_collision_primitives=True,
        ball_mode="none",
        ball_pos=(0.0, 0.0, 1.0),
        ball_radius=0.0366,
        ball_mass=0.145,
        ball_gravcomp=1.0,
    )


def _rollout(
    env: Any,
    residual: np.ndarray,
    *,
    max_steps: int,
    knots: int,
    render: bool,
    width: int,
    height: int,
    camera_id: int,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    env.reset()
    reward = 0.0
    done = False
    info: dict[str, Any] = {}
    for step_idx in range(max_steps):
        if render:
            frames.append(
                env.render(
                    mode="rgb_array",
                    height=int(height),
                    width=int(width),
                    camera_id=int(camera_id),
                )
            )
        physics = env.dm_env.physics
        rec = {
            "step": step_idx,
            "reward": reward,
            "done": done,
            "info": _json_safe(info),
            "bat_sites_world_m": {
                site_name: _site_xpos(physics, site_name)
                for site_name in OLD_BAT_SITES
            },
        }
        rec.update(_metric_snapshot(env))
        records.append(rec)
        if done:
            break
        action = _residual_at_step(
            residual,
            action_shape=env.action_space.shape,
            knots=knots,
            step_idx=step_idx,
            total_steps=max_steps,
        )
        _, reward, done, info = env.step(action)

    physics = env.dm_env.physics
    model_counts = {
        "nq": int(physics.model.nq),
        "nv": int(physics.model.nv),
        "nu": int(physics.model.nu),
        "nbody": int(physics.model.nbody),
        "ngeom": int(physics.model.ngeom),
        "nsite": int(physics.model.nsite),
    }
    _annotate_bat_speed(records)
    return records, frames, model_counts


def _annotate_bat_speed(records: list[dict[str, Any]], *, dt: float = 0.01) -> None:
    prev: dict[str, np.ndarray] | None = None
    for rec in records:
        current: dict[str, np.ndarray] = {}
        speeds: dict[str, float] = {}
        velocities: dict[str, list[float]] = {}
        for site, value in (rec.get("bat_sites_world_m") or {}).items():
            if value is None:
                continue
            point = np.asarray(value, dtype=np.float64)
            current[site] = point
            if prev is None or site not in prev:
                vel = np.zeros(3, dtype=np.float64)
            else:
                vel = (point - prev[site]) / float(dt)
            velocities[site] = vel.tolist()
            speeds[site] = float(np.linalg.norm(vel))
        rec["bat_site_velocity_mps"] = velocities
        rec["bat_site_speed_mps"] = speeds
        prev = current


def _site_series(records: list[dict[str, Any]], site: str) -> np.ndarray:
    values = []
    for rec in records:
        value = (rec.get("bat_sites_world_m") or {}).get(site)
        if value is None:
            return np.zeros((0, 3), dtype=np.float64)
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _target_speed_at(
    target: dict[str, Any],
    *,
    site: str,
    physical_step: int,
    args: argparse.Namespace,
) -> float | None:
    target_step = (
        int(physical_step)
        - int(args.physical_contact_step)
        + int(args.kinematic_contact_step)
    )
    speeds = target["speeds_mps"][site]
    if target_step < 0 or target_step >= len(speeds):
        return None
    return float(speeds[target_step])


def _score(
    records: list[dict[str, Any]],
    residual: np.ndarray,
    baseline_residual: np.ndarray,
    target: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    pose = _pose_metrics(records, args.focus_steps)
    site = str(args.target_site)
    current_points = _site_series(records, site)
    current_speed = _speed(current_points) if len(current_points) else np.zeros(0)

    focus_steps = [
        step
        for step in sorted(args.focus_steps)
        if 0 <= step < len(current_speed)
        and _target_speed_at(target, site=site, physical_step=step, args=args) is not None
    ]
    if not focus_steps:
        focus_steps = list(range(len(current_speed)))
    current_focus = np.asarray([current_speed[step] for step in focus_steps], dtype=np.float64)
    target_focus = np.asarray(
        [
            _target_speed_at(target, site=site, physical_step=step, args=args)
            or 0.0
            for step in focus_steps
        ],
        dtype=np.float64,
    )
    if len(current_focus):
        speed_rmse = float(np.sqrt(np.mean(np.square(current_focus - target_focus))))
        speed_ratio = float(np.mean(current_focus) / max(float(np.mean(target_focus)), 1.0e-8))
        focus_peak_idx = int(np.argmax(current_focus))
        current_peak = float(current_focus[focus_peak_idx])
        current_peak_step = int(focus_steps[focus_peak_idx])
        target_peak = float(np.max(target_focus))
    else:
        speed_rmse = 0.0
        speed_ratio = 0.0
        current_peak = 0.0
        current_peak_step = None
        target_peak = 0.0

    contact_step = min(max(0, int(args.physical_contact_step)), max(0, len(current_speed) - 1))
    contact_speed = float(current_speed[contact_step]) if len(current_speed) else 0.0
    target_contact_speed = (
        _target_speed_at(target, site=site, physical_step=contact_step, args=args) or 0.0
    )
    speed_shortfall = max(0.0, float(args.target_speed_floor_mps) - current_peak)
    contact_shortfall = max(0.0, float(args.contact_speed_floor_mps) - contact_speed)
    prior_delta = float(np.mean(np.square(residual - baseline_residual)))

    score = 0.0
    score += float(args.speed_rmse_weight) * speed_rmse
    score -= float(args.peak_speed_weight) * current_peak
    score -= float(args.contact_speed_weight) * contact_speed
    score += float(args.target_speed_floor_weight) * speed_shortfall * speed_shortfall
    score += float(args.contact_speed_floor_weight) * contact_shortfall * contact_shortfall
    score += float(args.prior_weight) * prior_delta
    score += float(args.body_weight) * float(pose["body_abs_mean"] or 0.0)
    score += float(args.joint_weight) * float(pose["joint_abs_mean"] or 0.0)
    score += float(args.appendage_weight) * float(pose["appendages_l2_mean"] or 0.0)
    score += float(args.end_effector_weight) * float(pose["end_effectors_l2_mean"] or 0.0)
    score += float(args.focus_body_weight) * float(pose["focus_body_abs_mean"] or 0.0)
    score += float(args.max_body_weight) * float(pose["max_body_abs"] or 0.0)
    if args.expected_frames > 0 and len(records) < args.expected_frames:
        score += float(args.shortfall_penalty) * (
            args.expected_frames - len(records)
        ) / float(args.expected_frames)

    diagnostics = {
        "pose": pose,
        "frames": len(records),
        "target_site": site,
        "focus_steps": focus_steps,
        "speed_rmse_mps": speed_rmse,
        "speed_mean_ratio": speed_ratio,
        "current_peak_speed_mps": current_peak,
        "current_peak_speed_mph": current_peak * MPH_PER_MPS,
        "current_peak_step": current_peak_step,
        "target_peak_speed_mps": target_peak,
        "target_peak_speed_mph": target_peak * MPH_PER_MPS,
        "physical_contact_step": int(args.physical_contact_step),
        "current_contact_speed_mps": contact_speed,
        "current_contact_speed_mph": contact_speed * MPH_PER_MPS,
        "target_contact_speed_mps": target_contact_speed,
        "target_contact_speed_mph": target_contact_speed * MPH_PER_MPS,
        "target_speed_floor_mps": float(args.target_speed_floor_mps),
        "target_speed_shortfall_mps": speed_shortfall,
        "contact_speed_floor_mps": float(args.contact_speed_floor_mps),
        "contact_speed_shortfall_mps": contact_shortfall,
        "prior_delta": prior_delta,
        "score": float(score),
    }
    return float(score), diagnostics


def search(args: argparse.Namespace) -> dict[str, Any]:
    import imageio.v2 as imageio

    _install_scene(args)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = _kinematic_target(args)
    _write_json(run_dir / "kinematic_bat_speed_target.json", target)

    rng = np.random.default_rng(args.seed)
    env = _make_env(args)
    try:
        action_dim = int(np.prod(env.action_space.shape))
        baseline_residual = _load_initial_residual(
            args.initial_residual,
            action_dim=action_dim,
            knots=args.knots,
        )
        mean = baseline_residual.copy()
        dim = int(mean.size)
        std = np.full(dim, float(args.initial_std), dtype=np.float32)
        best: dict[str, Any] | None = None
        trace: list[dict[str, Any]] = []

        baseline_records, _, model_counts = _rollout(
            env,
            baseline_residual,
            max_steps=args.max_steps,
            knots=args.knots,
            render=False,
            width=args.width,
            height=args.height,
            camera_id=args.camera_id,
        )
        baseline_score, baseline_diag = _score(
            baseline_records,
            baseline_residual,
            baseline_residual,
            target,
            args,
        )
        baseline = {"score": baseline_score, "diagnostics": baseline_diag, "records": None}
        print(
            "baseline score={score:.3f} frames={frames} peak={peak:.2f}mps "
            "contact={contact:.2f}mps target_contact={target:.2f}mps rmse={rmse:.2f}".format(
                score=baseline_score,
                frames=baseline_diag["frames"],
                peak=baseline_diag["current_peak_speed_mps"],
                contact=baseline_diag["current_contact_speed_mps"],
                target=baseline_diag["target_contact_speed_mps"],
                rmse=baseline_diag["speed_rmse_mps"],
            ),
            flush=True,
        )

        for iteration in range(args.iters):
            candidates = [mean.copy()]
            while len(candidates) < args.population:
                sample = mean + std * rng.standard_normal(dim)
                candidates.append(np.clip(sample, -1.0, 1.0).astype(np.float32))
            scored: list[dict[str, Any]] = []
            for index, residual in enumerate(candidates):
                records, _, _ = _rollout(
                    env,
                    residual,
                    max_steps=args.max_steps,
                    knots=args.knots,
                    render=False,
                    width=args.width,
                    height=args.height,
                    camera_id=args.camera_id,
                )
                score, diagnostics = _score(records, residual, baseline_residual, target, args)
                item = {
                    "iteration": iteration,
                    "index": index,
                    "score": score,
                    "frames": len(records),
                    "diagnostics": diagnostics,
                    "residual": residual.tolist(),
                }
                scored.append(item)
                if best is None or score < best["score"]:
                    best = item
            scored.sort(key=lambda item: item["score"])
            elites = np.asarray(
                [item["residual"] for item in scored[: args.elites]],
                dtype=np.float32,
            )
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), float(args.min_std))
            first = scored[0]
            trace.append(
                {
                    "iteration": iteration,
                    "best_score": first["score"],
                    "best_frames": first["frames"],
                    "best_peak_speed_mps": first["diagnostics"]["current_peak_speed_mps"],
                    "best_contact_speed_mps": first["diagnostics"]["current_contact_speed_mps"],
                    "best_target_contact_speed_mps": first["diagnostics"]["target_contact_speed_mps"],
                    "best_speed_rmse_mps": first["diagnostics"]["speed_rmse_mps"],
                    "best_body": first["diagnostics"]["pose"]["body_abs_mean"],
                    "mean_std": float(np.mean(std)),
                }
            )
            print(
                "iter={it} score={score:.3f} frames={frames} peak={peak:.2f}mps "
                "contact={contact:.2f}mps target={target:.2f}mps rmse={rmse:.2f} "
                "body={body:.4f}".format(
                    it=iteration,
                    score=first["score"],
                    frames=first["frames"],
                    peak=first["diagnostics"]["current_peak_speed_mps"],
                    contact=first["diagnostics"]["current_contact_speed_mps"],
                    target=first["diagnostics"]["target_contact_speed_mps"],
                    rmse=first["diagnostics"]["speed_rmse_mps"],
                    body=first["diagnostics"]["pose"]["body_abs_mean"] or float("nan"),
                ),
                flush=True,
            )
            assert best is not None
            np.save(run_dir / "partial_best_residual.npy", np.asarray(best["residual"], dtype=np.float32))
            _write_json(
                run_dir / "partial_summary.json",
                {
                    "status": "partial",
                    "platform": platform.platform(),
                    "run_dir": str(run_dir),
                    "baseline": baseline,
                    "best": best,
                    "trace": trace,
                    "target": {k: v for k, v in target.items() if k not in {"sites"}},
                    "model_counts": model_counts,
                    "config": _json_safe(vars(args)),
                },
            )

        assert best is not None
        best_residual = np.asarray(best["residual"], dtype=np.float32)
        records, frames, model_counts = _rollout(
            env,
            best_residual,
            max_steps=args.max_steps,
            knots=args.knots,
            render=True,
            width=args.width,
            height=args.height,
            camera_id=args.camera_id,
        )
        score, diagnostics = _score(records, best_residual, baseline_residual, target, args)
        video_path = run_dir / "best_batspeed_imitation.mp4"
        imageio.mimwrite(video_path, frames, fps=30)
        summary = {
            "platform": platform.platform(),
            "video_path": str(video_path),
            "run_dir": str(run_dir),
            "score": score,
            "frames": len(records),
            "terminated": bool(records[-1]["done"]) if records else None,
            "baseline": baseline,
            "diagnostics": diagnostics,
            "best": best,
            "trace": trace,
            "target": {k: v for k, v in target.items() if k not in {"sites"}},
            "model_counts": model_counts,
            "physical_scene": {
                "bat_mesh": str(args.bat_mesh),
                "parent_hand": args.parent_hand,
                "bat_pos": list(args.bat_pos),
                "bat_euler_deg": list(args.bat_euler_deg),
                "ball_mode": "none",
            },
            "config": _json_safe(vars(args)),
            "records": records,
        }
        _write_json(run_dir / "summary.json", summary)
        np.save(run_dir / "best_residual.npy", best_residual)
        print(f"Wrote {video_path}", flush=True)
        print(f"Wrote {run_dir / 'summary.json'}", flush=True)
        print(
            "final score={score:.3f} frames={frames} peak={peak:.2f}mps "
            "contact={contact:.2f}mps target={target:.2f}mps rmse={rmse:.2f}".format(
                score=score,
                frames=len(records),
                peak=diagnostics["current_peak_speed_mps"],
                contact=diagnostics["current_contact_speed_mps"],
                target=diagnostics["target_contact_speed_mps"],
                rmse=diagnostics["speed_rmse_mps"],
            ),
            flush=True,
        )
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--constructor-path", type=Path)
    parser.add_argument("--mocap-path", type=Path, default=DEFAULT_PHYSICAL_MOCAP)
    parser.add_argument("--clip-snippet", default="CMU_124_07-0-100")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="batspeed_imitation_cem")
    parser.add_argument("--initial-residual", type=Path, default=DEFAULT_INITIAL_RESIDUAL)
    parser.add_argument("--amc-path", type=Path, default=DEFAULT_AMC)
    parser.add_argument("--kinematic-xml", type=Path, default=DEFAULT_KINEMATIC_XML)
    parser.add_argument("--kinematic-start-frame", type=int, default=280)
    parser.add_argument("--kinematic-steps", type=int, default=80)
    parser.add_argument("--kinematic-contact-step", type=int, default=9)
    parser.add_argument("--physical-contact-step", type=int, default=35)
    parser.add_argument("--bat-mesh", type=Path, default=Path("/workspace/exitvelo/assets/meshes/baseball_bat.stl"))
    parser.add_argument("--parent-hand", choices=("lhand", "rhand"), default="lhand")
    parser.add_argument("--bat-pos", type=float, nargs=3, default=OLD_BAT_POS)
    parser.add_argument("--bat-euler-deg", type=float, nargs=3, default=DEFAULT_BAT_EULER_DEG)
    parser.add_argument("--two-hand-anchor", type=float, nargs=3, default=(-0.04, 0.04, 0.06))
    parser.add_argument("--tiny-mass", type=float, default=0.0)
    parser.add_argument("--target-site", choices=tuple(OLD_BAT_SITES), default="bat_sweet")
    parser.add_argument("--focus-steps", type=_parse_step_ranges, default=_parse_step_ranges("25-50"))
    parser.add_argument("--knots", type=int, default=12)
    parser.add_argument("--residual-scale", type=float, default=0.18)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--initial-std", type=float, default=0.04)
    parser.add_argument("--min-std", type=float, default=0.008)
    parser.add_argument("--speed-rmse-weight", type=float, default=1.0)
    parser.add_argument("--peak-speed-weight", type=float, default=6.0)
    parser.add_argument("--contact-speed-weight", type=float, default=8.0)
    parser.add_argument("--target-speed-floor-mps", type=float, default=10.0)
    parser.add_argument("--target-speed-floor-weight", type=float, default=2.0)
    parser.add_argument("--contact-speed-floor-mps", type=float, default=10.0)
    parser.add_argument("--contact-speed-floor-weight", type=float, default=2.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--body-weight", type=float, default=2.0)
    parser.add_argument("--joint-weight", type=float, default=0.3)
    parser.add_argument("--appendage-weight", type=float, default=0.6)
    parser.add_argument("--end-effector-weight", type=float, default=0.8)
    parser.add_argument("--focus-body-weight", type=float, default=2.0)
    parser.add_argument("--max-body-weight", type=float, default=3.0)
    parser.add_argument("--shortfall-penalty", type=float, default=50.0)
    parser.add_argument("--expected-frames", type=int, default=95)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--termination-error-threshold", type=float, default=10.0)
    parser.add_argument("--act-noise", type=float, default=0.0)
    parser.add_argument("--ghost-offset", type=float, default=1.25)
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--base-device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-id", type=int, default=3)
    args = parser.parse_args()
    if args.elites <= 0 or args.elites > args.population:
        raise ValueError("--elites must be in [1, population]")
    search(args)


if __name__ == "__main__":
    main()
