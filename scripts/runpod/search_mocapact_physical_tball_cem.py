"""Physical tee-ball distance search for the MoCapAct batting tracker.

This is the Stage 2 transition from "the body can imitate the swing with a
massful welded bat" to "the body can perturb that swing to hit a tee ball
farther."  The search keeps the approved MoCapAct residual as a prior, installs
the original Exitvelo bat asset as a fixed body on the CMU hand, places a free
baseball at the current tee location, and runs a small CEM residual search.

The objective is intentionally simple:

* require real MuJoCo bat-ball contact
* score the finite-difference baseball velocity after contact/release
* convert that launch velocity to a no-drag carry estimate
* keep the body close to the mocap tracker with pose/prior penalties

The final candidate is rendered, but CEM evaluations are non-rendering so this
can run cheaply on the existing RunPod.
"""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from render_mocapact_old_bat_asset import (
    OLD_BAT_POS,
    OLD_BAT_SITES,
    _annotate_ball_velocity,
    _body_xpos,
    _body_xvelp,
    _contact_snapshot,
    _install_old_bat_walker,
    _make_env,
    _set_free_ball_state,
    _site_xpos,
)
from search_mocapact_residual_bias_cem import _residual_at_step
from search_mocapact_virtual_tball_cem import _carry_ft, _load_initial_residual, _pose_metrics
from train_mocapact_multiclip_residual_ppo import (
    DEFAULT_MOCAP,
    DEFAULT_OUT_DIR,
    DEFAULT_POLICY,
    _json_safe,
    _metric_snapshot,
    _write_json,
)


MPH_PER_MPS = 2.2369362920544
DEFAULT_CORRECTED_BAT_EULER_DEG = (210.0, -90.0, -10.0)
DEFAULT_CORRECTED_TEE_POS = (
    -0.28229016260805756,
    -0.11311196784975563,
    1.2160991328740112,
)
DEFAULT_PHYSICAL_MOCAP = Path(
    "/workspace/exitvelo/results/mocapact_custom/cmu_124_07_stride1.h5"
)
DEFAULT_INITIAL_RESIDUAL = Path(
    "/workspace/exitvelo/results/mocapact_residual_ppo/"
    "virtual_bat_tball_speedkick3860_twohand_pop16_iter2/best_residual.npy"
)


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
    ball_hold_until_step: int,
    ball_hold_pos: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    env.reset()
    reward = 0.0
    done = False
    info: dict[str, Any] = {}
    held_ball_supported = False

    for step_idx in range(max_steps):
        if ball_hold_until_step >= 0 and step_idx <= ball_hold_until_step:
            held_ball_supported = (
                _set_free_ball_state(env.dm_env.physics, ball_hold_pos)
                or held_ball_supported
            )
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
            "baseball_world_m": _body_xpos(physics, "baseball"),
            "baseball_linear_velocity_mps": _body_xvelp(physics, "baseball"),
            "contact": _contact_snapshot(physics),
            "ball_held": bool(ball_hold_until_step >= 0 and step_idx <= ball_hold_until_step),
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
        if ball_hold_until_step >= 0 and step_idx < ball_hold_until_step:
            held_ball_supported = (
                _set_free_ball_state(env.dm_env.physics, ball_hold_pos)
                or held_ball_supported
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
        "held_ball_state_supported": bool(held_ball_supported),
    }
    _annotate_ball_velocity(records)
    return records, frames, model_counts


def _first_contact_step(records: list[dict[str, Any]]) -> int | None:
    return next(
        (
            int(rec["step"])
            for rec in records
            if rec.get("contact", {}).get("bat_ball_contacts")
        ),
        None,
    )


def _launch_record(
    records: list[dict[str, Any]],
    *,
    first_contact_step: int | None,
    release_step: int,
    score_window_steps: int,
) -> dict[str, Any] | None:
    if first_contact_step is None:
        return None
    candidates = [
        rec
        for rec in records
        if int(rec["step"]) > first_contact_step
        and int(rec["step"]) >= release_step
        and int(rec["step"]) <= first_contact_step + score_window_steps
        and rec.get("baseball_fd_velocity_mps") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda rec: float(rec.get("baseball_fd_speed_mps") or 0.0))


def _launch_angle_deg(velocity: np.ndarray) -> float:
    horizontal = float(np.linalg.norm(velocity[:2]))
    if horizontal < 1.0e-8:
        return 90.0 if float(velocity[2]) >= 0.0 else -90.0
    return math.degrees(math.atan2(float(velocity[2]), horizontal))


def _contact_timing_penalty(first_contact_step: int | None, args: argparse.Namespace) -> float:
    if first_contact_step is None or args.contact_target_step < 0:
        return 0.0
    half_window = max(0.0, float(args.contact_window_steps) / 2.0)
    deviation = abs(float(first_contact_step) - float(args.contact_target_step))
    return max(0.0, deviation - half_window) ** 2


def _site_speed_at(
    records: list[dict[str, Any]],
    *,
    site_name: str,
    step: int | None,
    dt: float = 0.01,
) -> float:
    if step is None:
        return 0.0
    by_step = {int(rec["step"]): rec for rec in records}
    rec = by_step.get(int(step))
    prev = by_step.get(int(step) - 1)
    if rec is None or prev is None:
        return 0.0
    site = (rec.get("bat_sites_world_m") or {}).get(site_name)
    prev_site = (prev.get("bat_sites_world_m") or {}).get(site_name)
    if site is None or prev_site is None:
        return 0.0
    vel = (np.asarray(site, dtype=np.float64) - np.asarray(prev_site, dtype=np.float64)) / float(dt)
    return float(np.linalg.norm(vel))


def _score(
    records: list[dict[str, Any]],
    residual: np.ndarray,
    baseline_residual: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    pose = _pose_metrics(records, args.focus_steps)
    first_contact = _first_contact_step(records)
    release_step = max(0, int(args.ball_hold_until_step) + 1)
    launch = _launch_record(
        records,
        first_contact_step=first_contact,
        release_step=release_step,
        score_window_steps=int(args.score_window_steps),
    )
    contact_frames = sum(
        1 for rec in records if rec.get("contact", {}).get("bat_ball_contacts")
    )

    if launch is None:
        launch_velocity = np.zeros(3, dtype=np.float64)
        launch_step = None
        launch_speed_mps = 0.0
        launch_pos = np.asarray(args.ball_hold_pos, dtype=np.float64)
    else:
        launch_velocity = np.asarray(launch["baseball_fd_velocity_mps"], dtype=np.float64)
        launch_step = int(launch["step"])
        launch_speed_mps = float(np.linalg.norm(launch_velocity))
        launch_pos = np.asarray(launch.get("baseball_world_m") or args.ball_hold_pos, dtype=np.float64)

    carry_ft = _carry_ft(launch_velocity, float(launch_pos[2]))
    launch_angle = _launch_angle_deg(launch_velocity)
    exit_speed_mph = launch_speed_mps * MPH_PER_MPS
    horizontal_speed_mps = float(np.linalg.norm(launch_velocity[:2]))
    upward_speed_mps = max(0.0, float(launch_velocity[2]))
    downward_speed_mps = max(0.0, -float(launch_velocity[2]))
    launch_angle_shortfall = max(
        0.0,
        float(args.launch_angle_floor_deg) - float(launch_angle),
    )
    bat_speed_mps = _site_speed_at(
        records,
        site_name=str(args.bat_speed_site),
        step=first_contact,
    )
    bat_speed_shortfall = max(0.0, float(args.bat_speed_target_mps) - bat_speed_mps)
    prior_delta = float(np.mean(np.square(residual - baseline_residual)))
    timing_penalty = _contact_timing_penalty(first_contact, args)

    score = 0.0
    # Negative terms are objectives because CEM minimizes the score.
    score -= float(args.carry_weight) * carry_ft
    score -= float(args.exit_speed_weight) * exit_speed_mph
    score -= float(args.horizontal_speed_weight) * horizontal_speed_mps
    score -= float(args.upward_speed_weight) * upward_speed_mps
    score -= float(args.bat_speed_weight) * bat_speed_mps
    score += float(args.downward_speed_weight) * downward_speed_mps
    score += float(args.launch_angle_floor_weight) * launch_angle_shortfall * launch_angle_shortfall
    score += float(args.bat_speed_target_weight) * bat_speed_shortfall * bat_speed_shortfall
    score += float(args.prior_weight) * prior_delta
    score += float(args.body_weight) * float(pose["body_abs_mean"] or 0.0)
    score += float(args.joint_weight) * float(pose["joint_abs_mean"] or 0.0)
    score += float(args.appendage_weight) * float(pose["appendages_l2_mean"] or 0.0)
    score += float(args.end_effector_weight) * float(pose["end_effectors_l2_mean"] or 0.0)
    score += float(args.focus_body_weight) * float(pose["focus_body_abs_mean"] or 0.0)
    score += float(args.max_body_weight) * float(pose["max_body_abs"] or 0.0)
    score += float(args.max_hand_weight) * max(
        float(pose["max_right_hand_l2"] or 0.0),
        float(pose["max_left_hand_l2"] or 0.0),
    )
    score += float(args.contact_timing_weight) * timing_penalty
    if first_contact is None:
        score += float(args.no_contact_penalty)
    if launch is None:
        score += float(args.no_launch_penalty)
    if contact_frames < int(args.min_contact_frames):
        score += float(args.contact_frame_penalty) * (
            int(args.min_contact_frames) - contact_frames
        )
    if args.expected_frames > 0 and len(records) < args.expected_frames:
        score += float(args.shortfall_penalty) * (
            args.expected_frames - len(records)
        ) / float(args.expected_frames)

    diagnostics = {
        "pose": pose,
        "frames": len(records),
        "first_bat_ball_contact_step": first_contact,
        "contact_frames": contact_frames,
        "release_step": release_step,
        "launch_step": launch_step,
        "launch_velocity_mps": launch_velocity.tolist(),
        "launch_speed_mps": launch_speed_mps,
        "exit_speed_mph": exit_speed_mph,
        "horizontal_speed_mps": horizontal_speed_mps,
        "upward_speed_mps": upward_speed_mps,
        "downward_speed_mps": downward_speed_mps,
        "launch_angle_deg": launch_angle,
        "launch_angle_floor_deg": float(args.launch_angle_floor_deg),
        "launch_angle_shortfall_deg": launch_angle_shortfall,
        "bat_speed_site": str(args.bat_speed_site),
        "bat_speed_mps": bat_speed_mps,
        "bat_speed_mph": bat_speed_mps * MPH_PER_MPS,
        "bat_speed_target_mps": float(args.bat_speed_target_mps),
        "bat_speed_shortfall_mps": bat_speed_shortfall,
        "carry_ft": carry_ft,
        "prior_delta": prior_delta,
        "timing_penalty": timing_penalty,
        "score": float(score),
    }
    return float(score), diagnostics


def _install_physical_scene(args: argparse.Namespace) -> None:
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
        ball_mode="free",
        ball_pos=tuple(float(value) for value in args.ball_pos),
        ball_radius=float(args.ball_radius),
        ball_mass=float(args.ball_mass),
        ball_gravcomp=float(args.ball_gravcomp),
    )


def search(args: argparse.Namespace) -> dict[str, Any]:
    import imageio.v2 as imageio

    _install_physical_scene(args)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
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
            ball_hold_until_step=args.ball_hold_until_step,
            ball_hold_pos=tuple(float(value) for value in args.ball_hold_pos),
        )
        baseline_score, baseline_diagnostics = _score(
            baseline_records,
            baseline_residual,
            baseline_residual,
            args,
        )
        baseline = {
            "score": baseline_score,
            "diagnostics": baseline_diagnostics,
            "records": None,
        }
        print(
            "baseline score={score:.3f} frames={frames} contact={contact} "
            "exit={exit:.2f}mph carry={carry:.2f}ft angle={angle:.1f}deg".format(
                score=baseline_score,
                frames=baseline_diagnostics["frames"],
                contact=baseline_diagnostics["first_bat_ball_contact_step"],
                exit=baseline_diagnostics["exit_speed_mph"],
                carry=baseline_diagnostics["carry_ft"],
                angle=baseline_diagnostics["launch_angle_deg"],
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
                    ball_hold_until_step=args.ball_hold_until_step,
                    ball_hold_pos=tuple(float(value) for value in args.ball_hold_pos),
                )
                score, diagnostics = _score(records, residual, baseline_residual, args)
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
                    "best_contact": first["diagnostics"]["first_bat_ball_contact_step"],
                    "best_launch_step": first["diagnostics"]["launch_step"],
                    "best_exit_speed_mph": first["diagnostics"]["exit_speed_mph"],
                    "best_carry_ft": first["diagnostics"]["carry_ft"],
                    "best_launch_angle_deg": first["diagnostics"]["launch_angle_deg"],
                    "best_bat_speed_mps": first["diagnostics"]["bat_speed_mps"],
                    "mean_std": float(np.mean(std)),
                }
            )
            print(
                "iter={it} score={score:.3f} frames={frames} contact={contact} "
                "launch={launch} exit={exit:.2f}mph carry={carry:.2f}ft "
                "angle={angle:.1f}deg bat={bat:.2f}mps body={body:.4f}".format(
                    it=iteration,
                    score=first["score"],
                    frames=first["frames"],
                    contact=first["diagnostics"]["first_bat_ball_contact_step"],
                    launch=first["diagnostics"]["launch_step"],
                    exit=first["diagnostics"]["exit_speed_mph"],
                    carry=first["diagnostics"]["carry_ft"],
                    angle=first["diagnostics"]["launch_angle_deg"],
                    bat=first["diagnostics"]["bat_speed_mps"],
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
            ball_hold_until_step=args.ball_hold_until_step,
            ball_hold_pos=tuple(float(value) for value in args.ball_hold_pos),
        )
        score, diagnostics = _score(records, best_residual, baseline_residual, args)
        video_path = run_dir / "best_physical_tball_distance.mp4"
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
            "model_counts": model_counts,
            "physical_scene": {
                "bat_mesh": str(args.bat_mesh),
                "parent_hand": args.parent_hand,
                "bat_pos": list(args.bat_pos),
                "bat_euler_deg": list(args.bat_euler_deg),
                "ball_pos": list(args.ball_pos),
                "ball_hold_pos": list(args.ball_hold_pos),
                "ball_hold_until_step": int(args.ball_hold_until_step),
                "ball_radius_m": float(args.ball_radius),
                "ball_mass_kg": float(args.ball_mass),
            },
            "config": _json_safe(vars(args)),
            "records": records,
        }
        _write_json(run_dir / "summary.json", summary)
        np.save(run_dir / "best_residual.npy", best_residual)
        print(f"Wrote {video_path}", flush=True)
        print(f"Wrote {run_dir / 'summary.json'}", flush=True)
        print(
            "final score={score:.3f} frames={frames} contact={contact} "
            "launch={launch} exit={exit:.2f}mph carry={carry:.2f}ft "
            "angle={angle:.1f}deg".format(
                score=score,
                frames=len(records),
                contact=diagnostics["first_bat_ball_contact_step"],
                launch=diagnostics["launch_step"],
                exit=diagnostics["exit_speed_mph"],
                carry=diagnostics["carry_ft"],
                angle=diagnostics["launch_angle_deg"],
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
    parser.add_argument("--run-name", default="physical_tball_distance_cem")
    parser.add_argument("--initial-residual", type=Path, default=DEFAULT_INITIAL_RESIDUAL)
    parser.add_argument("--bat-mesh", type=Path, default=Path("/workspace/exitvelo/assets/meshes/baseball_bat.stl"))
    parser.add_argument("--parent-hand", choices=("lhand", "rhand"), default="lhand")
    parser.add_argument("--bat-pos", type=float, nargs=3, default=OLD_BAT_POS)
    parser.add_argument("--bat-euler-deg", type=float, nargs=3, default=DEFAULT_CORRECTED_BAT_EULER_DEG)
    parser.add_argument("--two-hand-anchor", type=float, nargs=3, default=(-0.04, 0.04, 0.06))
    parser.add_argument("--ball-pos", type=float, nargs=3, default=DEFAULT_CORRECTED_TEE_POS)
    parser.add_argument("--ball-hold-pos", type=float, nargs=3, default=DEFAULT_CORRECTED_TEE_POS)
    parser.add_argument("--ball-hold-until-step", type=int, default=36)
    parser.add_argument("--ball-radius", type=float, default=0.0366)
    parser.add_argument("--ball-mass", type=float, default=0.145)
    parser.add_argument("--ball-gravcomp", type=float, default=1.0)
    parser.add_argument("--knots", type=int, default=12)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elites", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--initial-std", type=float, default=0.018)
    parser.add_argument("--min-std", type=float, default=0.004)
    parser.add_argument("--score-window-steps", type=int, default=10)
    parser.add_argument("--carry-weight", type=float, default=2.0)
    parser.add_argument("--exit-speed-weight", type=float, default=0.45)
    parser.add_argument("--horizontal-speed-weight", type=float, default=0.25)
    parser.add_argument("--upward-speed-weight", type=float, default=0.75)
    parser.add_argument("--downward-speed-weight", type=float, default=0.0)
    parser.add_argument("--launch-angle-floor-deg", type=float, default=-90.0)
    parser.add_argument("--launch-angle-floor-weight", type=float, default=0.0)
    parser.add_argument("--bat-speed-site", choices=tuple(OLD_BAT_SITES), default="bat_sweet")
    parser.add_argument("--bat-speed-weight", type=float, default=0.0)
    parser.add_argument("--bat-speed-target-mps", type=float, default=0.0)
    parser.add_argument("--bat-speed-target-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=6.0)
    parser.add_argument("--body-weight", type=float, default=4.0)
    parser.add_argument("--joint-weight", type=float, default=0.6)
    parser.add_argument("--appendage-weight", type=float, default=1.0)
    parser.add_argument("--end-effector-weight", type=float, default=1.5)
    parser.add_argument("--focus-body-weight", type=float, default=4.0)
    parser.add_argument("--max-body-weight", type=float, default=5.0)
    parser.add_argument("--max-hand-weight", type=float, default=2.0)
    parser.add_argument("--no-contact-penalty", type=float, default=250.0)
    parser.add_argument("--no-launch-penalty", type=float, default=250.0)
    parser.add_argument("--min-contact-frames", type=int, default=1)
    parser.add_argument("--contact-frame-penalty", type=float, default=20.0)
    parser.add_argument("--contact-target-step", type=int, default=35)
    parser.add_argument("--contact-window-steps", type=int, default=12)
    parser.add_argument("--contact-timing-weight", type=float, default=0.35)
    parser.add_argument("--focus-steps", type=_parse_step_ranges, default=_parse_step_ranges("20-60"))
    parser.add_argument("--shortfall-penalty", type=float, default=35.0)
    parser.add_argument("--expected-frames", type=int, default=95)
    parser.add_argument("--tiny-mass", type=float, default=0.0)
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--termination-error-threshold", type=float, default=10.0)
    parser.add_argument("--act-noise", type=float, default=0.0)
    parser.add_argument("--ghost-offset", type=float, default=1.25)
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--base-device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-id", type=int, default=3)
    args = parser.parse_args()
    if args.elites <= 0 or args.elites > args.population:
        raise ValueError("--elites must be in [1, population]")
    search(args)


if __name__ == "__main__":
    main()
