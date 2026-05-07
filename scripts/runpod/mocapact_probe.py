"""RunPod MoCapAct probe for Exitvelo.

Run this on Linux after MoCapAct is installed. It verifies whether target CMU
clips load, steps the tracking environment briefly, and exports a neutral
rollout JSON compatible with Exitvelo's local MoCapAct adapter gates.

Local Windows usage is limited to ``--dry-run``; do not import MoCapAct here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_CLIPS = ["CMU_124_07", "CMU_124_08", "CMU_124_06", "CMU_016_22"]
DEFAULT_CANDIDATES = [
    {"clip_id": "CMU_124_07", "start_step": 260, "end_step": 360, "role": "primary_exitvelo_target"},
    {"clip_id": "CMU_124_08", "start_step": 260, "end_step": 360, "role": "secondary_exitvelo_raw_clip"},
    {"clip_id": "CMU_124_06", "start_step": 0, "end_step": 189, "role": "nearby_subject124_split_fallback"},
    {"clip_id": "CMU_016_22", "start_step": 0, "end_step": 82, "role": "known_mocapact_control"},
]


def _space_summary(space: Any) -> Any:
    if hasattr(space, "spaces"):
        return {key: _space_summary(value) for key, value in space.spaces.items()}
    out = {"type": type(space).__name__}
    for attr in ("shape", "dtype", "n"):
        if hasattr(space, attr):
            value = getattr(space, attr)
            out[attr] = str(value) if attr == "dtype" else value
    return out


def _array(value: Any) -> list[float]:
    return [float(x) for x in value]


def _physics_qpos_qvel(env: Any) -> tuple[list[float], list[float]]:
    physics = env.dm_env.physics
    qpos = physics.data.qpos.copy()
    qvel = physics.data.qvel.copy()
    return _array(qpos), _array(qvel)


def _make_env(
    *,
    clip_id: str,
    start_step: int | None,
    end_step: int | None,
    ref_steps: tuple[int, ...],
    mocap_path: str | None,
    min_steps: int,
    ghost_offset: float,
    termination_error_threshold: float,
    act_noise: float,
    always_init_at_clip_start: bool,
):
    import numpy as np

    from dm_control.locomotion.tasks.reference_pose import types
    from mocapact.envs import tracking

    try:
        from mocapact.clip_expert import utils as clip_expert_utils

        env_kwargs = clip_expert_utils.make_env_kwargs(
            clip_id=clip_id,
            mocap_path=mocap_path,
            start_step=start_step,
            end_step=end_step,
            min_steps=min_steps,
            ghost_offset=ghost_offset,
            always_init_at_clip_start=always_init_at_clip_start,
            termination_error_threshold=termination_error_threshold,
            act_noise=act_noise,
        )
        env_kwargs["ref_steps"] = ref_steps
        return tracking.MocapTrackingGymEnv(**env_kwargs)
    except Exception as exc:  # noqa: BLE001 - record utility failures but still try the raw env path
        utility_error = f"{type(exc).__name__}: {exc}"

    try:
        kwargs: dict[str, Any] = {"ids": [clip_id]}
        if start_step is not None:
            kwargs["start_steps"] = [int(start_step)]
        if end_step is not None:
            kwargs["end_steps"] = [int(end_step)]
        dataset = types.ClipCollection(**kwargs)
        task_kwargs = {
            "min_steps": max(0, int(min_steps) - 1),
            "ghost_offset": np.array([float(ghost_offset), 0.0, 0.0]),
            "always_init_at_clip_start": bool(always_init_at_clip_start),
            "termination_error_threshold": float(termination_error_threshold),
        }
        env = tracking.MocapTrackingGymEnv(
            dataset=dataset,
            ref_steps=ref_steps,
            mocap_path=mocap_path,
            act_noise=act_noise,
            task_kwargs=task_kwargs,
        )
    except Exception as raw_exc:  # noqa: BLE001
        raise RuntimeError(
            "failed with mocapact.clip_expert.utils.make_env_kwargs "
            f"({utility_error}); raw MocapTrackingGymEnv fallback also failed "
            f"({type(raw_exc).__name__}: {raw_exc})"
        ) from raw_exc
    setattr(env, "_exitvelo_probe_utility_error", utility_error)
    return env


def _reset_env(env: Any) -> Any:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, float(reward), bool(terminated or truncated), dict(info)
    obs, reward, done, info = step_result
    return obs, float(reward), bool(done), dict(info)


def _probe_clip(
    *,
    clip_id: str,
    start_step: int | None,
    end_step: int | None,
    role: str | None,
    rollout_steps: int,
    zero_action: bool,
    ref_steps: tuple[int, ...],
    mocap_path: str | None,
    min_steps: int,
    ghost_offset: float,
    termination_error_threshold: float,
    act_noise: float,
    always_init_at_clip_start: bool,
) -> dict:
    import numpy as np

    result: dict[str, Any] = {
        "clip_id": clip_id,
        "start_step": start_step,
        "end_step": end_step,
        "role": role,
        "loaded": False,
        "stepped": False,
        "error": None,
    }
    try:
        env = _make_env(
            clip_id=clip_id,
            start_step=start_step,
            end_step=end_step,
            ref_steps=ref_steps,
            mocap_path=mocap_path,
            min_steps=min_steps,
            ghost_offset=ghost_offset,
            termination_error_threshold=termination_error_threshold,
            act_noise=act_noise,
            always_init_at_clip_start=always_init_at_clip_start,
        )
        utility_error = getattr(env, "_exitvelo_probe_utility_error", None)
        if utility_error:
            result["clip_expert_utility_error"] = utility_error
        result["loaded"] = True
        result["observation_space"] = _space_summary(env.observation_space)
        result["action_space"] = _space_summary(env.action_space)
        obs = _reset_env(env)
        result["reset_observation_keys"] = sorted(obs.keys())
        qpos, qvel = _physics_qpos_qvel(env)
        result["qpos_dim"] = len(qpos)
        result["qvel_dim"] = len(qvel)
        result["action_dim"] = int(np.prod(env.action_space.shape))

        frames = []
        done = False
        reward = 0.0
        info: dict[str, Any] = {}
        for step_idx in range(max(0, rollout_steps)):
            qpos, qvel = _physics_qpos_qvel(env)
            action = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                if zero_action
                else env.action_space.sample()
            )
            frames.append({
                "t": float(step_idx * 0.03),
                "qpos": qpos,
                "qvel": qvel,
                "action": _array(action.ravel()),
                "reward": float(reward),
                "terminated": bool(done),
                "info": {
                    "time_in_clip": _json_float(info.get("time_in_clip")),
                    "start_time_in_clip": _json_float(info.get("start_time_in_clip")),
                    "last_time_in_clip": _json_float(info.get("last_time_in_clip")),
                },
            })
            if done:
                break
            _, reward, done, info = _step_env(env, action)
        result["stepped"] = rollout_steps == 0 or bool(frames)
        result["rollout"] = {
            "schema_version": 1,
            "source": "mocapact",
            "clip_id": clip_id,
            "start_step": int(start_step or 0),
            "end_step": int(end_step or ((start_step or 0) + len(frames))),
            "control_dt": 0.03,
            "physics_dt": None,
            "frames": frames,
        }
        close = getattr(env, "close", None)
        if callable(close):
            close()
    except Exception as exc:  # noqa: BLE001 - probe should report all failures
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _parse_ref_steps(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("ref steps must contain at least one integer")
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ref steps: {raw}") from exc


def _parse_candidate(raw: str) -> dict[str, Any]:
    parts = raw.split(":")
    if len(parts) == 1:
        return {"clip_id": parts[0], "start_step": None, "end_step": None, "role": "custom"}
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "candidate must be CLIP_ID or CLIP_ID:START_STEP:END_STEP[:ROLE]"
        )
    try:
        start_step = int(parts[1])
        end_step = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid candidate window: {raw}") from exc
    role = parts[3] if len(parts) == 4 else "custom"
    return {"clip_id": parts[0], "start_step": start_step, "end_step": end_step, "role": role}


def _candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidates:
        return [_parse_candidate(candidate) for candidate in args.candidates]
    if args.clip_ids:
        return [
            {
                "clip_id": clip_id,
                "start_step": args.start_step,
                "end_step": args.end_step,
                "role": "global_window",
            }
            for clip_id in args.clip_ids
        ]
    return list(DEFAULT_CANDIDATES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/runpod_mocapact_probe"))
    parser.add_argument("--clip-id", action="append", dest="clip_ids", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        default=None,
        help="Candidate as CLIP_ID or CLIP_ID:START_STEP:END_STEP[:ROLE]. Overrides default candidates.",
    )
    parser.add_argument("--start-step", type=int, default=260)
    parser.add_argument("--end-step", type=int, default=360)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--ref-steps", type=_parse_ref_steps, default=(0,))
    parser.add_argument("--mocap-path", type=Path, default=None)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--ghost-offset", type=float, default=1.0)
    parser.add_argument("--termination-error-threshold", type=float, default=0.3)
    parser.add_argument("--act-noise", type=float, default=0.0)
    parser.add_argument("--always-init-at-clip-start", action="store_true")
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = _candidate_specs(args)
    clip_ids = [candidate["clip_id"] for candidate in candidates]
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "dry_run": bool(args.dry_run),
        "clip_ids": clip_ids,
        "candidate_specs": candidates,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "rollout_steps": args.rollout_steps,
        "ref_steps": list(args.ref_steps),
        "mocap_path": str(args.mocap_path) if args.mocap_path else None,
        "min_steps": args.min_steps,
        "ghost_offset": args.ghost_offset,
        "termination_error_threshold": args.termination_error_threshold,
        "act_noise": args.act_noise,
        "always_init_at_clip_start": bool(args.always_init_at_clip_start),
        "zero_action": bool(args.zero_action),
        "clips": [],
        "best_rollout_path": None,
    }

    if args.dry_run:
        report["message"] = "dry run only; MoCapAct was not imported"
        _write_json(args.out_dir / "mocapact_probe_report.json", report)
        print(f"Wrote {args.out_dir / 'mocapact_probe_report.json'}")
        return

    for candidate in candidates:
        result = _probe_clip(
            clip_id=candidate["clip_id"],
            start_step=candidate.get("start_step"),
            end_step=candidate.get("end_step"),
            role=candidate.get("role"),
            rollout_steps=args.rollout_steps,
            zero_action=args.zero_action,
            ref_steps=args.ref_steps,
            mocap_path=str(args.mocap_path) if args.mocap_path else None,
            min_steps=args.min_steps,
            ghost_offset=args.ghost_offset,
            termination_error_threshold=args.termination_error_threshold,
            act_noise=args.act_noise,
            always_init_at_clip_start=bool(args.always_init_at_clip_start),
        )
        rollout = result.pop("rollout", None)
        if rollout is not None and result.get("loaded"):
            rollout_path = args.out_dir / f"{candidate['clip_id']}_rollout.json"
            _write_json(rollout_path, rollout)
            result["rollout_path"] = str(rollout_path)
            if report["best_rollout_path"] is None and rollout.get("frames"):
                report["best_rollout_path"] = str(rollout_path)
        report["clips"].append(result)

    _write_json(args.out_dir / "mocapact_probe_report.json", report)
    print(f"Wrote {args.out_dir / 'mocapact_probe_report.json'}")
    if report["best_rollout_path"]:
        print(f"Wrote {report['best_rollout_path']}")

    ok = any(clip.get("loaded") for clip in report["clips"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
