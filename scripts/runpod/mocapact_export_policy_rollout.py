"""Export a trained MoCapAct clip-expert policy as an Exitvelo rollout JSON.

Run this on Linux inside the MoCapAct virtualenv. It is the bridge from a
Stable-Baselines/MoCapAct checkpoint directory:

    eval_start/model/{best_model.zip, vecnormalize.pkl}

to Exitvelo's neutral ``src.motion.mocapact_rollout`` schema. The resulting
JSON can be copied back and fed through:

    scripts/io/export_mocapact_proxy_trajectory.py
    scripts/analysis/evaluate_proxy_batting.py

Local Windows usage should use ``--dry-run`` only; MoCapAct is imported only
for real exports.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("results/mocapact_policy_rollouts/CMU_124_07_policy_rollout.json")


def _array(value: Any) -> list[float]:
    return [float(x) for x in value]


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


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


def _physics_qpos_qvel(env: Any) -> tuple[list[float], list[float]]:
    physics = env.dm_env.physics
    qpos = physics.data.qpos.copy()
    qvel = physics.data.qvel.copy()
    return _array(qpos), _array(qvel)


def _control_dt(env: Any) -> float:
    dm_env = env.dm_env
    value = getattr(dm_env, "control_timestep", None)
    if callable(value):
        return float(value())
    if value is not None:
        return float(value)
    return 0.03


def _physics_dt(env: Any) -> float | None:
    try:
        return float(env.dm_env.physics.model.opt.timestep)
    except Exception:  # noqa: BLE001 - metadata only
        return None


def _read_clip_info(policy_root: Path) -> dict[str, int | str]:
    clip_info_path = policy_root.parent.parent / "clip_info.json"
    if not clip_info_path.exists():
        raise FileNotFoundError(
            f"could not find clip_info.json next to policy root: {clip_info_path}"
        )
    data = json.loads(clip_info_path.read_text(encoding="utf-8"))
    for key in ("clip_id", "start_step", "end_step"):
        if key not in data:
            raise ValueError(f"{clip_info_path} missing {key!r}")
    return {
        "clip_id": str(data["clip_id"]),
        "start_step": int(data["start_step"]),
        "end_step": int(data["end_step"]),
    }


def export_policy_rollout(
    *,
    policy_root: Path,
    out_path: Path,
    mocap_path: Path | None,
    clip_id: str | None,
    start_step: int | None,
    end_step: int | None,
    rollout_steps: int | None,
    min_steps: int,
    ghost_offset: float,
    always_init_at_clip_start: bool,
    termination_error_threshold: float,
    act_noise: float,
    deterministic: bool,
    device: str,
) -> dict[str, Any]:
    import numpy as np

    from mocapact import observables
    from mocapact.envs import tracking
    from mocapact.sb3 import utils as sb3_utils
    from mocapact.clip_expert import utils as clip_expert_utils

    inferred = _read_clip_info(policy_root)
    resolved_clip_id = clip_id or str(inferred["clip_id"])
    resolved_start = int(inferred["start_step"] if start_step is None else start_step)
    resolved_end = int(inferred["end_step"] if end_step is None else end_step)
    if resolved_end <= resolved_start:
        raise ValueError("--end-step must be greater than --start-step")

    steps = int(rollout_steps or (resolved_end - resolved_start))
    if steps <= 0:
        raise ValueError("--rollout-steps must be positive")

    env_kwargs = clip_expert_utils.make_env_kwargs(
        clip_id=resolved_clip_id,
        mocap_path=None if mocap_path is None else str(mocap_path),
        start_step=resolved_start,
        end_step=resolved_end,
        min_steps=min_steps,
        ghost_offset=ghost_offset,
        always_init_at_clip_start=always_init_at_clip_start,
        termination_error_threshold=termination_error_threshold,
        act_noise=act_noise,
    )
    env = tracking.MocapTrackingGymEnv(**env_kwargs)
    model = sb3_utils.load_policy(
        str(policy_root),
        observables.TIME_INDEX_OBSERVABLES,
        device=device,
    )

    frames: list[dict[str, Any]] = []
    obs = _reset_env(env)
    control_dt = _control_dt(env)
    physics_dt = _physics_dt(env)
    reward = 0.0
    done = False
    info: dict[str, Any] = {}
    try:
        for step_idx in range(steps):
            qpos, qvel = _physics_qpos_qvel(env)
            action, _ = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action).reshape(-1)
            if not np.all(np.isfinite(action_arr)):
                raise ValueError(f"policy produced non-finite action at frame {step_idx}")
            if not done:
                obs, reward, done, info = _step_env(env, action)
            frames.append({
                "t": float(step_idx * control_dt),
                "qpos": qpos,
                "qvel": qvel,
                "action": _array(action_arr),
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
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if not frames:
        raise RuntimeError("policy rollout produced no frames")

    out = {
        "schema_version": 1,
        "source": "mocapact_policy",
        "clip_id": resolved_clip_id,
        "start_step": resolved_start,
        "end_step": resolved_end,
        "control_dt": control_dt,
        "physics_dt": physics_dt,
        "frames": frames,
        "export_metadata": {
            "policy_root": str(policy_root),
            "mocap_path": None if mocap_path is None else str(mocap_path),
            "requested_rollout_steps": steps,
            "min_steps": int(min_steps),
            "ghost_offset": float(ghost_offset),
            "always_init_at_clip_start": bool(always_init_at_clip_start),
            "termination_error_threshold": float(termination_error_threshold),
            "act_noise": float(act_noise),
            "deterministic": bool(deterministic),
            "device": str(device),
        },
    }
    _write_json(out_path, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mocap-path", type=Path)
    parser.add_argument("--clip-id")
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--min-steps", type=int, default=33)
    parser.add_argument("--ghost-offset", type=float, default=1.0)
    parser.add_argument("--termination-error-threshold", type=float, default=0.3)
    parser.add_argument("--act-noise", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="store_true", help="Sample stochastic actions instead of deterministic actions.")
    parser.add_argument("--random-init", action="store_true", help="Use MoCapAct random-start initialization instead of clip start.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        report = {
            "platform": platform.platform(),
            "python": sys.version,
            "dry_run": True,
            "policy_root": str(args.policy_root),
            "out": str(args.out),
            "message": "dry run only; MoCapAct was not imported",
        }
        _write_json(args.out, report)
        print(f"Wrote {args.out}")
        return

    out = export_policy_rollout(
        policy_root=args.policy_root,
        out_path=args.out,
        mocap_path=args.mocap_path,
        clip_id=args.clip_id,
        start_step=args.start_step,
        end_step=args.end_step,
        rollout_steps=args.rollout_steps,
        min_steps=args.min_steps,
        ghost_offset=args.ghost_offset,
        always_init_at_clip_start=not args.random_init,
        termination_error_threshold=args.termination_error_threshold,
        act_noise=args.act_noise,
        deterministic=not args.sample,
        device=args.device,
    )
    print(f"Wrote {args.out}")
    print(
        "frames={frames} qpos_dim={qpos_dim} qvel_dim={qvel_dim} action_dim={action_dim} "
        "terminated={terminated}".format(
            frames=len(out["frames"]),
            qpos_dim=len(out["frames"][0]["qpos"]),
            qvel_dim=len(out["frames"][0]["qvel"]),
            action_dim=len(out["frames"][0]["action"]),
            terminated=any(frame["terminated"] for frame in out["frames"]),
        )
    )


if __name__ == "__main__":
    main()
