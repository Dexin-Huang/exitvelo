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


DEFAULT_CLIPS = ["CMU_124_07", "CMU_124_08", "CMU_016_22"]


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


def _make_env(clip_id: str, start_step: int | None, end_step: int | None, ref_steps: tuple[int, ...]):
    from dm_control.locomotion.tasks.reference_pose import types
    from mocapact.envs import tracking

    kwargs: dict[str, Any] = {"ids": [clip_id]}
    if start_step is not None:
        kwargs["start_steps"] = [int(start_step)]
    if end_step is not None:
        kwargs["end_steps"] = [int(end_step)]
    dataset = types.ClipCollection(**kwargs)
    return tracking.MocapTrackingGymEnv(dataset=dataset, ref_steps=ref_steps)


def _probe_clip(
    *,
    clip_id: str,
    start_step: int | None,
    end_step: int | None,
    rollout_steps: int,
    zero_action: bool,
    ref_steps: tuple[int, ...],
) -> dict:
    import numpy as np

    result: dict[str, Any] = {
        "clip_id": clip_id,
        "start_step": start_step,
        "end_step": end_step,
        "loaded": False,
        "stepped": False,
        "error": None,
    }
    try:
        env = _make_env(clip_id, start_step, end_step, ref_steps)
        result["loaded"] = True
        result["observation_space"] = _space_summary(env.observation_space)
        result["action_space"] = _space_summary(env.action_space)
        obs = env.reset()
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
            _, reward, done, info = env.step(action)
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
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/runpod_mocapact_probe"))
    parser.add_argument("--clip-id", action="append", dest="clip_ids", default=None)
    parser.add_argument("--start-step", type=int, default=260)
    parser.add_argument("--end-step", type=int, default=360)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clip_ids = args.clip_ids or DEFAULT_CLIPS
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "dry_run": bool(args.dry_run),
        "clip_ids": clip_ids,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "rollout_steps": args.rollout_steps,
        "zero_action": bool(args.zero_action),
        "clips": [],
        "best_rollout_path": None,
    }

    if args.dry_run:
        report["message"] = "dry run only; MoCapAct was not imported"
        _write_json(args.out_dir / "mocapact_probe_report.json", report)
        print(f"Wrote {args.out_dir / 'mocapact_probe_report.json'}")
        return

    for clip_id in clip_ids:
        result = _probe_clip(
            clip_id=clip_id,
            start_step=args.start_step,
            end_step=args.end_step,
            rollout_steps=args.rollout_steps,
            zero_action=args.zero_action,
            ref_steps=(0,),
        )
        rollout = result.pop("rollout", None)
        if rollout is not None and result.get("loaded"):
            rollout_path = args.out_dir / f"{clip_id}_rollout.json"
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
