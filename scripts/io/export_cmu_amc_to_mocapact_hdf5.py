"""Export a local CMU AMC clip as a dm_control/MoCapAct HDF5 reference file.

MoCapAct's ``mocap_path`` flag does not read raw ASF/AMC files. It expects the
same fitted-trajectory HDF5 schema used by
``dm_control.locomotion.mocap.loader.HDF5TrajectoryLoader``. This script bridges
that gap for the Exitvelo target swing by:

1. parsing AMC with dm_control's own ``parse_amc.convert`` path;
2. setting a CMU 2020 walker to every converted qpos/qvel frame;
3. recording the reference-pose features that dm_control tracking consumes;
4. writing those features into the HDF5 schema used by MoCapAct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from dm_control import composer, mjcf
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.tasks.reference_pose import tracking, types, utils
from dm_control.locomotion.walkers import cmu_humanoid

from src.motion.cmu_replay import CMUMocapReplay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMC = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124" / "124_07.amc"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "results"
    / "mocapact_custom_hdf5"
    / "CMU_124_07_from_amc_30hz.h5"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "mocapact_custom_hdf5"
    / "CMU_124_07_from_amc_30hz_summary.json"
)

WALKER_ATTRS = {
    "name": "CMU_2020",
    "model": 4,
    "mass": 70.0,
    "end_effector_names": np.array(
        [b"rradius", b"lradius", b"rfoot", b"lfoot"]
    ),
    "appendage_names": np.array(
        [b"rradius", b"lradius", b"rfoot", b"lfoot", b"head"]
    ),
}


def export_clip(
    *,
    amc_path: Path,
    out_path: Path,
    clip_id: str,
    control_dt: float,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and write a MoCapAct-compatible HDF5 file for one AMC clip."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    replay = CMUMocapReplay(amc_path, control_dt=control_dt)
    qpos = np.asarray(replay.qpos_trajectory, dtype=np.float64)
    qvel = np.asarray(replay.qvel_trajectory, dtype=np.float64)
    if max_frames is not None:
        qpos = qpos[:, :max_frames]
        qvel = qvel[:, : max(max_frames - 1, 0)]

    if qpos.shape[0] != 63:
        raise ValueError(f"expected 63 qpos rows from parse_amc, got {qpos.shape}")
    if qvel.shape[0] != 62:
        raise ValueError(f"expected 62 qvel rows from parse_amc, got {qvel.shape}")
    if qpos.shape[1] < 2:
        raise ValueError("need at least two frames to export a reference clip")

    qvel_full = np.zeros((62, qpos.shape[1]), dtype=np.float64)
    qvel_full[:, : qvel.shape[1]] = qvel
    if qvel.shape[1]:
        qvel_full[:, qvel.shape[1] :] = qvel[:, -1:]

    features = _extract_walker_features(qpos=qpos, qvel=qvel_full)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    _write_hdf5(
        out_path=out_path,
        clip_id=clip_id,
        dt=control_dt,
        num_steps=qpos.shape[1],
        features=features,
    )

    validation = validate_hdf5(
        hdf5_path=out_path,
        clip_id=clip_id,
        start_step=0,
        end_step=min(qpos.shape[1], 90),
    )
    window = {
        "source_start_frame": 260,
        "source_end_frame": 360,
        "source_dt": 0.01,
        "target_start_step": int(round(260 * 0.01 / control_dt)),
        "target_end_step": int(round(360 * 0.01 / control_dt)),
    }
    target_window_validation = None
    if window["target_end_step"] < qpos.shape[1]:
        target_window_validation = validate_hdf5(
            hdf5_path=out_path,
            clip_id=clip_id,
            start_step=window["target_start_step"],
            end_step=window["target_end_step"],
        )
    return {
        "status": "written",
        "clip_id": clip_id,
        "amc_path": _rel(amc_path),
        "hdf5_path": _rel(out_path),
        "control_dt": control_dt,
        "num_steps": int(qpos.shape[1]),
        "duration_s": float((qpos.shape[1] - 1) * control_dt),
        "qpos_shape": list(qpos.shape),
        "qvel_shape": list(qvel_full.shape),
        "feature_shapes": {
            key: list(value.shape) for key, value in features.items()
        },
        "exitvelo_100hz_window_to_30hz": window,
        "validation": validation,
        "target_window_validation": target_window_validation,
    }


def validate_hdf5(
    *,
    hdf5_path: Path,
    clip_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    """Validate with dm_control's loader and tracking task reset path."""
    clip_loader = loader.HDF5TrajectoryLoader(str(hdf5_path))
    keys = [
        key.decode("utf-8") if isinstance(key, bytes) else str(key)
        for key in clip_loader.keys()
    ]
    if clip_id not in keys:
        raise AssertionError(f"{clip_id} missing from HDF5 loader keys: {keys}")
    trajectory = clip_loader.get_trajectory(
        clip_id,
        start_step=start_step,
        end_step=end_step,
    )
    as_dict = trajectory.as_dict()

    dataset = types.ClipCollection(
        ids=(clip_id,),
        start_steps=(start_step,),
        end_steps=(end_step,),
        weights=(1.0,),
    )
    task = tracking.MultiClipMocapTracking(
        walker=cmu_humanoid.CMUHumanoidPositionControlledV2020,
        arena=composer.Arena(),
        ref_path=str(hdf5_path),
        ref_steps=(0, 1, 2),
        dataset=dataset,
        always_init_at_clip_start=True,
        min_steps=2,
        disable_props=True,
        termination_error_threshold=0.3,
    )
    env = composer.Environment(task=task)
    timestep = env.reset()
    observation_keys = sorted(timestep.observation.keys())
    return {
        "loader_keys": keys,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "trajectory_num_steps": int(trajectory.num_steps),
        "trajectory_dt": float(trajectory.dt),
        "dict_shapes": {
            key: list(value.shape) for key, value in as_dict.items()
        },
        "tracking_reset_ok": True,
        "tracking_initial_termination_error": float(task._termination_error),
        "tracking_observation_keys": observation_keys,
    }


def _extract_walker_features(*, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, np.ndarray]:
    arena = composer.Arena()
    walker = utils.add_walker(cmu_humanoid.CMUHumanoidPositionControlledV2020, arena)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    rows: dict[str, list[np.ndarray]] = {
        "position": [],
        "quaternion": [],
        "joints": [],
        "center_of_mass": [],
        "end_effectors": [],
        "velocity": [],
        "angular_velocity": [],
        "joints_velocity": [],
        "appendages": [],
        "body_positions": [],
        "body_quaternions": [],
    }
    for frame in range(qpos.shape[1]):
        utils.set_walker(physics, walker, qpos[:, frame], qvel[:, frame])
        physics.forward()
        feature = utils.get_features(physics, walker)
        for key in rows:
            rows[key].append(np.asarray(feature[key], dtype=np.float64).copy())

    stacked: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        arr = np.asarray(values, dtype=np.float64)
        if key in {"end_effectors", "appendages", "body_positions", "body_quaternions"}:
            arr = arr.reshape(arr.shape[0], -1)
        stacked[key] = arr
    return stacked


def _write_hdf5(
    *,
    out_path: Path,
    clip_id: str,
    dt: float,
    num_steps: int,
    features: dict[str, np.ndarray],
) -> None:
    with h5py.File(out_path, "w") as h5:
        clip = h5.create_group(clip_id)
        clip.attrs["year"] = 0
        clip.attrs["month"] = 0
        clip.attrs["day"] = 0
        clip.attrs["dt"] = float(dt)
        clip.attrs["num_steps"] = int(num_steps)

        walkers = clip.create_group("walkers")
        walker = walkers.create_group("walker_0")
        for key, value in WALKER_ATTRS.items():
            walker.attrs[key] = value

        scaling = walker.create_group("scaling")
        subtree = scaling.create_group("subtree_0")
        subtree.attrs["body_name"] = "root"
        subtree.attrs["parent_length"] = 0.0
        subtree.attrs["size_factor"] = 1.2
        walker.create_group("markers")

        for key, arr in features.items():
            if arr.shape[0] != num_steps:
                raise ValueError(f"{key} has {arr.shape[0]} rows, expected {num_steps}")
            walker.create_dataset(key, data=arr.T, compression="gzip")

        clip.create_group("props")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    args = _parse_args()
    summary = export_clip(
        amc_path=args.amc,
        out_path=args.out,
        clip_id=args.clip_id,
        control_dt=args.control_dt,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "Wrote MoCapAct HDF5 "
        f"{summary['hdf5_path']} ({summary['num_steps']} steps, "
        f"dt={summary['control_dt']})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amc", type=Path, default=DEFAULT_AMC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--clip-id", default="CMU_124_07")
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.03,
        help="Reference clip timestep. 0.03 matches dm_control/MoCapAct CMU HDF5.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
