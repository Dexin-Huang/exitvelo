"""Open-loop replay of CMU mocap data using dm_control's parse_amc.

This uses dm_control's parse_amc module which handles the exact coordinate
transforms (Y-up to Z-up via 90-degree X rotation) and scaling (factor 0.056444)
needed for the CMU humanoid model.

Usage
-----
>>> replay = CMUMocapReplay("data/raw/cmu_subject_124/124_07.amc")
>>> action = replay.get_action()  # returns (56,) array of joint targets (radians)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# dm_control imports for AMC parsing
from dm_control import mujoco as dm_mujoco
from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

# Module-level caches so the dm_control physics and parsed AMC data are not
# reloaded on every instantiation.
_PHYSICS_CACHE = None
_AMC_CACHE = {}


def _get_physics():
    """Return a cached dm_control CMU humanoid physics object."""
    global _PHYSICS_CACHE
    if _PHYSICS_CACHE is None:
        xml_string, assets = humanoid_CMU.get_model_and_assets()
        _PHYSICS_CACHE = dm_mujoco.Physics.from_xml_string(xml_string, assets)
    return _PHYSICS_CACHE


class CMUMocapReplay:
    """Open-loop replay of CMU mocap trajectory for the CMU humanoid.

    Uses dm_control's parse_amc.convert() which properly handles:
    - Y-up to Z-up coordinate transform (90-degree X rotation)
    - CMU unit scaling (factor 0.056444)
    - Euler to quaternion conversion for root orientation
    - Joint angle mapping from AMC format to MuJoCo qpos format
    - Resampling to the desired control timestep
    """

    def __init__(
        self,
        amc_path: str | Path,
        control_dt: float = 0.01,  # 100 Hz control (frame_skip=5 * timestep=0.002)
    ):
        """Load and convert an AMC file to qpos/qvel trajectory.

        Parameters
        ----------
        amc_path : path to .amc file from CMU mocap database
        control_dt : control loop period in seconds
        """
        self.amc_path = Path(amc_path)

        # Use cached physics and AMC parse results
        physics = _get_physics()

        cache_key = (str(self.amc_path), control_dt)
        if cache_key in _AMC_CACHE:
            converted = _AMC_CACHE[cache_key]
        else:
            converted = parse_amc.convert(str(self.amc_path), physics, control_dt)
            _AMC_CACHE[cache_key] = converted

        # converted.qpos shape: (nq, n_frames) = (63, T)
        # converted.qvel shape: (nv, n_frames-1) = (62, T-1)
        # converted.time shape: (T,)
        self.qpos_trajectory = converted.qpos      # (63, T)
        self.qvel_trajectory = converted.qvel      # (62, T-1)
        self.time = converted.time                  # (T,)

        self.n_frames = self.qpos_trajectory.shape[1]
        self.n_qpos = self.qpos_trajectory.shape[0]   # 63
        self.n_qvel = self.qvel_trajectory.shape[0]    # 62
        self.control_dt = control_dt

        # Playback state
        self.step_idx = 0

    def reset(self, start_frame: int = 0):
        """Reset playback to a specific frame."""
        self.step_idx = start_frame

    @property
    def done(self) -> bool:
        """True when all frames have been consumed."""
        return self.step_idx >= self.n_frames

    @property
    def num_frames(self) -> int:
        return self.n_frames

    def get_qpos(self, frame: int | None = None) -> np.ndarray:
        """Get the full qpos (63-dim) at a specific frame.

        Returns
        -------
        qpos : (63,) array [root_pos(3) + root_quat(4) + joint_angles(56)]
        """
        if frame is None:
            frame = min(self.step_idx, self.n_frames - 1)
        frame = min(frame, self.n_frames - 1)
        return self.qpos_trajectory[:, frame].copy()

    def qpos_at(self, t_seconds: float) -> np.ndarray:
        """Pure lookup: qpos at time `t_seconds`, linearly interpolated
        between the two surrounding frames. Does not mutate `step_idx`.

        Used by the named-residual controller to apply `swing_timing_s`
        offsets without disturbing the underlying playback cursor.
        """
        # Clamp t_seconds to the valid range BEFORE computing alpha so
        # negative offsets don't blend with frame 1.
        t_max = (self.n_frames - 1) * self.control_dt
        t_seconds = float(np.clip(t_seconds, 0.0, t_max))
        frame_f = t_seconds / self.control_dt
        frame_lo = int(np.floor(frame_f))
        alpha = frame_f - frame_lo
        frame_lo = max(0, min(frame_lo, self.n_frames - 1))
        frame_hi = max(0, min(frame_lo + 1, self.n_frames - 1))
        q_lo = self.qpos_trajectory[:, frame_lo]
        q_hi = self.qpos_trajectory[:, frame_hi]
        return q_lo * (1.0 - alpha) + q_hi * alpha

    def get_qvel(self, frame: int | None = None) -> np.ndarray:
        """Get the full qvel (62-dim) at a specific frame.

        Returns
        -------
        qvel : (62,) array [root_lin_vel(3) + root_ang_vel(3) + joint_vels(56)]
        """
        if frame is None:
            frame = min(self.step_idx, self.n_frames - 2)
        frame = min(frame, self.qvel_trajectory.shape[1] - 1)
        return self.qvel_trajectory[:, frame].copy()

    def get_action(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Return the joint target angles (56-dim) for PD control.

        This extracts just the hinge joint angles from qpos
        (skipping the 7-dim root free joint).

        Parameters
        ----------
        obs : ignored (signature kept for compatibility with RL policy APIs)

        Returns
        -------
        action : (56,) array of joint angle targets (radians)
        """
        qpos = self.get_qpos()
        action = qpos[7:].copy()  # skip root free joint (3 pos + 4 quat)
        self.step_idx += 1
        return action

    def get_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion) of the root for the current frame.

        Returns
        -------
        pos : (3,) root position
        quat : (4,) quaternion [w, x, y, z]
        """
        qpos = self.get_qpos()
        return qpos[:3].copy(), qpos[3:7].copy()

    def get_full_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos, qvel) for the current frame.

        Returns
        -------
        qpos : (63,) full qpos
        qvel : (62,) full qvel
        """
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        return qpos, qvel

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        return (
            f"CMUMocapReplay(file={self.amc_path.name}, "
            f"frames={self.n_frames}, "
            f"duration={self.time[-1]:.2f}s, "
            f"control_dt={self.control_dt})"
        )
