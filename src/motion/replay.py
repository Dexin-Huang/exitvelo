"""Open-loop replay of retargeted mocap trajectory as a policy.

Usage
-----
>>> replay = MocapReplay("data/processed/swing_124_07.npz")
>>> action = replay.get_action()  # returns (17,) array of joint targets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MocapReplay:
    """Open-loop replay of retargeted mocap trajectory as a policy."""

    def __init__(self, trajectory_path: str | Path, control_fps: float = 100.0):
        """Load a retargeted trajectory.

        Parameters
        ----------
        trajectory_path : path to .npz file produced by retarget.save_retargeted_motion
        control_fps : control loop frequency of the MuJoCo simulation
        """
        data = np.load(trajectory_path)
        self.joint_targets: np.ndarray = data["joint_targets"]  # (T, 17)
        self.root_pos: np.ndarray = data["root_pos"]            # (T, 3)
        self.root_quat: np.ndarray = data["root_quat"]          # (T, 4)
        self.mocap_fps: float = float(data["fps"])               # 120
        self.control_fps: float = control_fps
        self.fps_ratio: float = self.mocap_fps / self.control_fps
        self.num_frames: int = len(self.joint_targets)
        self.step_idx: int = 0

    def reset(self):
        """Reset playback to the first frame."""
        self.step_idx = 0

    @property
    def done(self) -> bool:
        """True when all mocap frames have been consumed."""
        frame = int(self.step_idx * self.control_fps / self.mocap_fps)
        return frame >= self.num_frames

    def _current_frame(self) -> int:
        return min(
            int(self.step_idx / self.fps_ratio),
            self.num_frames - 1,
        )

    def get_action(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Return the joint target angles for the current control step.

        Parameters
        ----------
        obs : ignored (signature kept for compatibility with RL policy APIs)

        Returns
        -------
        action : (17,) array of joint angle targets (radians)
        """
        frame = self._current_frame()
        action = self.joint_targets[frame].copy()
        self.step_idx += 1
        return action

    def get_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion) of the root for the current frame.

        Returns
        -------
        pos : (3,) root position in MuJoCo frame
        quat : (4,) quaternion [w, x, y, z]
        """
        frame = self._current_frame()
        return self.root_pos[frame].copy(), self.root_quat[frame].copy()

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:
        return (
            f"MocapReplay(frames={self.num_frames}, "
            f"mocap_fps={self.mocap_fps}, "
            f"control_fps={self.control_fps})"
        )
