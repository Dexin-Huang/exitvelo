"""Tracking controller that replays the retargeted reference trajectory."""

import numpy as np


class TrackingController:
    """Open-loop tracking of a retargeted mocap swing.

    Indexes into the saved joint_targets at each control step,
    converting between mocap fps and control fps.

    Parameters
    ----------
    trajectory_path : path to the retargeted .npz file
    control_fps : env control frequency (default 100 Hz)
    start_frame : mocap frame to begin playback from (default 0).
        This allows starting from the middle of a long motion so the
        swing aligns with the ball flight time.
    clamp_root : if True, get_root_state() returns the mocap root
        position/orientation for the current frame (used by eval
        scripts to clamp the humanoid root and prevent falling).
    """

    def __init__(self, trajectory_path, control_fps=100.0,
                 start_frame=0, clamp_root=True):
        data = np.load(trajectory_path)
        self.joint_targets = data["joint_targets"]   # (T, 17)
        self.root_pos = data["root_pos"]             # (T, 3)
        self.root_quat = data["root_quat"]           # (T, 4)
        self.mocap_fps = float(data["fps"])           # 120
        self.control_fps = control_fps
        self.fps_ratio = self.mocap_fps / self.control_fps
        self.start_frame = start_frame
        self.clamp_root = clamp_root
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0

    @property
    def done(self):
        return self._current_mocap_frame() >= len(self.joint_targets)

    def _current_mocap_frame(self):
        return min(
            self.start_frame + int(self.step_idx * self.control_fps / self.mocap_fps),
            len(self.joint_targets) - 1,
        )

    def get_action(self, obs):
        frame = self._current_mocap_frame()
        action = self.joint_targets[frame].copy()
        self.step_idx += 1
        return action

    def get_root_state(self):
        """Return (pos, quat) of the root for the current mocap frame."""
        frame = self._current_mocap_frame()
        return self.root_pos[frame].copy(), self.root_quat[frame].copy()
