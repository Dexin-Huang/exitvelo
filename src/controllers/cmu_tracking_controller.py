"""Tracking controller for the CMU humanoid batting environment.

Replays CMU mocap data with optional named-residual perturbations
(swing_timing_s / hip_fire_rad / uppercut_rad). Bat is on the left
hand with a bat_grip hinge joint inserted at scene action index 41.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.motion.cmu_replay import CMUMocapReplay
from src.controllers.swing_residuals import (
    SwingResiduals,
    apply_pelvis_trunk_yaw,
    apply_uppercut_pitch,
)

_BAT_GRIP_SCENE_IDX = 41  # bat_grip joint inserted after lhandrx in action array


class CMUTrackingController:
    def __init__(
        self,
        amc_path: str | Path = "data/raw/cmu_subject_124/124_07.amc",
        start_frame: int = 0,
        control_dt: float = 0.01,
        residuals: SwingResiduals | None = None,
        **kwargs,  # accept and ignore extra kwargs like env=
    ):
        self.replay = CMUMocapReplay(str(amc_path), control_dt=control_dt)
        self.start_frame = start_frame
        self.replay.reset(start_frame)
        residuals = (residuals or SwingResiduals()).clip()
        if abs(residuals.barrel_roll_rad) > 1e-9 or abs(residuals.plate_reach_m) > 1e-9:
            raise ValueError(
                "barrel_roll_rad and plate_reach_m are not implemented yet "
                "(deferred to a follow-up). Pass 0.0 for those scalars."
            )
        self.residuals = residuals
        self._step_idx = start_frame

    def reset(self):
        self.replay.reset(self.start_frame)
        self._step_idx = self.start_frame

    def set_residuals(self, residuals: SwingResiduals) -> None:
        """Apply a new (open-loop) residual vector. Must be called BEFORE
        the swing starts; per-step changes are ignored for Tier 2A.

        Raises ValueError if the deferred (barrel_roll / plate_reach)
        scalars are non-zero — those dimensions are not yet wired and
        silently ignoring them would mislead the optimizer.
        """
        if abs(residuals.barrel_roll_rad) > 1e-9:
            raise ValueError(
                "barrel_roll_rad is not implemented yet (deferred). "
                "Pass 0.0 for this scalar in Tier 2."
            )
        if abs(residuals.plate_reach_m) > 1e-9:
            raise ValueError(
                "plate_reach_m is not implemented yet (deferred). "
                "Pass 0.0 for this scalar in Tier 2."
            )
        self.residuals = residuals.clip()

    @property
    def done(self) -> bool:
        return self._step_idx >= self.replay.n_frames

    @property
    def swing_phase(self) -> float:
        """Normalized progress through the swing: 0 at start_frame, 1 at end."""
        total = max(1, self.replay.n_frames - self.start_frame)
        return float(np.clip((self._step_idx - self.start_frame) / total, 0.0, 1.0))

    def get_action(self, obs: np.ndarray | None = None) -> np.ndarray:
        """Return the 56-dim action (joint targets) for the current step,
        with residuals applied."""
        # Time-shifted mocap lookup. swing_timing_s shifts when in the swing
        # we are reading the reference pose.
        t = self._step_idx * self.replay.control_dt + self.residuals.swing_timing_s
        # Clamp to the available range so we never look up past the clip
        t_max = (self.replay.n_frames - 1) * self.replay.control_dt
        t = float(np.clip(t, 0.0, t_max))

        q_ref = self.replay.qpos_at(t)
        q_ref = apply_pelvis_trunk_yaw(q_ref, self.residuals.hip_fire_rad)
        q_ref = apply_uppercut_pitch(q_ref, self.residuals.uppercut_rad)

        mocap_joints = q_ref[7:].copy()  # drop the 7-dim root free joint

        # Insert bat_grip=0 at scene index 41 (existing scene contract).
        scene_action = np.zeros(57, dtype=np.float64)
        scene_action[:_BAT_GRIP_SCENE_IDX] = mocap_joints[:_BAT_GRIP_SCENE_IDX]
        scene_action[_BAT_GRIP_SCENE_IDX] = 0.0
        scene_action[_BAT_GRIP_SCENE_IDX + 1:57] = mocap_joints[_BAT_GRIP_SCENE_IDX:56]

        self._step_idx += 1
        return scene_action[:56].astype(np.float32)

    def get_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Root pos / quat at the current step (no residual transforms)."""
        t = self._step_idx * self.replay.control_dt
        t_max = (self.replay.n_frames - 1) * self.replay.control_dt
        t = float(np.clip(t, 0.0, t_max))
        q = self.replay.qpos_at(t)
        return q[:3].copy(), q[3:7].copy()
