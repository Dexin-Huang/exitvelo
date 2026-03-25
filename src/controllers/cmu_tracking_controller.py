"""Tracking controller for the CMU humanoid batting environment.

Replays CMU mocap data. Bat is on the right hand with a bat_grip hinge joint
inserted at scene action index 53.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.motion.cmu_replay import CMUMocapReplay

_BAT_GRIP_SCENE_IDX = 53  # bat_grip joint inserted after rhandrx in action array


class CMUTrackingController:
    def __init__(
        self,
        amc_path: str | Path = "data/raw/cmu_subject_124/124_07.amc",
        start_frame: int = 0,
        control_dt: float = 0.01,
        **kwargs,  # accept and ignore extra kwargs like env=
    ):
        self.replay = CMUMocapReplay(str(amc_path), control_dt=control_dt)
        self.start_frame = start_frame
        self.replay.reset(start_frame)

    def reset(self):
        self.replay.reset(self.start_frame)

    @property
    def done(self) -> bool:
        return self.replay.done

    def get_action(self, obs: np.ndarray | None = None) -> np.ndarray:
        mocap_joints = self.replay.get_action(obs)  # (56,)
        # Insert bat_grip=0 at index 53
        scene_action = np.zeros(57, dtype=np.float64)
        scene_action[:_BAT_GRIP_SCENE_IDX] = mocap_joints[:_BAT_GRIP_SCENE_IDX]
        scene_action[_BAT_GRIP_SCENE_IDX] = 0.0
        scene_action[_BAT_GRIP_SCENE_IDX + 1:57] = mocap_joints[_BAT_GRIP_SCENE_IDX:56]
        return scene_action[:56].astype(np.float32)

    def get_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.replay.get_root_state()
