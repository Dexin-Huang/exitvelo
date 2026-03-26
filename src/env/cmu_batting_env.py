"""MuJoCo-based batting environment using the dm_control CMU humanoid (56 joints).

This replaces the old 17-joint gymnasium humanoid batting env with one that uses
the dm_control CMU humanoid skeleton, which has a 1:1 joint mapping with CMU
mocap data. The bat is attached to the left hand via a hinge joint (bat_grip).
No retargeting needed.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.env.contacts import detect_bat_ball_contact

_XML_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "mujoco" / "cmu_batting_scene.xml"

# Joint index ranges for the CMU humanoid within qpos/qvel
_QPOS_JOINT_START = 7     # after root free joint (3 pos + 4 quat)
_QPOS_JOINT_END = 64      # 57 hinge joints (56 humanoid + 1 bat_grip)
_QVEL_JOINT_START = 6     # after root free joint (3 lin + 3 ang)
_QVEL_JOINT_END = 63      # 57 hinge dofs

# Ball free-joint indices (after all hinge joints)
_BALL_QPOS_START = 64     # 3 pos + 4 quat
_BALL_QVEL_START = 63     # 3 lin + 3 ang

_N_JOINTS = 56

# Sensor data indices (from the XML sensor layout)
_SENSOR_BAT_TIP_POS = slice(16, 19)
_SENSOR_BAT_TIP_VEL = slice(19, 22)
_SENSOR_BALL_POS = slice(22, 25)
_SENSOR_BALL_VEL = slice(25, 28)


class CMUBattingEnv(gym.Env):
    """Gymnasium environment for CMU humanoid baseball batting.

    Uses the dm_control CMU humanoid skeleton (56 actuated joints, 64 qpos)
    with a bat attached to the left hand via a hinge joint (bat_grip),
    and a baseball as a free body.

    Observation (140-dim):
        joint_pos (56) + joint_vel (56) + root_pos (3) + root_quat (4) +
        root_lin_vel (3) + root_ang_vel (3) + bat_tip_pos (3) + bat_tip_vel (3) +
        ball_pos (3) + ball_vel (3) + ball_rel (3)

    Action (56-dim):
        Target joint positions fed through PD control.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # ── Scaled MLB pitch defaults ──
    # Scale factor: MuJoCo humanoid (1.4m) / real MLB player (1.83m) = 0.765
    MLB_PITCH_X = -12.82
    MLB_PITCH_Y = 0.0
    MLB_PITCH_HEIGHT = 1.40
    MLB_PITCH_SPEED = 33.24

    def __init__(
        self,
        render_mode=None,
        frame_skip: int = 5,
        max_steps: int = 500,
        pitch_speed: float = MLB_PITCH_SPEED,
        pitch_height: float = MLB_PITCH_HEIGHT,
        pitch_x: float = MLB_PITCH_X,
        pitch_y: float = MLB_PITCH_Y,
        pitch_velocity=None,
    ):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0

        # Pitch configuration
        self.pitch_speed = pitch_speed
        self.pitch_height = pitch_height
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.pitch_velocity = (
            np.array(pitch_velocity, dtype=np.float64)
            if pitch_velocity is not None
            else None
        )

        # Pre-compute actuator-to-joint index mapping (vectorised PD control)
        jnt_ids = self.model.actuator_trnid[:, 0]
        self._act_qpos_idx = self.model.jnt_qposadr[jnt_ids]  # (56,)
        self._act_qvel_idx = self.model.jnt_dofadr[jnt_ids]    # (56,)

        # PD gains for the 56 CMU joints
        # Group joints by type and assign appropriate gains
        self.kp = self._build_kp()
        self.kd = 0.1 * self.kp

        # Build joint limits for action space (in radians)
        joint_ranges = np.zeros((_N_JOINTS, 2))
        for i in range(_N_JOINTS):
            jnt_id = self.model.actuator_trnid[i, 0]
            joint_ranges[i] = self.model.jnt_range[jnt_id]

        self.joint_lo = joint_ranges[:, 0].astype(np.float32)
        self.joint_hi = joint_ranges[:, 1].astype(np.float32)

        self.action_space = spaces.Box(
            low=self.joint_lo, high=self.joint_hi, dtype=np.float32
        )

        # Observation space (140 dims)
        obs_dim = 140
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Renderer
        self._renderer = None
        if self.render_mode is not None:
            self._init_renderer()

        # Contact tracking
        self._contact_made = False

        # Cache contact geom IDs for fast lookup
        self._bat_geom_ids = set()
        for name in ['bat_barrel', 'bat_handle', 'bat_taper', 'bat_end', 'bat_knob']:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._bat_geom_ids.add(gid)
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')

    def _build_kp(self) -> np.ndarray:
        """Build per-joint PD proportional gains.

        Groups joints by body region and assigns gains accordingly.
        The joint order in the CMU humanoid XML (alphabetical by actuator name) is:
          headrx, headry, headrz,
          lclaviclery, lclaviclerz,
          lfemurrx, lfemurry, lfemurrz,
          lfingersrx,
          lfootrx, lfootrz,
          lhandrx, lhandrz,
          lhumerusrx, lhumerusry, lhumerusrz,
          lowerbackrx, lowerbackry, lowerbackrz,
          lowerneckrx, lowerneckry, lowerneckrz,
          lradiusrx,
          lthumbrx, lthumbrz,
          ltibiarx,
          ltoesrx,
          lwristry,
          rclaviclery, rclaviclerz,
          rfemurrx, rfemurry, rfemurrz,
          rfingersrx,
          rfootrx, rfootrz,
          rhandrx, rhandrz,
          rhumerusrx, rhumerusry, rhumerusrz,
          rradiusrx,
          rthumbrx, rthumbrz,
          rtibiarx,
          rtoesrx,
          rwristry,
          thoraxrx, thoraxry, thoraxrz,
          upperbackrx, upperbackry, upperbackrz,
          upperneckrx, upperneckry, upperneckrz

        We build kp in actuator order so it aligns with the vectorised PD
        control in step(), which indexes qpos/qvel via _act_qpos_idx/_act_qvel_idx
        (both derived from actuator_trnid).
        """
        kp = np.zeros(_N_JOINTS)

        for i in range(_N_JOINTS):
            jnt_id = self.model.actuator_trnid[i, 0]
            jnt_name = self.model.joint(jnt_id).name

            # Assign gains based on joint type
            if 'femur' in jnt_name:
                kp[i] = 100.0  # hip joints - high gain
            elif 'tibia' in jnt_name:
                kp[i] = 80.0   # knee
            elif 'foot' in jnt_name or 'toes' in jnt_name:
                kp[i] = 40.0   # ankle/toes
            elif 'lowerback' in jnt_name or 'upperback' in jnt_name:
                kp[i] = 100.0  # spine - high gain for stability
            elif 'thorax' in jnt_name:
                kp[i] = 80.0   # thorax
            elif 'neck' in jnt_name or 'head' in jnt_name:
                kp[i] = 20.0   # neck/head - low gain
            elif 'clavicle' in jnt_name:
                kp[i] = 40.0   # clavicle
            elif 'humerus' in jnt_name:
                kp[i] = 50.0   # shoulder
            elif 'radius' in jnt_name:
                kp[i] = 40.0   # elbow
            elif 'wrist' in jnt_name:
                kp[i] = 30.0   # wrist
            elif 'hand' in jnt_name:
                kp[i] = 20.0   # hand
            elif 'finger' in jnt_name or 'thumb' in jnt_name:
                kp[i] = 10.0   # fingers
            else:
                kp[i] = 30.0   # default

        return kp

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------
    def _init_renderer(self):
        if self.render_mode in ("human", "rgb_array"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def render(self):
        if self._renderer is None:
            return None
        # Use the 'track' camera
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if cam_id >= 0:
            self._renderer.update_scene(self.data, camera=cam_id)
        else:
            self._renderer.update_scene(self.data)
        img = self._renderer.render()
        return img

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Humanoid neutral stance from default qpos0
        self.data.qpos[:_QPOS_JOINT_END] = self.model.qpos0[:_QPOS_JOINT_END]
        self.data.qvel[:_QVEL_JOINT_END] = 0.0

        # Launch ball
        self._launch_ball()

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self._contact_made = False

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.joint_lo, self.joint_hi)

        # Vectorised PD control using pre-computed index arrays
        q = self.data.qpos[self._act_qpos_idx]
        qdot = self.data.qvel[self._act_qvel_idx]
        ctrl = self.kp * (action - q) - self.kd * qdot

        # Clip to actuator control range
        ctrl_range = self.model.actuator_ctrlrange
        ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Check contact (pass cached geom IDs to avoid per-step name lookups)
        contact, contact_info = detect_bat_ball_contact(
            self.model, self.data,
            bat_geom_ids=self._bat_geom_ids,
            ball_geom_id=self._ball_geom_id,
        )
        if contact:
            self._contact_made = True

        # Ball velocity post-contact
        ball_vel = self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3].copy()
        ball_speed_post = float(np.linalg.norm(ball_vel)) if contact else 0.0

        # Ball position
        ball_pos = self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3]

        # Torso height (z-component of root position)
        torso_z = self.data.qpos[2]

        # Termination checks
        terminated = False
        truncated = False
        termination_reason = ""

        if contact:
            terminated = True
            termination_reason = "contact"
        elif ball_pos[0] > 2.0:
            terminated = True
            termination_reason = "ball_past_batter"
        elif torso_z < 0.5:
            terminated = True
            termination_reason = "humanoid_fell"
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "timeout"

        obs = self._get_obs()
        reward = 0.0  # reward shaping deferred to training code

        info = {
            "contact": contact,
            "ball_speed_post": ball_speed_post,
            "termination_reason": termination_reason,
        }
        if contact_info is not None:
            info["contact_info"] = contact_info

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        # Read only the 56 actuated joints (skips bat_grip which has no actuator)
        joint_pos = self.data.qpos[self._act_qpos_idx].copy()
        joint_vel = self.data.qvel[self._act_qvel_idx].copy()
        root_pos = self.data.qpos[0:3]
        root_quat = self.data.qpos[3:7]
        root_lin_vel = self.data.qvel[0:3]
        root_ang_vel = self.data.qvel[3:6]

        # Sensor data
        bat_tip_pos = self.data.sensordata[_SENSOR_BAT_TIP_POS]
        bat_tip_vel = self.data.sensordata[_SENSOR_BAT_TIP_VEL]
        ball_pos = self.data.sensordata[_SENSOR_BALL_POS]
        ball_vel = self.data.sensordata[_SENSOR_BALL_VEL]

        # Relative position: ball - bat_tip
        ball_rel = ball_pos - bat_tip_pos

        obs = np.concatenate([
            joint_pos,      # 56
            joint_vel,      # 56
            root_pos,       # 3
            root_quat,      # 4
            root_lin_vel,   # 3
            root_ang_vel,   # 3
            bat_tip_pos,    # 3
            bat_tip_vel,    # 3
            ball_pos,       # 3
            ball_vel,       # 3
            ball_rel,       # 3
        ])  # total = 140
        return obs

    # ------------------------------------------------------------------
    # Ball launch
    # ------------------------------------------------------------------
    def _launch_ball(self):
        # Position: in front of batter
        self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3] = [
            self.pitch_x, self.pitch_y, self.pitch_height
        ]
        # Quaternion: identity
        self.data.qpos[_BALL_QPOS_START + 3:_BALL_QPOS_START + 7] = [1, 0, 0, 0]

        # Velocity
        if self.pitch_velocity is not None:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = self.pitch_velocity
        else:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = [
                self.pitch_speed, 0.0, 0.0
            ]
        # No angular velocity
        self.data.qvel[_BALL_QVEL_START + 3:_BALL_QVEL_START + 6] = [0, 0, 0]

    # ------------------------------------------------------------------
    # Utility: set qpos directly (for mocap replay)
    # ------------------------------------------------------------------
    def set_humanoid_qpos(self, qpos: np.ndarray):
        """Set humanoid qpos (root + hinge joints).

        Accepts 63-dim (7 root + 56 joints) or 64-dim (7 root + 57 joints
        including bat_grip), which maps directly to qpos[0:n].
        Useful for mocap replay.
        """
        n = len(qpos)
        assert n in (63, 64), f"Expected 63 or 64 dim qpos, got {n}"
        self.data.qpos[:n] = qpos

    def set_humanoid_qvel(self, qvel: np.ndarray):
        """Set humanoid qvel (root + hinge DOFs).

        Accepts 62-dim (6 root + 56 joints) or 63-dim (6 root + 57 DOFs
        including bat_grip), which maps directly to qvel[0:n].
        """
        n = len(qvel)
        assert n in (62, 63), f"Expected 62 or 63 dim qvel, got {n}"
        self.data.qvel[:n] = qvel

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
