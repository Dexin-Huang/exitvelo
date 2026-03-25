"""MuJoCo-based batting environment using the dm_control CMU humanoid (56 joints).

This replaces the old 17-joint gymnasium humanoid batting env with one that uses
the dm_control CMU humanoid skeleton, which has a 1:1 joint mapping with CMU
mocap data. No retargeting needed.
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

    Uses the dm_control CMU humanoid skeleton (56 actuated joints, 63 qpos)
    with a bat as a free body positioned between both hands each frame,
    and a baseball as a free body.

    Observation (130-dim):
        joint_pos (56) + joint_vel (56) + root_pos (3) + root_quat (4) +
        root_lin_vel (3) + root_ang_vel (3) + bat_tip_pos (3) + bat_tip_vel (3) +
        ball_pos (3) + ball_vel (3) + ball_rel (3)
        Total = 56 + 56 + 3 + 4 + 3 + 3 + 3 + 3 + 3 + 3 + 3 = 140
        (Wait: 56+56+3+4+3+3+3+3+3+3+3 = 140)

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
    MLB_PITCH_VZ = -0.002

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

        # PD gains for the 56 CMU joints
        # Group joints by type and assign appropriate gains
        self.kp = self._build_kp()
        self.kd = 0.1 * self.kp

        # Build joint limits for action space (in radians)
        joint_ranges = np.zeros((_N_JOINTS, 2))
        for i in range(_N_JOINTS):
            # hinge joints start at joint index 1 (0 is root free)
            jnt_id = i + 1  # skip root free joint
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

        But we need kp for qpos-ordered joints (not actuator order).
        The qpos order follows the body tree (joint order in XML).
        """
        # Get actuator names to figure out the mapping
        # Actually, kp is indexed by qpos hinge joint order, not actuator order.
        # The qpos hinge joint order is determined by the body tree in the XML.
        kp = np.zeros(_N_JOINTS)

        for i in range(_N_JOINTS):
            jnt_id = i + 1  # skip root free joint
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

        # PD control: torque = kp * (target - q) - kd * qdot
        q = self.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END]
        qdot = self.data.qvel[_QVEL_JOINT_START:_QVEL_JOINT_END]
        torque = self.kp * (action - q) - self.kd * qdot

        # Map torques to actuators
        # The actuator order differs from the qpos joint order.
        # We need to map: for each actuator, find its corresponding joint,
        # then look up the PD torque for that joint.
        ctrl = np.zeros(self.model.nu)
        for act_idx in range(self.model.nu):
            # Get the joint that this actuator drives
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            # The qpos hinge joint index (0-indexed from first hinge)
            hinge_idx = jnt_id - 1  # subtract 1 for the root free joint
            if 0 <= hinge_idx < _N_JOINTS:
                ctrl[act_idx] = torque[hinge_idx]

        # Clip to actuator control range
        ctrl_range = self.model.actuator_ctrlrange
        ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # Check contact
        contact, contact_info = detect_bat_ball_contact(self.model, self.data)
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
        joint_pos = self.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END].copy()  # 56
        joint_vel = self.data.qvel[_QVEL_JOINT_START:_QVEL_JOINT_END].copy()   # 56
        root_pos = self.data.qpos[0:3].copy()                                    # 3
        root_quat = self.data.qpos[3:7].copy()                                   # 4
        root_lin_vel = self.data.qvel[0:3].copy()                                 # 3
        root_ang_vel = self.data.qvel[3:6].copy()                                 # 3

        # Sensor data
        bat_tip_pos = self.data.sensordata[_SENSOR_BAT_TIP_POS].copy()    # 3
        bat_tip_vel = self.data.sensordata[_SENSOR_BAT_TIP_VEL].copy()    # 3
        ball_pos = self.data.sensordata[_SENSOR_BALL_POS].copy()            # 3
        ball_vel = self.data.sensordata[_SENSOR_BALL_VEL].copy()            # 3

        # Relative position: ball - bat_tip
        ball_rel = ball_pos - bat_tip_pos                                     # 3

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
        """Set humanoid qpos (root + 56 hinge joints).

        Accepts 63-dim (7 root + 56 joints), which maps directly
        to qpos[0:63].  Useful for mocap replay.
        """
        n = len(qpos)
        assert n == 63, f"Expected 63 dim qpos, got {n}"
        self.data.qpos[:63] = qpos

    def set_humanoid_qvel(self, qvel: np.ndarray):
        """Set humanoid qvel (root + 56 hinge DOFs).

        Accepts 62-dim (6 root + 56 joints), which maps directly
        to qvel[0:62].
        """
        n = len(qvel)
        assert n == 62, f"Expected 62 dim qvel, got {n}"
        self.data.qvel[:62] = qvel

    def set_bat_state(self, pos: np.ndarray, quat: np.ndarray):
        """Set the bat free-body position and orientation.

        Parameters
        ----------
        pos : (3,) world position of the bat body origin
        quat : (4,) quaternion [w, x, y, z] for bat orientation
        """
        self.data.qpos[_BAT_QPOS_START:_BAT_QPOS_START + 3] = pos
        self.data.qpos[_BAT_QPOS_START + 3:_BAT_QPOS_START + 7] = quat
        # Zero velocity so bat doesn't fly away
        self.data.qvel[_BAT_QVEL_START:_BAT_QVEL_START + 6] = 0.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
