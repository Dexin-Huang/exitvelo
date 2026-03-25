"""MuJoCo-based batting environment for baseball swing simulation."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.env.contacts import detect_bat_ball_contact

_XML_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "mujoco" / "batting_scene.xml"

# Joint index ranges within qpos / qvel (humanoid only, excluding root free joint)
_QPOS_JOINT_START = 7   # after root free joint (3 pos + 4 quat)
_QPOS_JOINT_END = 24    # 17 hinge joints
_QVEL_JOINT_START = 6   # after root free joint (3 lin + 3 ang)
_QVEL_JOINT_END = 23    # 17 hinge dofs

# Ball free-joint indices
_BALL_QPOS_START = 24   # 3 pos + 4 quat
_BALL_QVEL_START = 23   # 3 lin + 3 ang

_N_JOINTS = 17


class BattingEnv(gym.Env):
    """Gymnasium environment for humanoid baseball batting.

    Observation (62-dim):
        joint_pos (17) + joint_vel (17) + root_pos (3) + root_quat (4) +
        root_lin_vel (3) + root_ang_vel (3) + bat_tip_pos (3) + bat_tip_vel (3) +
        ball_pos (3) + ball_vel (3) + ball_rel (3)

    Action (17-dim):
        Target joint positions fed through PD control.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # ── Scaled MLB pitch defaults ──
    # Scale factor: MuJoCo humanoid (1.4m) / real MLB player (1.83m) = 0.765
    # Release distance: 16.76m * 0.765 = 12.82m
    # Release height:   1.83m * 0.765  = 1.40m
    # Arrival speed:    38 m/s * sqrt(0.765) = 33.24 m/s (Froude-scaled 85 mph)
    # Middle-middle strike zone height: 0.67m
    # vz solved so ball drops from 1.40m to 0.67m under gravity over flight time
    MLB_PITCH_X = -12.82
    MLB_PITCH_Y = 0.0
    MLB_PITCH_HEIGHT = 1.40
    MLB_PITCH_SPEED = 33.24
    MLB_PITCH_VZ = -0.002  # nearly flat — gravity drop ≈ height difference

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
        # If pitch_velocity is provided as [vx, vy, vz], use it directly;
        # otherwise fall back to [pitch_speed, 0, 0].
        self.pitch_velocity = (
            np.array(pitch_velocity, dtype=np.float64)
            if pitch_velocity is not None
            else None
        )

        # PD gains -- stronger for legs/abdomen, lighter for arms
        # Joint order (17): abdomen_z, abdomen_y, abdomen_x,
        #   right_hip_x, right_hip_z, right_hip_y, right_knee,
        #   left_hip_x, left_hip_z, left_hip_y, left_knee,
        #   right_shoulder1, right_shoulder2, right_elbow,
        #   left_shoulder1, left_shoulder2, left_elbow
        self.kp = np.array(
            [100, 100, 100,          # abdomen
             100, 100, 100, 100,     # right leg
             100, 100, 100, 100,     # left leg
             50, 50, 50,             # right arm
             50, 50, 50],            # left arm
            dtype=np.float64,
        )
        self.kd = 0.1 * self.kp

        # Build joint limits for action space (in radians)
        joint_ranges = np.zeros((_N_JOINTS, 2))
        for i in range(_N_JOINTS):
            # hinge joints start at joint index 1 (0 is root free)
            jnt_id = i + 1  # skip root free joint
            joint_ranges[i] = self.model.jnt_range[jnt_id]  # already in radians after compilation

        self.joint_lo = joint_ranges[:, 0].astype(np.float32)
        self.joint_hi = joint_ranges[:, 1].astype(np.float32)

        self.action_space = spaces.Box(
            low=self.joint_lo, high=self.joint_hi, dtype=np.float32
        )

        # Observation space (62 dims)
        obs_dim = 62
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Renderer
        self._renderer = None
        if self.render_mode is not None:
            self._init_renderer()

        # Contact tracking
        self._contact_made = False

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------
    def _init_renderer(self):
        if self.render_mode == "human":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        elif self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def render(self):
        if self._renderer is None:
            return None
        # Use the 'track' camera (index 0) which follows the torso
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

        # Clip to actuator control range
        ctrl_range = self.model.actuator_ctrlrange
        torque = np.clip(torque, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = torque

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

        # Torso height
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
        joint_pos = self.data.qpos[_QPOS_JOINT_START:_QPOS_JOINT_END].copy()  # 17
        joint_vel = self.data.qvel[_QVEL_JOINT_START:_QVEL_JOINT_END].copy()  # 17
        root_pos = self.data.qpos[0:3].copy()                                  # 3
        root_quat = self.data.qpos[3:7].copy()                                 # 4
        root_lin_vel = self.data.qvel[0:3].copy()                               # 3
        root_ang_vel = self.data.qvel[3:6].copy()                               # 3

        # Sensor data
        bat_tip_pos = self.data.sensordata[0:3].copy()    # 3
        bat_tip_vel = self.data.sensordata[3:6].copy()    # 3
        ball_pos = self.data.sensordata[6:9].copy()        # 3
        ball_vel = self.data.sensordata[9:12].copy()       # 3

        # Relative position: ball - bat_tip
        ball_rel = ball_pos - bat_tip_pos                   # 3

        obs = np.concatenate([
            joint_pos,      # 17
            joint_vel,      # 17
            root_pos,       # 3
            root_quat,      # 4
            root_lin_vel,   # 3
            root_ang_vel,   # 3
            bat_tip_pos,    # 3
            bat_tip_vel,    # 3
            ball_pos,       # 3
            ball_vel,       # 3
            ball_rel,       # 3
        ])  # total = 62
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

        # Velocity: use explicit velocity if provided, else pitch in +x
        if self.pitch_velocity is not None:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = self.pitch_velocity
        else:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = [
                self.pitch_speed, 0.0, 0.0
            ]
        # No angular velocity
        self.data.qvel[_BALL_QVEL_START + 3:_BALL_QVEL_START + 6] = [0, 0, 0]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
