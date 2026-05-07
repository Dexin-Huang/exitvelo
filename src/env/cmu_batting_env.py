"""MuJoCo-based batting environment using the dm_control CMU humanoid (56 joints).

This replaces the old 17-joint gymnasium humanoid batting env with one that uses
the dm_control CMU humanoid skeleton, which has a 1:1 joint mapping with CMU
mocap data. The bat is attached to the left hand via a hinge joint (bat_grip).
No retargeting needed.

The contact path uses the analytical Nathan formula in `src/env/contacts.py`
for ball exit velocity (the MuJoCo `mj_step` contact solver misses fast
bat-ball collisions in this scene). On contact the env runs the analytical
ball trajectory integrator for carry / launch / total distance metrics,
which are exposed through the reward and the info dict.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.env.contacts import (
    GROUND_Z,
    detect_bat_ball_contact,
    integrate_ball_trajectory,
    nathan_exit_velocity,
)

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

# Swept-contact threshold: ball + bat collider radii (m). Generous because
# the analytical Nathan formula doesn't care exactly where on the bat.
_SWEPT_CONTACT_RADIUS = 0.06


class CMUBattingEnv(gym.Env):
    """Gymnasium environment for CMU humanoid baseball batting.

    Uses the dm_control CMU humanoid skeleton (56 actuated joints, 64 qpos)
    with a bat attached to the left hand via a hinge joint (bat_grip),
    and a baseball as a free body.

    Observation (154-dim):
        joint_pos (56) + joint_vel (56) + root_pos (3) + root_quat (4) +
        root_lin_vel (3) + root_ang_vel (3) + bat_tip_pos (3) + bat_tip_vel (3) +
        ball_pos (3) + ball_vel (3) + ball_rel_to_bat_tip (3) +
        bat_sweet_pos (3) + bat_sweet_vel (3) + ball_rel_to_sweet (3) +
        swing_phase (1) + sin_phase (1) + cos_phase (1) +
        has_contact (1) + time_since_contact (1)

    Action (56-dim):
        Target joint positions fed through PD control.

    Reward:
        Dense pre-contact: w_dense * exp(-k * |ball - sweet|^2) (only before contact)
        Sparse contact:    w_contact (1.0 on the contact step, else 0)
        Control penalty:   w_ctrl * |action|^2
        Terminal-on-contact: w_exit * max(0, exit_mph - 40)
                            + w_launch * exp(-((launch_deg - 28)/18)^2)
                            + w_carry * carry_ft
        Terminal-on-miss:  w_miss * -1.0

    Termination:
        - contact: episode ends after the analytical post-contact metrics are
          recorded in info (carry, total distance, launch, exit_mph).
        - ball past batter, humanoid fell, timeout: episode ends with miss
          penalty (and zero post-contact metrics).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # ── Real MLB field dimensions ──
    # Reference: MLB official rules; Statcast / Driveline.
    # Coordinate frame: home plate at origin, +x toward pitcher's mound,
    # +z up, +y toward 3B side from a RH batter's perspective.
    # The CMU humanoid is scaled up from mocap to 1.83 m (typical MLB hitter)
    # via SCALE = 1.343 in render_reel.py. Env distances are real-world.
    MLB_MOUND_TO_PLATE_M = 18.44      # 60 ft 6 in
    MLB_PLATE_WIDTH_M = 0.4318         # 17 in
    MLB_BALL_RADIUS_M = 0.0365         # regulation ball
    MLB_STRIKE_ZONE_LO_M = 0.50        # ~knee height for 1.83 m batter
    MLB_STRIKE_ZONE_HI_M = 1.50        # ~mid-chest letters
    MLB_RELEASE_HEIGHT_M = 1.85        # avg release point above plate plane
    MLB_RELEASE_EXTENSION_M = 1.95     # avg release ~6.4 ft in front of mound
    MLB_PITCH_SPEED_MS = 41.6          # 93 mph fastball

    # Effective distance the ball travels: mound-to-plate minus release extension
    MLB_PITCH_FLIGHT_M = MLB_MOUND_TO_PLATE_M - MLB_RELEASE_EXTENSION_M  # ~16.49 m

    # Default pitch starts at the release point: -PITCH_FLIGHT_M from plate
    # at release height, aimed +x at the strike-zone center.
    MLB_PITCH_X = -MLB_PITCH_FLIGHT_M  # ~-16.49 m
    MLB_PITCH_Y = 0.0
    MLB_PITCH_HEIGHT = MLB_RELEASE_HEIGHT_M
    MLB_PITCH_SPEED = MLB_PITCH_SPEED_MS

    # Reward weights (starting set per SPEC.md §1)
    W_CONTACT = 5.0
    W_DENSE_PRE = 1.0     # multiplies exp(-k * d^2)
    DENSE_K = 100.0       # decay constant for distance shaping
    W_EXIT = 0.05
    W_LAUNCH = 2.0
    W_CARRY = 0.02
    W_CTRL = -1e-4
    W_MISS = -2.0

    # Bat-speed gate: contacts below this sweet-spot speed (m/s) are
    # ignored. Stops the policy from learning to gently tap the ball
    # for the contact bonus instead of swinging.
    MIN_BAT_SPEED_FOR_CONTACT = 4.0  # ~9 mph. A real swing is 15-30 m/s.

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
        pitch_jitter: bool = False,
        # Jitter ranges chosen from MLB Statcast distributions:
        # speed 70-82 mph covers off-speed -> mid-90s fastball range,
        # plate height 0.5-1.5 m matches the strike zone for a 1.83 m batter,
        # lateral plate offset ~+/- 0.22 m straddles the 17 in plate width,
        # release-time offset ~+/- 40 ms covers typical timing variance.
        pitch_jitter_speed: tuple[float, float] = (31.3, 36.7),         # 70-82 mph
        pitch_jitter_height: tuple[float, float] = (0.65, 1.45),         # in strike zone
        pitch_jitter_x_offset: tuple[float, float] = (-0.22, 0.22),      # ~plate width
        pitch_jitter_release_t: tuple[float, float] = (-0.04, 0.04),     # +/-40 ms
    ):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0

        # Pitch configuration (defaults / fixed)
        self.pitch_speed = pitch_speed
        self.pitch_height = pitch_height
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.pitch_velocity = (
            np.array(pitch_velocity, dtype=np.float64)
            if pitch_velocity is not None
            else None
        )

        # Jitter ranges
        self.pitch_jitter = pitch_jitter
        self._jitter_speed = pitch_jitter_speed
        self._jitter_height = pitch_jitter_height
        self._jitter_x_offset = pitch_jitter_x_offset
        self._jitter_release_t = pitch_jitter_release_t

        # Pre-compute actuator-to-joint index mapping (vectorised PD control)
        jnt_ids = self.model.actuator_trnid[:, 0]
        self._act_qpos_idx = self.model.jnt_qposadr[jnt_ids]  # (56,)
        self._act_qvel_idx = self.model.jnt_dofadr[jnt_ids]    # (56,)

        # PD gains for the 56 CMU joints
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

        # Observation space (154 dims after the env-refactor additions)
        obs_dim = 154
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Renderer
        self._renderer = None
        if self.render_mode is not None:
            self._init_renderer()

        # Cache contact geom + site IDs
        self._bat_geom_ids = set()
        for name in ['bat_barrel', 'bat_handle', 'bat_taper', 'bat_end', 'bat_knob']:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._bat_geom_ids.add(gid)
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
        self._sweet_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'bat_sweet')

        # Per-step state
        self._contact_made = False
        self._contact_step = -1
        self._exit_velocity = np.zeros(3)
        self._exit_mph = 0.0
        self._launch_deg = 0.0
        self._carry_ft = 0.0
        self._total_ft = 0.0
        self._prev_sweet_pos = np.zeros(3)
        self._sweet_vel = np.zeros(3)
        self._swing_phase = 0.0  # set externally via set_swing_phase()

    # ------------------------------------------------------------------
    # PD gains
    # ------------------------------------------------------------------
    def _build_kp(self) -> np.ndarray:
        kp = np.zeros(_N_JOINTS)
        for i in range(_N_JOINTS):
            jnt_id = self.model.actuator_trnid[i, 0]
            jnt_name = self.model.joint(jnt_id).name
            if 'femur' in jnt_name:
                kp[i] = 100.0
            elif 'tibia' in jnt_name:
                kp[i] = 80.0
            elif 'foot' in jnt_name or 'toes' in jnt_name:
                kp[i] = 40.0
            elif 'lowerback' in jnt_name or 'upperback' in jnt_name:
                kp[i] = 100.0
            elif 'thorax' in jnt_name:
                kp[i] = 80.0
            elif 'neck' in jnt_name or 'head' in jnt_name:
                kp[i] = 20.0
            elif 'clavicle' in jnt_name:
                kp[i] = 40.0
            elif 'humerus' in jnt_name:
                kp[i] = 50.0
            elif 'radius' in jnt_name:
                kp[i] = 40.0
            elif 'wrist' in jnt_name:
                kp[i] = 30.0
            elif 'hand' in jnt_name:
                kp[i] = 20.0
            elif 'finger' in jnt_name or 'thumb' in jnt_name:
                kp[i] = 10.0
            else:
                kp[i] = 30.0
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
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if cam_id >= 0:
            self._renderer.update_scene(self.data, camera=cam_id)
        else:
            self._renderer.update_scene(self.data)
        return self._renderer.render()

    # ------------------------------------------------------------------
    # External hooks
    # ------------------------------------------------------------------
    def set_swing_phase(self, phase: float) -> None:
        """Caller (the controller) tells the env where in the swing we are.
        phase in [0, 1]. Used in the observation as phase + sin + cos."""
        self._swing_phase = float(np.clip(phase, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Humanoid neutral stance from default qpos0
        self.data.qpos[:_QPOS_JOINT_END] = self.model.qpos0[:_QPOS_JOINT_END]
        self.data.qvel[:_QVEL_JOINT_END] = 0.0

        # Resolve pitch: options > jitter > defaults
        pitch_overrides = (options or {}).get("pitch", {})
        if self.pitch_jitter and not pitch_overrides:
            self._sample_pitch_jitter()
        for k, v in pitch_overrides.items():
            setattr(self, f"pitch_{k}", v)

        self._launch_ball()
        mujoco.mj_forward(self.model, self.data)

        # Reset per-episode state
        self.step_count = 0
        self._contact_made = False
        self._contact_step = -1
        self._exit_velocity = np.zeros(3)
        self._exit_mph = 0.0
        self._launch_deg = 0.0
        self._carry_ft = 0.0
        self._total_ft = 0.0
        self._swing_phase = 0.0
        self._prev_sweet_pos = self.data.site_xpos[self._sweet_site_id].copy() \
            if self._sweet_site_id >= 0 else np.zeros(3)
        self._sweet_vel = np.zeros(3)

        return self._get_obs(), {}

    def _sample_pitch_jitter(self) -> None:
        rng = self.np_random
        self.pitch_speed = float(rng.uniform(*self._jitter_speed))
        self.pitch_height = float(rng.uniform(*self._jitter_height))
        # Lateral plate offset is along +y (perpendicular to pitch direction).
        y_off = float(rng.uniform(*self._jitter_x_offset))
        self.pitch_y = self.MLB_PITCH_Y + y_off
        # Release-time offset: simulate by translating release point along x.
        # Negative offset = ball released earlier from a slightly closer x.
        rt_off = float(rng.uniform(*self._jitter_release_t))
        self.pitch_x = self.MLB_PITCH_X + self.pitch_speed * rt_off

    def step(self, action):
        action = np.clip(action, self.joint_lo, self.joint_hi)

        # Vectorised PD control
        q = self.data.qpos[self._act_qpos_idx]
        qdot = self.data.qvel[self._act_qvel_idx]
        ctrl = self.kp * (action - q) - self.kd * qdot
        ctrl_range = self.model.actuator_ctrlrange
        ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = ctrl

        # Track bat_sweet position before stepping so we can compute its
        # velocity and run swept ball-vs-sweet contact detection.
        if self._sweet_site_id >= 0:
            prev_sweet_world = self.data.site_xpos[self._sweet_site_id].copy()
        else:
            prev_sweet_world = np.zeros(3)
        prev_ball = self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3].copy()

        # Step simulation (frame_skip sub-steps), tracking min distance
        # between ball and sweet spot across the sub-steps.
        min_dist = np.inf
        sim_dt = self.model.opt.timestep
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._sweet_site_id >= 0:
                sw = self.data.site_xpos[self._sweet_site_id]
                bp = self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3]
                d = float(np.linalg.norm(bp - sw))
                if d < min_dist:
                    min_dist = d
        self.step_count += 1

        # Sweet-spot velocity over the whole control step
        if self._sweet_site_id >= 0:
            curr_sweet_world = self.data.site_xpos[self._sweet_site_id].copy()
            self._sweet_vel = (curr_sweet_world - prev_sweet_world) / (
                self.frame_skip * sim_dt
            )
        else:
            curr_sweet_world = np.zeros(3)

        # MuJoCo contact: telemetry only per SPEC.md. The analytical Nathan
        # path is the contact source of truth.
        mj_contact, contact_info = detect_bat_ball_contact(
            self.model, self.data,
            bat_geom_ids=self._bat_geom_ids,
            ball_geom_id=self._ball_geom_id,
        )
        swept_contact = (min_dist < _SWEPT_CONTACT_RADIUS) and not self._contact_made
        first_contact_this_step = swept_contact

        # On contact: run the analytical Nathan + flight pipeline ONCE.
        # Ignore weak contacts (bat barely moving) so the policy can't
        # game the contact bonus by gently nudging the ball.
        if first_contact_this_step:
            sweet_speed = float(np.linalg.norm(self._sweet_vel))
            if sweet_speed < self.MIN_BAT_SPEED_FOR_CONTACT:
                first_contact_this_step = False
            else:
                self._contact_made = True
                self._contact_step = self.step_count
                ball_vel_pre = self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3].copy()
                self._exit_velocity = nathan_exit_velocity(self._sweet_vel, ball_vel_pre)
                # Update the ball qvel so existing scripts that read
                # data.qvel[63:66] post-contact see the analytical exit state.
                self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = self._exit_velocity
                traj, _vels, _bounces = integrate_ball_trajectory(
                    curr_sweet_world.copy(), self._exit_velocity.copy()
                )
                speed_ms = float(np.linalg.norm(self._exit_velocity))
                self._exit_mph = speed_ms * 2.237
                horiz = float(np.linalg.norm(self._exit_velocity[:2]))
                self._launch_deg = float(np.degrees(np.arctan2(self._exit_velocity[2], horiz))) if horiz > 0 else 0.0
                from src.env.contacts import carry_distance_ft, total_distance_ft
                self._carry_ft = carry_distance_ft(traj, contact_xy=curr_sweet_world[:2])
                self._total_ft = total_distance_ft(traj)

        # Termination
        ball_pos = self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3]
        torso_z = self.data.qpos[2]
        terminated = False
        truncated = False
        termination_reason = ""
        if self._contact_made:
            # Terminate on contact step now that we have analytical metrics.
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

        # Reward
        reward = self._compute_reward(
            action=action,
            first_contact=first_contact_this_step,
            terminated=terminated,
            termination_reason=termination_reason,
            min_dist=min_dist if min_dist != np.inf else 1.0,
        )

        # Backwards compat: existing scripts (run_cmu_reference_hit.py,
        # src/eval/evaluate.py) read `ball_speed_post` from info.
        ball_speed_post = float(np.linalg.norm(self._exit_velocity)) if self._contact_made else 0.0

        info = {
            "contact": self._contact_made,
            "first_contact": first_contact_this_step,
            "exit_mph": self._exit_mph,
            "launch_deg": self._launch_deg,
            "carry_ft": self._carry_ft,
            "total_ft": self._total_ft,
            "min_bat_ball_dist": float(min_dist) if min_dist != np.inf else 1.0,
            "termination_reason": termination_reason,
            "ball_speed_post": ball_speed_post,
        }
        if contact_info is not None:
            info["mj_contact_info"] = contact_info

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        *,
        action: np.ndarray,
        first_contact: bool,
        terminated: bool,
        termination_reason: str,
        min_dist: float,
    ) -> float:
        r = 0.0

        # Dense pre-contact shaping (only before contact has happened)
        if not self._contact_made or first_contact:
            r += self.W_DENSE_PRE * float(np.exp(-self.DENSE_K * min_dist * min_dist))

        # Sparse contact bonus
        if first_contact:
            r += self.W_CONTACT

        # Control penalty
        r += self.W_CTRL * float(np.dot(action, action))

        # Terminal contact reward
        if terminated and termination_reason == "contact":
            r += self.W_EXIT * max(0.0, self._exit_mph - 40.0)
            r += self.W_LAUNCH * float(np.exp(-((self._launch_deg - 28.0) / 18.0) ** 2))
            r += self.W_CARRY * self._carry_ft

        # Terminal miss penalty
        if terminated and termination_reason in ("ball_past_batter", "humanoid_fell"):
            r += self.W_MISS

        return float(r)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        joint_pos = self.data.qpos[self._act_qpos_idx].copy()
        joint_vel = self.data.qvel[self._act_qvel_idx].copy()
        root_pos = self.data.qpos[0:3]
        root_quat = self.data.qpos[3:7]
        root_lin_vel = self.data.qvel[0:3]
        root_ang_vel = self.data.qvel[3:6]

        bat_tip_pos = self.data.sensordata[_SENSOR_BAT_TIP_POS]
        bat_tip_vel = self.data.sensordata[_SENSOR_BAT_TIP_VEL]
        ball_pos = self.data.sensordata[_SENSOR_BALL_POS]
        ball_vel = self.data.sensordata[_SENSOR_BALL_VEL]
        ball_rel_tip = ball_pos - bat_tip_pos

        if self._sweet_site_id >= 0:
            sweet_pos = self.data.site_xpos[self._sweet_site_id].copy()
        else:
            sweet_pos = bat_tip_pos.copy()
        sweet_vel = self._sweet_vel.copy()
        ball_rel_sweet = ball_pos - sweet_pos

        phase = self._swing_phase
        sin_p = float(np.sin(2 * np.pi * phase))
        cos_p = float(np.cos(2 * np.pi * phase))

        time_since_contact = (
            (self.step_count - self._contact_step) * self.frame_skip * self.model.opt.timestep
            if self._contact_step >= 0 else 0.0
        )

        return np.concatenate([
            joint_pos,                                     # 56
            joint_vel,                                     # 56
            root_pos, root_quat, root_lin_vel, root_ang_vel,  # 13
            bat_tip_pos, bat_tip_vel, ball_pos, ball_vel, ball_rel_tip,  # 15
            sweet_pos, sweet_vel, ball_rel_sweet,          # 9
            np.array([phase, sin_p, cos_p], dtype=np.float64),  # 3
            np.array([float(self._contact_made), time_since_contact], dtype=np.float64),  # 2
        ])  # total = 154

    # ------------------------------------------------------------------
    # Ball launch
    # ------------------------------------------------------------------
    def _launch_ball(self):
        self.data.qpos[_BALL_QPOS_START:_BALL_QPOS_START + 3] = [
            self.pitch_x, self.pitch_y, self.pitch_height
        ]
        self.data.qpos[_BALL_QPOS_START + 3:_BALL_QPOS_START + 7] = [1, 0, 0, 0]
        if self.pitch_velocity is not None:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = self.pitch_velocity
        else:
            self.data.qvel[_BALL_QVEL_START:_BALL_QVEL_START + 3] = [
                self.pitch_speed, 0.0, 0.0
            ]
        self.data.qvel[_BALL_QVEL_START + 3:_BALL_QVEL_START + 6] = [0, 0, 0]

    # ------------------------------------------------------------------
    # Utility: set qpos directly (for mocap replay)
    # ------------------------------------------------------------------
    def set_humanoid_qpos(self, qpos: np.ndarray):
        n = len(qpos)
        assert n in (63, 64), f"Expected 63 or 64 dim qpos, got {n}"
        self.data.qpos[:n] = qpos

    def set_humanoid_qvel(self, qvel: np.ndarray):
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
