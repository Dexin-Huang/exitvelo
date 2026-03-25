"""Retarget CMU mocap motions to the MuJoCo gymnasium humanoid.

Coordinate systems
------------------
- CMU  : Y-up   (right-handed)
- MuJoCo : Z-up (right-handed)

Transform:  mj_pos = [cmu_x, -cmu_z, cmu_y]

The gymnasium humanoid has 17 actuated hinge joints (in actuator order):
    abdomen_y, abdomen_z, abdomen_x,
    right_hip_x, right_hip_z, right_hip_y, right_knee,
    left_hip_x, left_hip_z, left_hip_y, left_knee,
    right_shoulder1, right_shoulder2, right_elbow,
    left_shoulder1, left_shoulder2, left_elbow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.motion.asf_amc_parser import Joint, euler2mat, mat2quat

# ---------------------------------------------------------------------------
# MuJoCo joint spec  (order matches the actuator list in humanoid.xml)
# ---------------------------------------------------------------------------

MUJOCO_JOINTS = [
    # (name, axis (unnormalised from XML), range_deg)
    ("abdomen_y",       np.array([0, 1, 0], dtype=float),   (-75,  30)),
    ("abdomen_z",       np.array([0, 0, 1], dtype=float),   (-45,  45)),
    ("abdomen_x",       np.array([1, 0, 0], dtype=float),   (-35,  35)),
    ("right_hip_x",     np.array([1, 0, 0], dtype=float),   (-25,   5)),
    ("right_hip_z",     np.array([0, 0, 1], dtype=float),   (-60,  35)),
    ("right_hip_y",     np.array([0, 1, 0], dtype=float),   (-110, 20)),
    ("right_knee",      np.array([0,-1, 0], dtype=float),   (-160, -2)),
    ("left_hip_x",      np.array([-1, 0, 0], dtype=float),  (-25,   5)),
    ("left_hip_z",      np.array([0, 0,-1], dtype=float),   (-60,  35)),
    ("left_hip_y",      np.array([0, 1, 0], dtype=float),   (-110, 20)),
    ("left_knee",       np.array([0,-1, 0], dtype=float),   (-160, -2)),
    ("right_shoulder1", np.array([2, 1, 1], dtype=float),   (-85,  60)),
    ("right_shoulder2", np.array([0,-1, 1], dtype=float),   (-85,  60)),
    ("right_elbow",     np.array([0,-1, 1], dtype=float),   (-90,  50)),
    ("left_shoulder1",  np.array([2,-1, 1], dtype=float),   (-60,  85)),
    ("left_shoulder2",  np.array([0, 1, 1], dtype=float),   (-60,  85)),
    ("left_elbow",      np.array([0,-1,-1], dtype=float),   (-90,  50)),
]

MUJOCO_JOINT_NAMES = [j[0] for j in MUJOCO_JOINTS]
NUM_JOINTS = len(MUJOCO_JOINTS)  # 17

# Normalised axes
_AXES = {name: ax / np.linalg.norm(ax) for name, ax, _ in MUJOCO_JOINTS}
_RANGES = {name: rng for name, _, rng in MUJOCO_JOINTS}

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _cmu_euler_to_mat(angles_deg: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Build rotation matrix from CMU Euler angles (degrees)."""
    return euler2mat(angles_deg, order)


def _transform_rotation_cmu_to_mj(R_cmu: np.ndarray) -> np.ndarray:
    """Rotate a CMU (Y-up) rotation matrix into MuJoCo (Z-up) frame.

    The coordinate transform is:
        mj_x =  cmu_x
        mj_y = -cmu_z
        mj_z =  cmu_y

    This corresponds to pre- and post-multiplying by the basis change matrix.
    """
    # Basis change: maps CMU basis vectors to MuJoCo basis vectors
    #   CMU x -> MJ x    (1, 0, 0)
    #   CMU y -> MJ z    (0, 0, 1)
    #   CMU z -> MJ -y   (0,-1, 0)
    T = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=float)
    return T @ R_cmu @ T.T


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def decompose_rotation_to_two_axes(
    R_target: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
) -> tuple[float, float]:
    """Find theta1, theta2 such that R(axis1, t1) @ R(axis2, t2) ~ R_target.

    Uses numerical optimisation (Nelder-Mead).
    """
    a1 = axis1 / np.linalg.norm(axis1)
    a2 = axis2 / np.linalg.norm(axis2)

    def objective(params):
        t1, t2 = params
        R_approx = _axis_angle_to_matrix(a1, t1) @ _axis_angle_to_matrix(a2, t2)
        return np.linalg.norm(R_target - R_approx, "fro")

    result = minimize(objective, [0.0, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    return float(result.x[0]), float(result.x[1])


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Main retargeting
# ---------------------------------------------------------------------------

def retarget_frame(
    frame: dict[str, np.ndarray],
    asf_joints: dict[str, Joint],
) -> np.ndarray:
    """Retarget a single CMU frame to 17 MuJoCo joint angles (degrees).

    Parameters
    ----------
    frame : dict mapping CMU joint name -> angle array (degrees)
    asf_joints : parsed ASF skeleton joints

    Returns
    -------
    angles : (17,) array in the order of MUJOCO_JOINT_NAMES
    """
    angles = np.zeros(NUM_JOINTS)

    # Scaling factor for simple angle mapping.
    # CMU angles may be larger than MuJoCo ranges allow, so we apply a
    # conservative scale and then clamp.
    SCALE = 0.8

    # --- Abdomen (lowerback) ---
    if "lowerback" in frame:
        lb = frame["lowerback"]
        # CMU lowerback typically has dof rx, ry, rz
        # Map:  ry -> abdomen_y,  rz -> abdomen_z,  rx -> abdomen_x
        rx = lb[0] if len(lb) > 0 else 0.0
        ry = lb[1] if len(lb) > 1 else 0.0
        rz = lb[2] if len(lb) > 2 else 0.0
        angles[0] = ry * SCALE   # abdomen_y
        angles[1] = rz * SCALE   # abdomen_z
        angles[2] = rx * SCALE   # abdomen_x

    # --- Right leg ---
    if "rfemur" in frame:
        rf = frame["rfemur"]
        rx = rf[0] if len(rf) > 0 else 0.0
        ry = rf[1] if len(rf) > 1 else 0.0
        rz = rf[2] if len(rf) > 2 else 0.0
        angles[3] = rx * SCALE   # right_hip_x
        angles[4] = rz * SCALE   # right_hip_z
        angles[5] = ry * SCALE   # right_hip_y

    if "rtibia" in frame:
        rt = frame["rtibia"]
        # rtibia has 1 DOF (rx).  CMU flexion is positive (0..170 deg).
        # MuJoCo right_knee axis is (0,-1,0), range (-160, -2).
        # Negate to map CMU positive flexion -> MuJoCo negative range.
        rx = rt[0] if len(rt) > 0 else 0.0
        angles[6] = -rx * SCALE  # right_knee (negated)

    # --- Left leg ---
    if "lfemur" in frame:
        lf = frame["lfemur"]
        rx = lf[0] if len(lf) > 0 else 0.0
        ry = lf[1] if len(lf) > 1 else 0.0
        rz = lf[2] if len(lf) > 2 else 0.0
        # Left side: hip_x axis is (-1,0,0), hip_z axis is (0,0,-1)
        # The negated axes mean that a positive MuJoCo angle rotates in
        # the opposite direction compared to a positive CMU angle on the
        # corresponding standard axis.  We therefore negate.
        angles[7] = rx * SCALE   # left_hip_x  (axis -1,0,0; CMU rx is already neg for flexion)
        angles[8] = rz * SCALE   # left_hip_z  (axis 0,0,-1; keep sign, clamped later)
        angles[9] = ry * SCALE   # left_hip_y

    if "ltibia" in frame:
        lt = frame["ltibia"]
        # Same sign convention as right knee: negate CMU positive flexion.
        rx = lt[0] if len(lt) > 0 else 0.0
        angles[10] = -rx * SCALE  # left_knee (negated)

    # --- Right shoulder (non-trivial axis decomposition) ---
    if "rhumerus" in frame:
        rh = frame["rhumerus"]
        if len(rh) >= 3:
            # Build CMU rotation matrix, transform to MuJoCo frame, then
            # decompose onto the two non-standard shoulder axes.
            R_cmu = _cmu_euler_to_mat(rh[:3])
            R_mj = _transform_rotation_cmu_to_mj(R_cmu)
            t1, t2 = decompose_rotation_to_two_axes(
                R_mj,
                _AXES["right_shoulder1"],
                _AXES["right_shoulder2"],
            )
            angles[11] = np.rad2deg(t1)  # right_shoulder1
            angles[12] = np.rad2deg(t2)  # right_shoulder2
        elif len(rh) >= 1:
            angles[11] = rh[0] * SCALE

    # --- Right elbow ---
    if "rradius" in frame:
        rr = frame["rradius"]
        rx = rr[0] if len(rr) > 0 else 0.0
        angles[13] = rx * SCALE  # right_elbow
    elif "rhand" in frame:
        # Some ASF files use different naming
        pass

    # --- Left shoulder ---
    if "lhumerus" in frame:
        lh = frame["lhumerus"]
        if len(lh) >= 3:
            R_cmu = _cmu_euler_to_mat(lh[:3])
            R_mj = _transform_rotation_cmu_to_mj(R_cmu)
            t1, t2 = decompose_rotation_to_two_axes(
                R_mj,
                _AXES["left_shoulder1"],
                _AXES["left_shoulder2"],
            )
            angles[14] = np.rad2deg(t1)  # left_shoulder1
            angles[15] = np.rad2deg(t2)  # left_shoulder2
        elif len(lh) >= 1:
            angles[14] = lh[0] * SCALE

    # --- Left elbow ---
    if "lradius" in frame:
        lr = frame["lradius"]
        rx = lr[0] if len(lr) > 0 else 0.0
        angles[15 + 1] = rx * SCALE  # left_elbow  (index 16)
    elif "lhand" in frame:
        pass

    # Clamp to MuJoCo joint ranges
    for i, (name, _, rng) in enumerate(MUJOCO_JOINTS):
        angles[i] = _clamp(angles[i], rng[0], rng[1])

    return angles


def retarget_root(
    frame: dict[str, np.ndarray],
    asf_joints: dict[str, Joint],
    position_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CMU root position & orientation to MuJoCo frame.

    Returns (pos_xyz, quat_wxyz).
    """
    root_data = frame.get("root", np.zeros(6))

    # Root data is [tx, ty, tz, rx, ry, rz]  (position in ASF units, angles in degrees)
    tx, ty, tz = root_data[0], root_data[1], root_data[2]
    rx, ry, rz = root_data[3], root_data[4], root_data[5]

    # Position transform: CMU Y-up -> MuJoCo Z-up
    #   mj_x =  cmu_x
    #   mj_y = -cmu_z
    #   mj_z =  cmu_y
    pos = np.array([tx, -tz, ty]) * position_scale

    # Orientation
    R_cmu = _cmu_euler_to_mat(np.array([rx, ry, rz]), order="XYZ")
    R_mj = _transform_rotation_cmu_to_mj(R_cmu)
    quat = mat2quat(R_mj)

    return pos, quat


def retarget_motion(
    asf_joints: dict[str, Joint],
    amc_frames: list[dict[str, np.ndarray]],
    position_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retarget a full motion sequence.

    Parameters
    ----------
    asf_joints : dict from parse_asf
    amc_frames : list of frame dicts from parse_amc
    position_scale : float or None
        If None, auto-detect from ASF root position (CMU typically stores
        positions in ~inches or ~cm; MuJoCo uses metres).

    Returns
    -------
    joint_targets : (T, 17) joint angles in degrees
    root_positions : (T, 3) root position in MuJoCo frame (metres)
    root_quats : (T, 4) root quaternion [w, x, y, z]
    """
    T = len(amc_frames)
    joint_targets = np.zeros((T, NUM_JOINTS))
    root_positions = np.zeros((T, 3))
    root_quats = np.zeros((T, 4))

    # Auto-detect position scale from root position in first frame.
    # CMU typically uses a unit scale factor given in the ASF header.
    # A reasonable heuristic: if root Y > 50, likely in some unit where
    # dividing by ~100 gives meters. The humanoid stands at ~1.4m.
    if position_scale is None:
        first_root = amc_frames[0].get("root", np.zeros(6))
        root_height_cmu = first_root[1]  # Y is up in CMU
        if abs(root_height_cmu) > 10:
            # Estimate scale: humanoid height in MuJoCo is ~1.4m
            position_scale = 1.4 / max(abs(root_height_cmu), 1e-6)
        else:
            position_scale = 0.01  # fallback: cm -> m

    print(f"  [retarget] Using position_scale = {position_scale:.6f}")
    print(f"  [retarget] Processing {T} frames ...")

    for t in range(T):
        joint_targets[t] = retarget_frame(amc_frames[t], asf_joints)
        root_positions[t], root_quats[t] = retarget_root(
            amc_frames[t], asf_joints, position_scale
        )

    # Convert joint angles to radians for MuJoCo (MuJoCo XML says angle="degree"
    # for the compiler, but ctrl values go through the actuator; for position
    # control the targets should be in radians).
    # Actually, the humanoid.xml uses <compiler angle="degree"/>, so the joint
    # ranges are specified in degrees and the actuators expect torque inputs,
    # not position targets. For our trajectory, we keep degrees as that is
    # what a PD controller would use as reference angles.
    # We convert to radians here since that is the standard for MuJoCo qpos.
    joint_targets_rad = np.deg2rad(joint_targets)

    return joint_targets_rad, root_positions, root_quats


def save_retargeted_motion(
    output_path: str | Path,
    joint_targets: np.ndarray,
    root_positions: np.ndarray,
    root_quats: np.ndarray,
    fps: float = 120.0,
):
    """Save retargeted motion to .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        joint_targets=joint_targets,
        root_pos=root_positions,
        root_quat=root_quats,
        fps=fps,
    )
    print(f"  [save] Saved retargeted motion to {output_path}")
    print(f"         Shape: joint_targets={joint_targets.shape}, "
          f"root_pos={root_positions.shape}, root_quat={root_quats.shape}")
