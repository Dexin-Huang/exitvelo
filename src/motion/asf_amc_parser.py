"""Parser for CMU Motion Capture ASF/AMC file formats.

No external dependencies beyond numpy.

ASF (Acclaim Skeleton File) defines the skeleton hierarchy.
AMC (Acclaim Motion Capture) defines per-frame joint angles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Joint:
    """Represents a single joint/bone in the ASF skeleton."""

    name: str
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Euler angles (deg)
    axis_order: str = "XYZ"  # rotation order string
    dof: list[str] = field(default_factory=list)  # e.g. ['rx','ry','rz']
    limits: list[tuple[float, float]] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rotation helpers  (pure numpy, no external deps)
# ---------------------------------------------------------------------------

def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_AXIS_FN = {"X": _rx, "Y": _ry, "Z": _rz}


def euler2mat(angles: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Convert Euler angles (degrees) to a 3x3 rotation matrix.

    Parameters
    ----------
    angles : array-like of length 3
        Rotation angles in **degrees**.
    order : str
        Rotation order, e.g. ``'XYZ'``, ``'ZYX'``, etc.

    Returns
    -------
    R : (3, 3) rotation matrix.
    """
    angles = np.asarray(angles, dtype=float)
    rads = np.deg2rad(angles)
    order = order.upper()
    R = np.eye(3)
    for ax_char, rad in zip(order, rads):
        R = R @ _AXIS_FN[ax_char](rad)
    return R


def mat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)  # normalise


# ---------------------------------------------------------------------------
# ASF parser
# ---------------------------------------------------------------------------

def parse_asf(filepath: str | Path) -> dict[str, Joint]:
    """Parse an ASF skeleton file.

    Returns a dict mapping joint name -> Joint.  The special ``'root'`` key
    stores the root joint metadata.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    joints: dict[str, Joint] = {}

    # Root joint
    root = Joint(name="root")
    root.dof = ["tx", "ty", "tz", "rx", "ry", "rz"]
    joints["root"] = root

    # ---- helpers ----
    section: str | None = None
    bone_block: list[str] = []
    in_bone = False

    def _flush_bone(block: list[str]):
        """Parse a single bone block and add to joints."""
        j = Joint(name="")
        i = 0
        while i < len(block):
            tok = block[i].strip()
            if tok.startswith("name"):
                j.name = tok.split()[1]
            elif tok.startswith("direction"):
                parts = tok.split()
                j.direction = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tok.startswith("length"):
                j.length = float(tok.split()[1])
            elif tok.startswith("axis"):
                parts = tok.split()
                j.axis = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) > 4:
                    j.axis_order = parts[4].upper()
            elif tok.startswith("dof"):
                j.dof = tok.split()[1:]
            elif tok.startswith("limits"):
                # limits can span multiple lines
                # Collect limit pairs from this line and subsequent lines
                rest = tok[len("limits"):].strip()
                limit_text = rest
                i += 1
                while i < len(block):
                    line = block[i].strip()
                    if line.startswith("(") or (line and line[0] == '('):
                        limit_text += " " + line
                        i += 1
                    else:
                        i -= 1
                        break
                # Parse all (lo hi) pairs
                pairs = re.findall(r"\(\s*([-\d.e+]+)\s+([-\d.e+]+)\s*\)", limit_text)
                j.limits = [(float(lo), float(hi)) for lo, hi in pairs]
            i += 1
        if j.name:
            joints[j.name] = j

    # ---- parse root section ----
    root_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":root"):
            root_section = True
            continue
        if root_section:
            if stripped.startswith(":"):
                root_section = False
            else:
                if stripped.startswith("position"):
                    parts = stripped.split()
                    root.direction = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                elif stripped.startswith("orientation"):
                    parts = stripped.split()
                    root.axis = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                elif stripped.startswith("order"):
                    root.dof = stripped.split()[1:]
                elif stripped.startswith("axis"):
                    root.axis_order = stripped.split()[1].upper()

    # ---- parse bonedata ----
    in_bonedata = False
    bone_block = []
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":bonedata"):
            in_bonedata = True
            continue
        if in_bonedata:
            if stripped.startswith(":") and not stripped.startswith(":bonedata"):
                # flush any remaining bone
                if bone_block:
                    _flush_bone(bone_block)
                    bone_block = []
                in_bonedata = False
                continue
            if stripped == "begin":
                bone_block = []
                brace_depth += 1
                continue
            if stripped == "end":
                brace_depth -= 1
                if bone_block:
                    _flush_bone(bone_block)
                    bone_block = []
                continue
            if brace_depth > 0:
                bone_block.append(stripped)

    # ---- parse hierarchy ----
    in_hierarchy = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":hierarchy"):
            in_hierarchy = True
            continue
        if in_hierarchy:
            if stripped == "begin":
                continue
            if stripped == "end" or (stripped.startswith(":") and stripped != ":hierarchy"):
                in_hierarchy = False
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                parent_name = parts[0]
                child_names = parts[1:]
                for child_name in child_names:
                    if child_name in joints:
                        joints[child_name].parent = parent_name
                    if parent_name in joints:
                        if child_name not in joints[parent_name].children:
                            joints[parent_name].children.append(child_name)

    return joints


# ---------------------------------------------------------------------------
# AMC parser
# ---------------------------------------------------------------------------

def parse_amc(filepath: str | Path, joints: dict[str, Joint]) -> list[dict[str, np.ndarray]]:
    """Parse an AMC motion file.

    Parameters
    ----------
    filepath : path to .amc file
    joints : dict from :func:`parse_asf` (needed to know DOF counts)

    Returns
    -------
    frames : list of dicts, each mapping joint_name -> np.ndarray of angles.
        For the root, the array has 6 elements [tx, ty, tz, rx, ry, rz].
        For other joints, the array length equals len(joint.dof).
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    frames: list[dict[str, np.ndarray]] = []
    current_frame: dict[str, np.ndarray] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(":"):
            continue

        parts = stripped.split()

        # A line with a single integer is a frame number
        if len(parts) == 1:
            try:
                _frame_num = int(parts[0])
                # Save previous frame
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = {}
                continue
            except ValueError:
                pass

        # Joint data line: joint_name val1 [val2] [val3] ...
        if current_frame is not None and len(parts) >= 2:
            joint_name = parts[0]
            values = np.array([float(v) for v in parts[1:]])
            current_frame[joint_name] = values

    # Don't forget last frame
    if current_frame is not None:
        frames.append(current_frame)

    return frames


# ---------------------------------------------------------------------------
# Convenience: print skeleton hierarchy
# ---------------------------------------------------------------------------

def print_skeleton(joints: dict[str, Joint], root: str = "root", indent: int = 0):
    """Pretty-print the skeleton tree."""
    j = joints.get(root)
    if j is None:
        return
    dof_str = ", ".join(j.dof) if j.dof else "none"
    print(f"{'  ' * indent}{j.name}  (dof: {dof_str}, length: {j.length:.2f})")
    for child in j.children:
        print_skeleton(joints, child, indent + 1)
