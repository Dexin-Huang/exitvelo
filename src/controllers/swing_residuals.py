"""Named swing-timing residuals — the action space for Tier 2A and Tier 2B.

5 scalars (swing_timing_s, hip_fire_rad, uppercut_rad, barrel_roll_rad,
plate_reach_m) with physical bounds and a normalized [-1, +1] view. The
first 3 are wired through the mocap-replay pipeline; the last 2 are
deferred (they raise NotImplementedError).

Entry points (consumers):
  src/controllers/cmu_tracking_controller.py  -- applies them to qpos
  src/optim/cma_runner.py                     -- Tier 2A search space
  src/optim/swing_residual_env.py             -- Tier 2B PPO action space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

# Joint indices in the dm_control CMU humanoid qpos (63-dim).
# Verified against scripts/probe_joints output:
#   qadr 21: lowerbackrz  (twist around vertical = pelvis-trunk yaw)
#   qadr 28: thoraxry     (forward-back lean = uppercut)
QPOS_LOWERBACK_RZ = 21
QPOS_THORAX_RY = 28


@dataclass
class SwingResiduals:
    """5 named open-loop swing-timing residuals.

    All units are physical. Defaults are zero (= raw mocap reference).
    Bounds are enforced by `denormalize`/`clip`.
    """

    swing_timing_s: float = 0.0
    hip_fire_rad: float = 0.0
    uppercut_rad: float = 0.0
    barrel_roll_rad: float = 0.0
    plate_reach_m: float = 0.0

    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "swing_timing_s":  (-0.08, +0.08),
        "hip_fire_rad":    (-0.30, +0.30),
        "uppercut_rad":    (-0.22, +0.26),
        "barrel_roll_rad": (-0.25, +0.25),
        "plate_reach_m":   (-0.10, +0.10),
    }

    @classmethod
    def from_normalized(cls, x: np.ndarray | list) -> "SwingResiduals":
        """Build from a 5-vector in `[-1, +1]` (the optimizer's view)."""
        x = np.asarray(x, dtype=np.float64).flatten()
        assert x.shape == (5,), f"expected 5-d normalized vector, got {x.shape}"
        names = list(cls.BOUNDS.keys())
        kwargs = {}
        for i, name in enumerate(names):
            lo, hi = cls.BOUNDS[name]
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            kwargs[name] = float(mid + half * np.clip(x[i], -1.0, 1.0))
        return cls(**kwargs)

    def to_normalized(self) -> np.ndarray:
        """Inverse of `from_normalized`. Useful for warm-starting."""
        out = np.zeros(5, dtype=np.float64)
        for i, (name, (lo, hi)) in enumerate(self.BOUNDS.items()):
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            v = getattr(self, name)
            out[i] = 0.0 if half == 0 else (v - mid) / half
        return out

    def clip(self) -> "SwingResiduals":
        """Project values into the configured bounds."""
        kwargs = {}
        for name, (lo, hi) in self.BOUNDS.items():
            kwargs[name] = float(np.clip(getattr(self, name), lo, hi))
        return SwingResiduals(**kwargs)


# ---------------------------------------------------------- transforms

def apply_pelvis_trunk_yaw(qpos: np.ndarray, hip_fire_rad: float) -> np.ndarray:
    """Add `hip_fire_rad` to the lowerback yaw joint of `qpos`.

    Modifies a copy and returns it. Operates on the dm_control CMU
    humanoid's 63-dim qpos layout where `qpos[21]` is `lowerbackrz`.
    """
    q = qpos.copy()
    q[QPOS_LOWERBACK_RZ] = q[QPOS_LOWERBACK_RZ] + float(hip_fire_rad)
    return q


def apply_uppercut_pitch(qpos: np.ndarray, uppercut_rad: float) -> np.ndarray:
    """Add `uppercut_rad` to the thorax forward-back lean joint."""
    q = qpos.copy()
    q[QPOS_THORAX_RY] = q[QPOS_THORAX_RY] + float(uppercut_rad)
    return q


def adjust_bat_grip_anchor(*args, **kwargs):  # pragma: no cover
    """Placeholder for the bat-grip equality-anchor adjustment used by
    `barrel_roll_rad` and `plate_reach_m`. Not wired in this iteration.
    """
    raise NotImplementedError(
        "barrel_roll and plate_reach are deferred to a follow-up; "
        "use 0.0 for those scalars in Tier 2."
    )
