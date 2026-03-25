"""End-to-end test of the CMU mocap pipeline.

Steps:
  1. Download CMU Subject 124 data (if not present)
  2. Parse ASF skeleton and AMC motion
  3. Print skeleton hierarchy and joint info
  4. Retarget motion to MuJoCo humanoid
  5. Save retargeted motion to data/processed/
  6. Print summary
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_cmu_data import main as download_main
from src.motion.asf_amc_parser import parse_asf, parse_amc, print_skeleton
from src.motion.retarget import (
    retarget_motion,
    save_retargeted_motion,
    MUJOCO_JOINT_NAMES,
)
from src.motion.replay import MocapReplay

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> int:
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Download CMU Subject 124 data")
    print("=" * 60)
    asf_path = DATA_DIR / "124.asf"
    amc_path = DATA_DIR / "124_07.amc"

    if not asf_path.exists() or not amc_path.exists():
        download_main()
    else:
        print("  Data already present, skipping download.")

    if not asf_path.exists():
        print(f"ERROR: ASF file not found: {asf_path}")
        return 1
    if not amc_path.exists():
        print(f"ERROR: AMC file not found: {amc_path}")
        return 1

    print()

    # ------------------------------------------------------------------
    # 2. Parse ASF
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Parse ASF skeleton")
    print("=" * 60)
    joints = parse_asf(asf_path)
    print(f"  Loaded {len(joints)} joints from {asf_path.name}")
    print()

    # ------------------------------------------------------------------
    # 3. Print skeleton hierarchy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Skeleton hierarchy")
    print("=" * 60)
    print_skeleton(joints)
    print()

    # Print joint details
    print("Joint details:")
    print(f"  {'Name':<16} {'DOF':<20} {'Length':>8}  {'Direction'}")
    print("  " + "-" * 70)
    for name, j in joints.items():
        if name == "root":
            continue
        dof_str = ",".join(j.dof) if j.dof else "-"
        dir_str = f"[{j.direction[0]:6.3f}, {j.direction[1]:6.3f}, {j.direction[2]:6.3f}]"
        print(f"  {name:<16} {dof_str:<20} {j.length:>8.3f}  {dir_str}")
    print()

    # ------------------------------------------------------------------
    # 4. Parse AMC
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Parse AMC motion")
    print("=" * 60)
    t0 = time.time()
    frames = parse_amc(amc_path, joints)
    dt = time.time() - t0
    print(f"  Loaded {len(frames)} frames from {amc_path.name} ({dt:.2f}s)")

    if len(frames) == 0:
        print("ERROR: No frames parsed from AMC file!")
        return 1

    # Print first frame summary
    f0 = frames[0]
    print(f"  Frame 0 joints: {list(f0.keys())}")
    if "root" in f0:
        print(f"  Root values: {f0['root']}")

    # Check which CMU joints are present that map to MuJoCo
    cmu_keys_needed = ["lowerback", "rfemur", "rtibia", "lfemur", "ltibia",
                       "rhumerus", "rradius", "lhumerus", "lradius"]
    mapped = [k for k in cmu_keys_needed if k in f0]
    missing = [k for k in cmu_keys_needed if k not in f0]
    print(f"  Mapped CMU joints found:   {mapped}")
    if missing:
        print(f"  WARNING: Missing CMU joints: {missing}")
        warnings.append(f"Missing CMU joints for retargeting: {missing}")
    print()

    # ------------------------------------------------------------------
    # 5. Retarget
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Retarget motion to MuJoCo humanoid")
    print("=" * 60)
    t0 = time.time()
    joint_targets, root_pos, root_quat = retarget_motion(joints, frames)
    dt = time.time() - t0
    print(f"  Retargeting completed in {dt:.2f}s")
    print()

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Save retargeted motion")
    print("=" * 60)
    out_path = OUT_DIR / "swing_124_07.npz"
    save_retargeted_motion(out_path, joint_targets, root_pos, root_quat, fps=120.0)
    print()

    # ------------------------------------------------------------------
    # 7. Test replay
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Test MocapReplay")
    print("=" * 60)
    replay = MocapReplay(out_path)
    print(f"  {replay}")
    action = replay.get_action()
    print(f"  First action shape: {action.shape}")
    print(f"  First action (rad): {action}")
    pos, quat = replay.get_root_state()
    print(f"  Root pos: {pos}")
    print(f"  Root quat: {quat}")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total frames:        {len(frames)}")
    print(f"  MuJoCo joints:       {len(MUJOCO_JOINT_NAMES)}")
    print(f"  Joint targets shape: {joint_targets.shape}")
    print(f"  Root pos shape:      {root_pos.shape}")
    print(f"  Root quat shape:     {root_quat.shape}")
    print(f"  Output file:         {out_path}")
    print(f"  Output size:         {out_path.stat().st_size:,} bytes")

    # Angle statistics
    print(f"\n  Joint angle statistics (radians):")
    print(f"  {'Joint':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 50)
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        col = joint_targets[:, i]
        print(f"  {name:<20} {col.min():8.3f} {col.max():8.3f} "
              f"{col.mean():8.3f} {col.std():8.3f}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  No warnings.")

    print("\nPipeline test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
