"""Sanity check for the BattingEnv MuJoCo environment."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import imageio
import mujoco
import numpy as np

from src.env.batting_env import BattingEnv


def main():
    print("=" * 60)
    print("BattingEnv Sanity Check")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Create environment and print shapes
    # ---------------------------------------------------------------
    env = BattingEnv(render_mode="rgb_array")
    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action shape:      {env.action_space.shape}")
    print(f"Obs space:         {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    assert obs.shape == (62,), f"Expected obs shape (62,), got {obs.shape}"
    assert env.action_space.shape == (17,), f"Expected action shape (17,), got {env.action_space.shape}"
    print("[PASS] Shapes are correct.")

    # ---------------------------------------------------------------
    # 2. Step 200 times with zero action -- check humanoid upright
    # ---------------------------------------------------------------
    print("\n--- Test: 200 steps with zero action ---")
    obs, info = env.reset()
    upright = True
    for step_i in range(200):
        action = np.zeros(17, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  Episode ended at step {step_i + 1}: {info.get('termination_reason', '?')}")
            if info.get("termination_reason") == "humanoid_fell":
                upright = False
            break
    else:
        torso_z = obs[34]  # root_pos z is at index 17+17+2 = 36... let me use env.data
        torso_z = env.data.qpos[2]
        print(f"  Torso z after 200 steps: {torso_z:.3f}")
        if torso_z < 0.5:
            upright = False

    if upright:
        print("[PASS] Humanoid stayed upright with zero action.")
    else:
        print("[WARN] Humanoid fell with zero action (may be expected without balance controller).")

    # ---------------------------------------------------------------
    # 3. Step 200 times with random actions
    # ---------------------------------------------------------------
    print("\n--- Test: 200 steps with random actions ---")
    obs, info = env.reset()
    for step_i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  Episode ended at step {step_i + 1}: {info.get('termination_reason', '?')}")
            break
    else:
        print(f"  Completed 200 steps. Torso z = {env.data.qpos[2]:.3f}")
    print("[PASS] Random-action rollout completed without crash.")

    # ---------------------------------------------------------------
    # 4. Contact detection test -- place ball near bat tip
    # ---------------------------------------------------------------
    print("\n--- Test: Contact detection (manual ball placement) ---")
    obs, info = env.reset()

    # Get bat tip position from sensor data
    mujoco.mj_forward(env.model, env.data)
    bat_tip_pos = env.data.sensordata[0:3].copy()
    print(f"  Bat tip position: {bat_tip_pos}")

    # Place ball right at the bat tip
    from src.env.contacts import detect_bat_ball_contact

    env.data.qpos[24:27] = bat_tip_pos + np.array([0.0, 0.0, 0.0])  # exactly at bat tip
    env.data.qpos[27:31] = [1, 0, 0, 0]
    env.data.qvel[23:29] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Step a few times to let contact resolve
    contact_detected = False
    for _ in range(10):
        mujoco.mj_step(env.model, env.data)
        hit, cinfo = detect_bat_ball_contact(env.model, env.data)
        if hit:
            contact_detected = True
            print(f"  Contact detected at pos: {cinfo['pos']}")
            break

    if contact_detected:
        print("[PASS] Contact detection works.")
    else:
        print("[WARN] No contact detected in manual test. Ball may need to be closer to bat geoms.")
        # Try placing ball closer to barrel center
        # barrel is at pos="0.50 0.50 0.50" in right_lower_arm frame
        # Get barrel body transform
        barrel_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'bat_barrel')
        barrel_pos = env.data.geom_xpos[barrel_id].copy()
        print(f"  Barrel geom world pos: {barrel_pos}")
        env.data.qpos[24:27] = barrel_pos
        env.data.qpos[27:31] = [1, 0, 0, 0]
        env.data.qvel[23:29] = 0.0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(10):
            mujoco.mj_step(env.model, env.data)
            hit, cinfo = detect_bat_ball_contact(env.model, env.data)
            if hit:
                contact_detected = True
                print(f"  Contact detected at barrel pos: {cinfo['pos']}")
                break
        if contact_detected:
            print("[PASS] Contact detection works (barrel placement).")
        else:
            print("[FAIL] Contact detection did not trigger.")

    # ---------------------------------------------------------------
    # 5. Save rendered frames
    # ---------------------------------------------------------------
    print("\n--- Test: Render and save frames ---")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    obs, info = env.reset()
    frames_saved = 0
    for step_i in range(100):
        action = np.zeros(17, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_i % 30 == 0:
            frame = env.render()
            if frame is not None:
                fname = results_dir / f"frame_{step_i:04d}.png"
                imageio.imwrite(str(fname), frame)
                print(f"  Saved {fname.name} ({frame.shape})")
                frames_saved += 1

        if terminated or truncated:
            break

    if frames_saved > 0:
        print(f"[PASS] Saved {frames_saved} frame(s) to results/")
    else:
        print("[WARN] No frames saved.")

    env.close()
    print("\n" + "=" * 60)
    print("Sanity check complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
