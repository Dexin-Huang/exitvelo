"""Render full hit reel with Nathan collision physics + air drag."""

import sys
from pathlib import Path
import numpy as np
import mujoco
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.motion.cmu_replay import CMUMocapReplay
from src.env.contacts import (
    detect_bat_ball_contact,
    integrate_ball_with_drag as ball_step_drag,
    nathan_exit_speed,
    G,
    Q_NATHAN,
)

SCALE = 1.343
ROOT_OFFSET = np.array([0.010, 0.439, 0.0])


def main():
    m = mujoco.MjModel.from_xml_path("assets/mujoco/cmu_batting_scene.xml")
    for i in range(m.nbody):
        m.body_pos[i] *= SCALE
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith(("box_", "home_", "strike_")):
            continue
        m.geom_size[i] *= SCALE
        m.geom_pos[i] *= SCALE
    for i in range(m.nsite):
        m.site_pos[i] *= SCALE
        m.site_size[i] *= SCALE
    for i in range(m.njnt):
        m.jnt_pos[i] *= SCALE
    m.stat.extent = 100.0

    d = mujoco.MjData(m)
    replay = CMUMocapReplay("data/raw/cmu_subject_124/124_07.amc", control_dt=0.01)

    bgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    m.geom_size[bgid][0] = 0.037
    m.geom_rgba[bgid] = [1, 0.2, 0, 1]

    bat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bat")
    rhand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rhand")
    head_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
    sweet_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bat_sweet")

    rarm_joints = ["rhumerusrz", "rhumerusry", "rhumerusrx", "rradiusrx", "rwristry"]
    rarm_qidx = [
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        for jn in rarm_joints
    ]
    rarm_dof = [qi - 1 for qi in rarm_qidx]

    def set_hum(qm):
        d.qpos[0:3] = qm[0:3] * SCALE + ROOT_OFFSET
        d.qpos[3:7] = qm[3:7]
        d.qpos[7:48] = qm[7:48]
        d.qpos[48] = 0
        d.qpos[49:64] = qm[48:63]

    def ik_rh():
        bp = d.xpos[bat_id]
        br = d.xmat[bat_id].reshape(3, 3)
        anch = bp + br @ np.array([-0.04 * SCALE, 0.04 * SCALE, 0.06 * SCALE])
        for _ in range(15):
            mujoco.mj_forward(m, d)
            err = anch - d.xpos[rhand_id]
            if np.linalg.norm(err) < 0.01:
                break
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, rhand_id)
            dq = 0.3 * jacp[:, rarm_dof].T @ err
            for i, qi in enumerate(rarm_qidx):
                d.qpos[qi] += dq[i]

    swing_start = 245

    # Get bat velocity at contact frame
    mujoco.mj_resetData(m, d)
    replay.step_idx = 284
    set_hum(replay.get_qpos())
    mujoco.mj_forward(m, d)
    ik_rh()
    mujoco.mj_forward(m, d)
    sweet_284 = d.site_xpos[sweet_id].copy()

    replay.step_idx = 285
    set_hum(replay.get_qpos())
    mujoco.mj_forward(m, d)
    ik_rh()
    mujoco.mj_forward(m, d)
    sweet_285 = d.site_xpos[sweet_id].copy()
    target = sweet_285.copy()

    bat_vel_vec = (sweet_285 - sweet_284) / 0.01
    bat_speed = np.linalg.norm(bat_vel_vec)
    bat_dir = bat_vel_vec / bat_speed

    # Nathan exit speed
    pitch_speed = 41.6
    exit_speed = (1 + Q_NATHAN) * bat_speed + Q_NATHAN * pitch_speed
    print(f"Bat speed: {bat_speed:.1f} m/s ({bat_speed*2.237:.0f} mph)")
    print(f"Nathan exit speed: {exit_speed:.1f} m/s ({exit_speed*2.237:.0f} mph)")

    # Exit direction: bat direction angled up 28 degrees
    launch_angle = np.radians(28)
    exit_dir = bat_dir.copy()
    exit_dir[2] = np.tan(launch_angle) * np.linalg.norm(exit_dir[:2])
    exit_dir = exit_dir / np.linalg.norm(exit_dir)
    exit_vel = exit_dir * exit_speed
    print(f"Launch angle: {np.degrees(np.arctan2(exit_vel[2], np.linalg.norm(exit_vel[:2]))):.1f} deg")

    # Ball trajectory to target
    t_flight = 0.40
    ball_start = np.array([target[0], target[1] - pitch_speed * t_flight, 2.458])
    ball_vel_init = (target - ball_start - 0.5 * G * t_flight**2) / t_flight

    # Init
    mujoco.mj_resetData(m, d)
    replay.step_idx = swing_start
    set_hum(replay.get_qpos())
    mujoco.mj_forward(m, d)
    ik_rh()
    mujoco.mj_forward(m, d)
    batter_pos = d.xpos[head_id].copy()
    d.qpos[64:67] = ball_start
    d.qpos[67:71] = [1, 0, 0, 0]
    d.qvel[:] = 0

    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    mid = (ball_start + batter_pos) / 2
    cam.lookat[:] = mid
    cam.distance = 16.76 * 0.8
    cam.azimuth = 90
    cam.elevation = -8

    frames = []
    contacted = False
    contact_pos = None
    post_v = None

    # Phase 1: Stance
    for _ in range(60):
        if _ % 3 == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    # Phase 2: Pitch + swing
    csl = cam.lookat.copy()
    csd = cam.distance
    for step in range(60):
        cf = min(max(swing_start + step, 0), replay.n_frames - 1)
        replay.step_idx = cf
        set_hum(replay.get_qpos())
        mujoco.mj_forward(m, d)
        ik_rh()

        t_s = step * 0.01
        if not contacted:
            bp = ball_start + ball_vel_init * t_s + 0.5 * G * t_s**2
            d.qpos[64:67] = bp
            mujoco.mj_forward(m, d)
            c, _ = detect_bat_ball_contact(m, d)
            if c:
                contacted = True
                contact_pos = bp.copy()
                post_v = exit_vel.copy()
                print(f"CONTACT step {step}! exit={exit_speed:.1f} m/s ({exit_speed*2.237:.0f} mph)")
        else:
            # Post-contact: integrate from contact with drag
            dt_post = (step - 40) * 0.01
            if dt_post > 0:
                bp_post = contact_pos.copy()
                bv_post = post_v.copy()
                n_sub = int(dt_post / 0.001)
                for _ in range(n_sub):
                    bp_post, bv_post = ball_step_drag(bp_post, bv_post, 0.001)
                if bp_post[2] < 0.037:
                    bp_post[2] = 0.037
                d.qpos[64:67] = bp_post

        mujoco.mj_forward(m, d)
        tf = min(step / 40, 1.0)
        cam.lookat[:] = csl * (1 - tf) + np.array([0.3, -0.3, 1.2]) * tf
        cam.distance = csd * (1 - tf) + 5 * tf
        if step % 3 == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    # Phase 3: Ball flight with drag + bounce + roll
    if contacted:
        last_qm = replay.get_qpos()
        ball_p = contact_pos.copy()
        ball_v = post_v.copy()
        bounces = 0

        for step in range(5000):
            set_hum(last_qm)
            mujoco.mj_forward(m, d)
            ik_rh()

            for _ in range(10):
                ball_p, ball_v = ball_step_drag(ball_p, ball_v, 0.001)
                if ball_p[2] <= 0.037:
                    ball_p[2] = 0.037
                    if ball_v[2] < -0.3:
                        bounces += 1
                        ball_v[2] = -ball_v[2] * 0.35
                        ball_v[0] *= 0.80
                        ball_v[1] *= 0.80
                    else:
                        ball_v[2] = 0
                        ball_v[0] *= 0.995
                        ball_v[1] *= 0.995

            d.qpos[64:67] = ball_p
            mujoco.mj_forward(m, d)

            a = 0.03
            cam.lookat[:] = cam.lookat[:] * (1 - a) + ball_p * a
            cam.distance = min(80, max(5, cam.distance + 0.06))
            cam.elevation = max(-15, cam.elevation - 0.005)

            if step % 3 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

            if bounces > 0 and np.linalg.norm(ball_v) < 0.15:
                dist = np.linalg.norm(ball_p[:2] - contact_pos[:2])
                dist_ft = dist * 3.281
                print(f"Stopped! {dist:.1f}m ({dist_ft:.0f} ft), {bounces} bounces")
                if dist_ft > 300:
                    print("DINGER!")
                for _ in range(45):
                    renderer.update_scene(d, camera=cam)
                    frames.append(renderer.render().copy())
                break

    renderer.close()
    imageio.mimwrite("results/full_hit_reel.mp4", frames, fps=30)
    print(f"Saved: {len(frames)} frames, {len(frames)/30:.1f}s")


if __name__ == "__main__":
    main()
