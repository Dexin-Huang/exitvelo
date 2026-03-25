import mujoco
import numpy as np


def detect_bat_ball_contact(model, data):
    """Check MuJoCo contact array for bat-ball collision pairs.
    Returns (is_contact, contact_info_dict_or_None).
    """
    bat_geom_ids = set()
    for name in ['bat_barrel', 'bat_handle', 'bat_taper', 'bat_end', 'bat_knob']:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            bat_geom_ids.add(gid)
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')

    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if (g1 in bat_geom_ids and g2 == ball_geom_id) or \
           (g2 in bat_geom_ids and g1 == ball_geom_id):
            return True, {
                'pos': contact.pos.copy(),
                'dist': contact.dist,
                'frame': contact.frame.copy(),
            }
    return False, None
