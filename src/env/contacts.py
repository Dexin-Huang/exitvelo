import mujoco
import numpy as np


def detect_bat_ball_contact(model, data, *, bat_geom_ids=None, ball_geom_id=None):
    """Check MuJoCo contact array for bat-ball collision pairs.

    Parameters
    ----------
    model, data : MuJoCo model and data objects.
    bat_geom_ids : optional set of pre-cached bat geom IDs.
    ball_geom_id : optional pre-cached ball geom ID.

    If cached IDs are not provided, they are looked up by name (backward compatible).

    Returns (is_contact, contact_info_dict_or_None).
    """
    if bat_geom_ids is None:
        bat_geom_ids = set()
        for name in ['bat_barrel', 'bat_handle', 'bat_taper', 'bat_end', 'bat_knob']:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                bat_geom_ids.add(gid)
    if ball_geom_id is None:
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
