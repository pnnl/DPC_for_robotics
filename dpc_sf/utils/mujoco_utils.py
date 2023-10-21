import numpy as np

def mj_get_state(data, omegas):
    # generalised positions/velocities not in right coordinates
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    qpos[1] *= -1 # y
    qpos[2] *= -1 # z
    qpos[5] *= -1 # q2

    qvel[1] *= -1 # ydot
    qvel[2] *= -1 # zdot
    qvel[4] *= -1 # qdot

    return np.concatenate([qpos, qvel, omegas]).flatten()

def state2qpv(state):
    qpos = np.zeros(len(state))

    qpos = state.squeeze()[0:7]
    qpos[1] *= -1 # y
    qpos[2] *= -1 # z
    qpos[5] *= -1 # q2

    qvel = state.squeeze()[7:13]
    qvel[1] *= -1 # ydot
    qvel[2] *= -1 # zdot
    qvel[4] *= -1 # qdot

    return qpos, qvel
    