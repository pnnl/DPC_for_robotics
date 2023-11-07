import numpy as np
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import equation_reference, waypoint_reference
# Calculate the MPC Cost For Trajectories
# ---------------------------------------
def calc_p2p_MPC_cost(x_history, u_history, Ti, Tf):

    N = x_history.shape[0]
    times = np.linspace(Ti, Tf, N)

    Q = np.zeros([17,17])
    Q[0,0] =   1 # x
    Q[1,1] =   1 # y
    Q[2,2] =   1 # z
    Q[3,3] =   0 # q0
    Q[4,4] =   0 # q1
    Q[5,5] =   0 # q2
    Q[6,6] =   0 # q3
    Q[7,7] =   1 # xdot
    Q[8,8] =   1 # ydot
    Q[9,9] =   1 # zdot
    Q[10,10] = 1 # p
    Q[11,11] = 1 # q
    Q[12,12] = 1 # r
    Q[13,13] = 0 # wM1
    Q[14,14] = 0 # wM2
    Q[15,15] = 0 # wM3
    Q[16,16] = 0 # wM4

    R = np.eye(4)

    r = quad_params["default_init_state_np"]
    r[0] = 2
    r[1] = 2
    r[3] = 1
    r_history = np.vstack([r]*N)

    state_error = r_history - x_history
    cost = np.array(0.)

    for k in range(N-1):
        timestep_input = u_history[k,:]
        timestep_state_error = state_error[k,:]
        cost += (timestep_state_error.T @ Q @ timestep_state_error + timestep_input.T @ R @ timestep_input)

    print(f"cost is: {cost}")

def generate_fig8_reference(Ti, Ts, Tf):
    ref = equation_reference(type='fig8', average_vel=1.0, Ts=Ts)
    r = []
    for t in np.arange(Ti, Tf, Ts):
        r.append(ref(t))
    return np.vstack(r)

def generate_wp_p2p_reference(Ti, Ts, Tf):
    ref = waypoint_reference(type='wp_p2p', average_vel=0.1)
    r = []
    for t in np.arange(Ti, Tf, Ts):
        r.append(ref(t))
    return np.vstack(r)

def generate_wp_p2p_reference(Ti, Ts, Tf):
    ref = waypoint_reference(type='wp_traj', average_vel=0.1)
    r = []
    for t in np.arange(Ti, Tf, Ts):
        r.append(ref(t))
    return np.vstack(r)

dpc_save_dir = "data/dpc_timehistories/"
mpc_save_dir = "data/mpc_timehistories/"

files = [
    # dpc_save_dir + "xu_fig8_mj_0.001.npz",
    dpc_save_dir + "xu_p2p_mj_0.001.npz",
    dpc_save_dir + "xu_traj_mj_0.001.npz",
    mpc_save_dir + "xu_fig8_mj_0.001.npz",
    mpc_save_dir + "xu_wp_p2p_mj_0.001.npz",
    mpc_save_dir + "xu_wp_traj_mj_0.001.npz",
]

for file in files:
    data = np.load(file)
    x, u = data['x_history'], data['u_history']
    Ti = 0; Ts=0.001; Tf = 10.0
    r = generate_fig8_reference(Ti, Ts, Tf)

    print('fin')


