"""
Script to generate numerical comparison between DPC and MPC of:
    - MPC cost
    - time to clear constraint come within a bounds of 0.5m of the obstacle
    - the time to compute
"""
from tqdm import tqdm
import time
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.scripts.e2e_mpc import e2e_test as run_mpc
from dpc_sf.control.dpc_wp_traj.run import run_traj as run_dpc_traj
from dpc_sf.control.safety_filter.run_func import run_p2p as run_dpc_sf_p2p
from dpc_sf.control.safety_filter.run_func import run_p2p_no_sf as run_dpc_p2p
import numpy as np
import matplotlib.pyplot as plt

# Test parameters
# ---------------
mpc_save_dir = 'data/mpc_timehistories/'
dpc_save_dir = 'data/dpc_timehistories/'

# point 2 point parameters
# this script will ignore the 1ms mpc as that takes 100 hours to run
mpc_p2p_save_name       = "mpc_obs_avoid_1ms.npz"
mpc_p2p_alt_save_name   = "mpc_obs_avoid_10ms.npz"
dpc_sf_p2p_save_name    = "dpc_sf_obs_avoid_1ms.npz"
dpc_p2p_save_name       = "dpc_obs_avoid_1ms.npz"
p2p_Ts = 0.001
p2p_Ti = 0.0
p2p_Tf = 5.0 # 3.0
p2p_mpc_prediction_horizon = 200 # 3000
p2p_times = np.arange(p2p_Ti, p2p_Tf, p2p_Ts)

# trajectory reference parameters
mpc_traj_save_name = "mpc_traj_10ms.npz"
dpc_traj_save_name = "dpc_traj_10ms.npz"
dpc_traj_alt_save_name = "dpc_traj_1ms.npz"
traj_Ts = 0.01
traj_Ti = 0.0
traj_Tf = 25.0 # 25.0
traj_mpc_prediction_horizon = 100 # 3000

generate_new_mpc_data = False
generate_new_dpc_data = False

# MPC Run/Load Tests
# ------------------
if generate_new_mpc_data is True:

    # Start the timer
    start_time = time.time()

    print(f"Running MPC tests with prediction horizon of length: {str(traj_mpc_prediction_horizon)}")
    print(f"corresponding to: {str(traj_mpc_prediction_horizon * traj_Ts)}s")

    run_mpc(
        test='wp_p2p', 
        backend='mj',
        save_trajectory = True,
        save_dir = mpc_save_dir,
        save_name = mpc_p2p_save_name,
        plot_prediction = False,
        Ts = p2p_Ts, Ti = p2p_Ti, Tf = p2p_Tf,
        N = p2p_mpc_prediction_horizon
    )

    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for wp_traj: {elapsed_time} seconds")

    # Start the timer
    start_time = time.time()

    run_mpc(
        test='wp_traj', 
        backend='mj',
        save_trajectory = True,
        save_dir = mpc_save_dir,
        plot_prediction = False,
        save_name = mpc_traj_save_name,
        Ts = traj_Ts, Ti = traj_Ti, Tf = traj_Tf,
        N = traj_mpc_prediction_horizon
    )

    # End the timer
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for wp_traj: {elapsed_time} seconds")

elif generate_new_mpc_data is False:
    print("Loading MPC data")

mpc_obs_avoid_data = np.load(mpc_save_dir + mpc_p2p_save_name)
mpc_obs_avoid_alt_data = np.load(mpc_save_dir + mpc_p2p_alt_save_name)
mpc_traj_data = np.load(mpc_save_dir + mpc_traj_save_name)

# DPC Run/Load Tests
# ------------------

if generate_new_dpc_data is True:

    # run_dpc_sf_p2p(
    #     Ts=p2p_Ts,
    #     Ti=p2p_Ti,
    #     Tf=p2p_Tf, 
    #     save_dir=dpc_save_dir       
    # )

    run_dpc_p2p(
        Ts=p2p_Ts,
        Ti=p2p_Ti,
        Tf=p2p_Tf, 
        save_dir=dpc_save_dir            
    )

    run_dpc_traj(
        Ts=traj_Ts,
        Ti=traj_Ti,
        Tf=traj_Tf, 
        save_dir=dpc_save_dir
    )
    # raise NotImplementedError
elif generate_new_dpc_data is False:
    print("Loading DPC data")

dpc_sf_obs_avoid_data = np.load(dpc_save_dir + dpc_sf_p2p_save_name)
dpc_obs_avoid_data = np.load(dpc_save_dir + dpc_p2p_save_name)
dpc_traj_data = np.load(dpc_save_dir + dpc_traj_save_name)
dpc_traj_data_alt = np.load(dpc_save_dir + dpc_traj_alt_save_name)

# Plot Obstacle Avoidance 2D
# --------------------------

def plot_2d_obs(npz_data_list: list, data_labels: list, save_dir: str = None, show: bool = False):
    plt.figure(figsize=(5, 5))

    # Iterate over each npz_data and corresponding label
    for npz_data, label in zip(npz_data_list, data_labels):
        # extract data from the numpy save
        x_history = npz_data['x_history']
        x, y = x_history[:,0], x_history[:,1]

        # Plot the quadcopter's trajectory
        plt.plot(x, y, '-', label=label)

    # Add fixed items to the plot
    plt.plot(2, 2, 'o', label="Reference Waypoint", color="red")
    
    # Plot the circle constraint at (1,1) with radius 0.5
    circle = plt.Circle((1, 1), 0.5, color='black', fill=False, label='Cylinder Obstace')
    plt.gca().add_patch(circle)

    # Plot the circle terminal constraint at (2,2) with a radius of 0.45
    terminal_set = plt.Circle((2, 2), 0.45, color='red', fill=False, label='Terminal Set')
    plt.gca().add_patch(terminal_set)


    # Set plot title and labels
    # plt.title('Top-down 2D Trajectory of Quadcopter')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend(prop={'size':7})
    plt.axis('equal')  # To ensure equal scaling for x and y axes

    # Show the plot
    if show is True:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir, format="svg")
        plt.close()

npz_data_list = [
    # mpc_obs_avoid_data,
    mpc_obs_avoid_alt_data,
    dpc_sf_obs_avoid_data,
    dpc_obs_avoid_data
]

data_labels = [
    # "MPC 1ms",
    "MPC 10ms",
    "DPC + SF 1ms",
    "DPC 1ms"
]

save_dir = dpc_save_dir + "obs_avoid_results.svg"

plot_2d_obs(npz_data_list, data_labels, save_dir)

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

mpc_oa_alt_x_history = mpc_obs_avoid_alt_data['x_history']
mpc_oa_alt_u_history = mpc_obs_avoid_alt_data['u_history']
mpc_oa_x_history = mpc_obs_avoid_data['x_history']
mpc_oa_u_history = mpc_obs_avoid_data['u_history']
dpc_oa_sf_x_history = dpc_sf_obs_avoid_data['x_history']
dpc_oa_sf_u_history = dpc_sf_obs_avoid_data['u_history']
dpc_oa_x_history = dpc_obs_avoid_data['x_history']
dpc_oa_u_history = dpc_obs_avoid_data['u_history']

calc_p2p_MPC_cost(mpc_oa_alt_x_history, mpc_oa_alt_u_history, p2p_Ti, p2p_Tf)
calc_p2p_MPC_cost(mpc_oa_x_history, mpc_oa_u_history, p2p_Ti, p2p_Tf)
calc_p2p_MPC_cost(dpc_oa_sf_x_history, dpc_oa_sf_u_history, p2p_Ti, p2p_Tf)
calc_p2p_MPC_cost(dpc_oa_x_history, dpc_oa_u_history, p2p_Ti, p2p_Tf)

# Calculate Time to Reach Terminal Set
# ------------------------------------
def calc_time2_term_set(x_history, Ti, Tf):
    N = x_history.shape[0]
    times = np.linspace(Ti, Tf, N)

    # calculate distance to terminal set of radius 0.45 at (2,2)
    for k in range(N):
        x = x_history[k,:]
        distance = np.sqrt((x[0]-2)**2+(x[1]-2)**2) - 0.45
        vel = np.linalg.norm(x[7:10])
        if distance < 0.0 and vel < 0.1:
            return times[k]
        
    print("failed to reach terminal set")

time2ReachTerm = calc_time2_term_set(x_history, p2p_Ti, p2p_Tf)


# Plot Trajectory Reference 3D
# ----------------------------

print('fin')
