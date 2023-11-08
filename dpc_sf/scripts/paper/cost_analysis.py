import numpy as np
import matplotlib.pyplot as plt

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

    return cost

# print(f"cost is: {cost}")

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

def generate_wp_traj_reference(Ti, Ts, Tf):
    ref = waypoint_reference(type='wp_traj', average_vel=1.0)
    r = []
    for t in np.arange(Ti, Tf, Ts):
        r.append(ref(t))
    return np.vstack(r)

dpc_save_dir = "data/dpc_timehistories/"
mpc_save_dir = "data/mpc_timehistories/"

files = [
    dpc_save_dir + "xu_fig8_mj_0.001.npz",
    dpc_save_dir + "xu_p2p_mj_0.001.npz",
    dpc_save_dir + "xu_traj_mj_0.001.npz",
    mpc_save_dir + "xu_fig8_mj_0.001.npz",
    mpc_save_dir + "xu_wp_p2p_mj_0.001.npz",
    mpc_save_dir + "xu_wp_traj_mj_0.001.npz",
]

def plot_wp_p2p_trajectories(array1, array2):
    # Create a new figure
    fig = plt.figure(figsize=(14, 7))

    # Create 3D subplot
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot the trajectories in 3D
    ax1.plot(array1[:, 0], array1[:, 1], -array1[:, 2], label='MPC Trajectory')
    ax1.plot(array2[:, 0], array2[:, 1], -array2[:, 2], label='DPC Trajectory')

    # Set labels for 3D subplot
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')

    # Draw a cylinder in 3D subplot
    x_center, y_center = 1, 1  # Center of the cylinder
    radius = 0.5
    z = np.linspace(-1, 1, 100)  # Updated z-axis limits
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center
    y_grid = radius * np.sin(theta_grid) + y_center
    ax1.plot_surface(x_grid, y_grid, z_grid, color='blue', alpha=0.3)

    # Plot the red dot in 3D subplot
    ax1.scatter(2, 2, -1, color='red', s=70)  # s is the size of the dot

    # Show the legend for 3D subplot
    ax1.legend()

    # Create 2D subplot for top-down view
    ax2 = fig.add_subplot(122)

    # Plot the trajectories in 2D
    ax2.plot(array1[:, 0], array1[:, 1], label='MPC Trajectory')
    ax2.plot(array2[:, 0], array2[:, 1], label='DPC Trajectory')

    # Draw a circle representing the cylinder in 2D subplot
    circle = plt.Circle((x_center, y_center), radius, color='blue', fill=False, alpha=0.3)
    ax2.add_artist(circle)

    ax2.scatter(2, 2, color='red', s=70)  # Only x and y coordinates needed

    # Set labels for 2D subplot
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_aspect('equal', 'box')  # Equal aspect ratio

    # Show the legend for 2D subplot
    ax2.legend()
    ax2.grid()

    # Show the plot
    # Save the plot as an SVG file
    plt.savefig("data/media/paper/wp_p2p_comparison.svg", format='svg')
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook


def compare_wp_p2p(mpc_file, dpc_file):
    dpc_data = np.load(dpc_file)
    dpc_x, dpc_u = dpc_data['x_history'], dpc_data['u_history']
    mpc_data = np.load(mpc_file)
    mpc_x, mpc_u = mpc_data['x_history'], mpc_data['u_history']
    Ti = 0; Ts=0.001; Tf = 5.0
    r = generate_wp_p2p_reference(Ti, Ts, Tf)

    print(f"DPC MPC cost: {calc_p2p_MPC_cost(dpc_x, dpc_u, Ti, Tf)}")
    print(f"MPC MPC cost: {calc_p2p_MPC_cost(mpc_x, mpc_u, Ti, Tf)}")

    plot_wp_p2p_trajectories(mpc_x, dpc_x)

    print('fin')

compare_wp_p2p(
    mpc_save_dir + "xu_wp_p2p_mj_0.001.npz",
    dpc_save_dir + "xu_p2p_mj_0.001.npz",
)

def plot_three_trajectories_in_3d(array1, array2, array3, save_path):
    # Create a new figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectories
    ax.plot(array1[:, 0], -array1[:, 1], -array1[:, 2], label='MPC Trajectory')
    ax.plot(array2[:, 0], -array2[:, 1], -array2[:, 2], label='DPC Trajectory')
    ax.plot(array3[:, 0], -array3[:, 1], -array3[:, 2], label='Reference Trajectory')

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot with a legend
    ax.legend()

    # Save the plot as an SVG file
    plt.savefig(save_path, format='svg')
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook

def calc_wp_traj_cost(x):
    pass

def compare_wp_traj(mpc_file, dpc_file):
    dpc_data = np.load(dpc_file)
    dpc_x, dpc_u = dpc_data['x_history'], dpc_data['u_history']
    mpc_data = np.load(mpc_file)
    mpc_x, mpc_u = mpc_data['x_history'], mpc_data['u_history']
    Ti = 0; Ts=0.001; Tf = 20.0
    r = generate_wp_traj_reference(Ti, Ts, Tf)

    plot_three_trajectories_in_3d(mpc_x, dpc_x, r, "data/media/paper/wp_traj.svg")

compare_wp_traj(
    mpc_save_dir + "xu_wp_traj_mj_0.001.npz",
    dpc_save_dir + "xu_traj_mj_0.001.npz",
)
# for file in files:
#     data = np.load(file)
#     x, u = data['x_history'], data['u_history']
#     Ti = 0; Ts=0.001; Tf = 10.0
#     r = generate_fig8_reference(Ti, Ts, Tf)
# 
print('fin')


