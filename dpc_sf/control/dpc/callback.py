import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dpc_sf.utils import pytorch_utils as ptu
from datetime import datetime
from neuromancer.callbacks import Callback
import torch
import numpy as np
import os
import imageio

def draw_cylinder(ax, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
    z = np.linspace(z_center - depth, z_center + depth, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center
    y_grid = radius * np.sin(theta_grid) + y_center
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')

def plot_wp_p2p(trajectories, save_path=None):
    # Extract x and y
    x_positions = trajectories['X'][0, :, 0]
    y_positions = trajectories['X'][0, :, 2]
    # Convert the PyTorch tensors to numpy arrays
    x_np = ptu.to_numpy(x_positions)
    y_np = ptu.to_numpy(y_positions)
    fig, ax = plt.subplots()
    # Plot the trajectory
    ax.plot(x_np, y_np)
    # Draw a circle at (1,1) with a radius of 0.5
    circle = Circle((1, 1), 0.5, fill=False)  # fill=False makes the circle hollow. Change to True if you want it filled.
    ax.add_patch(circle)
    # Plot a red dot at the point (2,2)
    ax.scatter(2, 2, color='red')
    # Setting equal scaling and showing the grid:
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.

def plot_wp_p2p_train(output, test_trajectory, save_path=None):
    # Extract x and y
    x_positions = ptu.to_numpy(output['train_X'][:, :, 0])
    y_positions = ptu.to_numpy(output['train_X'][:, :, 2])
    x_ref = ptu.to_numpy(output['train_R'][:,-1,0])
    y_ref = ptu.to_numpy(output['train_R'][:,-1,2])
    try:
        x_cyl = ptu.to_numpy(output['train_Cyl'][:,0,0])
        y_cyl = ptu.to_numpy(output['train_Cyl'][:,0,1])
        cylinder = True
    except:
        cylinder = False
    num_lines = x_positions.shape[0]

    # Choose a color map
    colormap = plt.get_cmap('rainbow')

    # Create color iterator
    colors = [colormap(i) for i in np.linspace(0, 1, num_lines)]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Changed to 2x2 layout
    
    # Define z_upper and z_lower for 3D plots and cylinder drawing
    z_upper = 2
    z_lower = -2

    # The First 3D plot
    axs[0, 0] = fig.add_subplot(2, 2, 1, projection='3d')  # Set the first subplot as 3D
    draw_cylinder(axs[0, 0], x_center=1, y_center=1, radius=0.5, depth=(z_upper - z_lower) / 2, z_center=(z_upper + z_lower) / 2)
    axs[0,0].scatter(2,2,1, color='red')
    for idx, color in enumerate(colors):
        # Existing plotting for 2D leftmost plot remains the same for axs[0, 1]
        axs[0, 1].plot(x_positions[idx], y_positions[idx], color=color)
        # ... (rest of the existing 2D plotting)
        # Draw a circle at (1,1) with a radius of 0.5
        if cylinder is True:
            circle = Circle((x_cyl[idx], y_cyl[idx]), 0.5, fill=False, color=color)  # fill=False makes the circle hollow. Change to True if you want it filled.
            axs[0, 1].add_patch(circle)
        # Plot a dot at the end_point, start point
        axs[0, 1].scatter(x_ref[idx], y_ref[idx], color=color)
        axs[0, 1].scatter(x_positions[idx,0], y_positions[idx,0], color=color)
        # Setting equal scaling and showing the grid:
        axs[0, 1].set_aspect('equal', 'box')
        axs[0, 1].grid(True)
        # Extracting and converting for 3D plotting in axs[0, 0]
        z_positions = ptu.to_numpy(output['train_X'][:, :, 4])  # Assuming the 4th index is for z
        axs[0, 0].plot(x_positions[idx], y_positions[idx], z_positions[idx], color=color)
    
    # Setting the axs[1, 0] as 3D subplot for the third plot
    axs[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
    draw_cylinder(axs[1, 0], x_center=1, y_center=1, radius=0.5, depth=(z_upper - z_lower) / 2, z_center=(z_upper + z_lower) / 2)
    
    num_test_lines = test_trajectory['X'].shape[0]

    # Continue plotting in the existing third plot (now axs[1, 0])
    for idx in range(num_test_lines):
        # ... [previous axs[2] plotting logic, just replace axs[2] with axs[1, 0]]
        # Plot the 3D trajectory in axs[2]
        x_positions = test_trajectory['X'][idx, :, 0]
        y_positions = test_trajectory['X'][idx, :, 2]
        z_positions = -test_trajectory['X'][idx, :, 4]
        # Convert the PyTorch tensors to numpy arrays
        x_np = ptu.to_numpy(x_positions)
        y_np = ptu.to_numpy(y_positions)
        z_np = ptu.to_numpy(z_positions)

        axs[1,0].plot(x_np, y_np, z_np, color='blue')

        axs[1,0].set_xlim(-1.2, 2.2)
        axs[1,0].set_ylim(-1.2, 2.2)
        axs[1,0].set_zlim(-2, 2)
        axs[1,0].scatter(2,2,-1, color='red')

    # 2D Plotting for the second plot (axs[0, 1])
    for idx in range(num_test_lines):
        # ... [previous axs[1] plotting logic, just replace axs[1] with axs[0, 1]]
        x_positions = test_trajectory['X'][idx, :, 0]
        y_positions = test_trajectory['X'][idx, :, 2]
        z_positions = -test_trajectory['X'][idx, :, 4]
        # Convert the PyTorch tensors to numpy arrays
        x_np = ptu.to_numpy(x_positions)
        y_np = ptu.to_numpy(y_positions)
        z_np = ptu.to_numpy(z_positions)

        axs[1,1].plot(x_np, y_np, color='blue')
        if cylinder is True:
            # Draw a circle at (1,1) with a radius of 0.5
            circle = Circle((1, 1), 0.5, fill=False, color='black')  # fill=False makes the circle hollow. Change to True if you want it filled.
            axs[1,1].add_patch(circle)
        # Plot a red dot at the point (2,2)
        axs[1,1].scatter(2, 2, color='red')
        # Setting equal scaling and showing the grid:
        axs[1,1].set_aspect('equal', 'box')
        axs[1,1].grid(True)
        # Setting the axs[2] as 3D subplot
        
    # Adjust the layout
    plt.tight_layout()
    # if save_path is not None:
    #     plt.savefig(save_path)
    # plt.show()

    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.

def plot_wp_p2p_train_old(output, test_trajectory, save_path=None):
    # Extract x and y
    x_positions = ptu.to_numpy(output['train_X'][:, :, 0])
    y_positions = ptu.to_numpy(output['train_X'][:, :, 2])
    x_ref = ptu.to_numpy(output['train_R'][:,-1,0])
    y_ref = ptu.to_numpy(output['train_R'][:,-1,2])
    try:
        x_cyl = ptu.to_numpy(output['train_Cyl'][:,0,0])
        y_cyl = ptu.to_numpy(output['train_Cyl'][:,0,1])
        cylinder = True
    except:
        cylinder = False
    num_lines = x_positions.shape[0]

    # Choose a color map
    colormap = plt.get_cmap('rainbow')

    # Create color iterator
    colors = [colormap(i) for i in np.linspace(0, 1, num_lines)]

    fig, axs = plt.subplots(1,3,figsize=(15,5))
    for idx, color in enumerate(colors):
        # Plot the trajectory
        axs[0].plot(x_positions[idx], y_positions[idx], color=color)
        # Draw a circle at (1,1) with a radius of 0.5
        if cylinder is True:
            circle = Circle((x_cyl[idx], y_cyl[idx]), 0.5, fill=False, color=color)  # fill=False makes the circle hollow. Change to True if you want it filled.
            axs[0].add_patch(circle)
        # Plot a dot at the end_point, start point
        axs[0].scatter(x_ref[idx], y_ref[idx], color=color)
        axs[0].scatter(x_positions[idx,0], y_positions[idx,0], color=color)
        # Setting equal scaling and showing the grid:
        axs[0].set_aspect('equal', 'box')
        axs[0].grid(True)



    num_test_lines = test_trajectory['X'].shape[0]
    axs[2] = fig.add_subplot(1, 3, 3, projection='3d')
    # Draw the cylinder in the 3D subplot
    z_upper = 2
    z_lower = -2
    draw_cylinder(axs[2], x_center=1, y_center=1, radius=0.5, depth=(z_upper - z_lower) / 2, z_center=(z_upper + z_lower) / 2)
 
    for idx in range(num_test_lines):
        # plot the test trajectory next to the old one;
        # Extract x and y
        x_positions = test_trajectory['X'][idx, :, 0]
        y_positions = test_trajectory['X'][idx, :, 2]
        z_positions = -test_trajectory['X'][idx, :, 4]
        # Convert the PyTorch tensors to numpy arrays
        x_np = ptu.to_numpy(x_positions)
        y_np = ptu.to_numpy(y_positions)
        z_np = ptu.to_numpy(z_positions)

        # Plot the trajectory
        axs[1].plot(x_np, y_np, color='blue')
        if cylinder is True:
            # Draw a circle at (1,1) with a radius of 0.5
            circle = Circle((1, 1), 0.5, fill=False, color='black')  # fill=False makes the circle hollow. Change to True if you want it filled.
            axs[1].add_patch(circle)
        # Plot a red dot at the point (2,2)
        axs[1].scatter(2, 2, color='red')
        # Setting equal scaling and showing the grid:
        axs[1].set_aspect('equal', 'box')
        axs[1].grid(True)
        # Setting the axs[2] as 3D subplot
        
        # Plot the 3D trajectory in axs[2]
        axs[2].plot(x_np, y_np, z_np, color='blue')

        axs[2].set_xlim(-1.2, 2.2)
        axs[2].set_ylim(-1.2, 2.2)
        axs[2].set_zlim(-2, 2)
        axs[2].scatter(2,2,-1, color='red')


    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.


def plot_p2p_traj(output, trajectories, save_path):

    R = ptu.to_numpy(output['train_R'])
    X = ptu.to_numpy(output['train_X'])

    x = X[:,:,0]
    y = X[:,:,2]
    z = X[:,:,4]
    xr = R[:,:,0]
    yr = R[:,:,2]
    zr = R[:,:,4]

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='3d')  # 121: 1x2 grid, first subplot, 3D projection
    
    num_lines = x.shape[0]
    cmap = plt.get_cmap('rainbow', num_lines)
    
    for i in range(num_lines):
        ax.plot(x[i, :], y[i, :], z[i, :], color=cmap(i), label=f'Line {i+1}')  # plot x-y-z line
        ax.plot(xr[i, :], yr[i, :], zr[i, :], color=cmap(i), linestyle='dashed')  # plot xr-yr-zr line with dashed style
    
    ax.set_title('3D Line pairs plotted with rainbow cmap')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Plot the test trajectories rather than the training data
    # --------------------------

    R = ptu.to_numpy(trajectories['R'])
    X = ptu.to_numpy(trajectories['X'])

    x = X[:,:,0]
    y = X[:,:,2]
    z = X[:,:,4]
    xr = R[:,:,0]
    yr = R[:,:,2]
    zr = R[:,:,4]

    # Setting up the second subplot (Empty for now)
    ax2 = fig.add_subplot(122, projection='3d')  # 122: 1x2 grid, second subplot
    ax2.set_title('Second Subplot')

    for i in range(1):
        ax2.plot(x[i, :], y[i, :], z[i, :], color='blue', label=f'Line {i+1}')  # plot x-y-z line
        ax2.plot(xr[i, :], yr[i, :], zr[i, :], color='green', linestyle='dashed')  # plot xr-yr-zr line with dashed style
    
    ax2.set_title('3D Line pairs plotted with rainbow cmap')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')

    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.

def plot_fig8_traj(output, trajectories, save_path):

    R = ptu.to_numpy(output['train_R'])
    X = ptu.to_numpy(output['train_X'])

    x = X[:,:,0]
    y = X[:,:,2]
    z = X[:,:,4]
    xr = R[:,:,0]
    yr = R[:,:,2]
    zr = R[:,:,4]

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='3d')  # 121: 1x2 grid, first subplot, 3D projection
    
    num_lines = x.shape[0]
    cmap = plt.get_cmap('rainbow', num_lines)
    
    for i in range(num_lines):
        ax.plot(x[i, :], y[i, :], z[i, :], color=cmap(i), label=f'Line {i+1}')  # plot x-y-z line
        ax.plot(xr[i, :], yr[i, :], zr[i, :], color=cmap(i), linestyle='dashed')  # plot xr-yr-zr line with dashed style
    
    ax.set_title('3D Line pairs plotted with rainbow cmap')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Plot the test trajectories rather than the training data
    # --------------------------

    R = ptu.to_numpy(trajectories['R'])
    X = ptu.to_numpy(trajectories['X'])

    x = X[:,:,0]
    y = X[:,:,2]
    z = X[:,:,4]
    xr = R[:,:,0]
    yr = R[:,:,2]
    zr = R[:,:,4]

    # Setting up the second subplot (Empty for now)
    ax2 = fig.add_subplot(122, projection='3d')  # 122: 1x2 grid, second subplot
    ax2.set_title('Second Subplot')

    for i in range(num_lines):
        ax2.plot(x[i, :], y[i, :], z[i, :], color='blue', label=f'Line {i+1}')  # plot x-y-z line
        ax2.plot(xr[i, :], yr[i, :], zr[i, :], color='green', linestyle='dashed')  # plot xr-yr-zr line with dashed style
    
    ax2.set_title('3D Line pairs plotted with rainbow cmap')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')

    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.


def plot_traj_old(output, trajectories, save_path):

    # Extract x and y
    x_positions = trajectories['X'][0, :, 0]
    y_positions = trajectories['X'][0, :, 2]
    z_positions = trajectories['X'][0, :, 4]
    x_references = trajectories['R'][0, :, 0]
    y_references = trajectories['R'][0, :, 2]
    z_references = trajectories['R'][0, :, 4]

    # Convert the PyTorch tensors to numpy arrays
    x_np = ptu.to_numpy(x_positions)
    y_np = ptu.to_numpy(y_positions)
    z_np = ptu.to_numpy(z_positions)
    xr_np = ptu.to_numpy(x_references)
    yr_np = ptu.to_numpy(y_references)
    zr_np = ptu.to_numpy(z_references)

    # Create the main figure
    fig = plt.figure(figsize=(15, 5))

    # 3D subplot
    ax1 = fig.add_subplot(131, projection='3d')  # 1 row, 3 columns, first plot
    ax1.plot(x_np, y_np, z_np, label='Positions', color='blue')
    ax1.plot(xr_np, yr_np, zr_np, label='References', color='red')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectories')

    # 2D subplot for X-Y plane
    ax2 = fig.add_subplot(132)  # 1 row, 3 columns, second plot
    ax2.plot(x_np, y_np, label='Positions', color='blue')
    ax2.plot(xr_np, yr_np, label='References', color='red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Plane')

    # 2D subplot for Y-Z plane
    ax3 = fig.add_subplot(133)  # 1 row, 3 columns, third plot
    ax3.plot(y_np, z_np, label='Positions', color='blue')
    ax3.plot(yr_np, zr_np, label='References', color='red')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.set_title('Y-Z Plane')

    plt.tight_layout()

    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.

class SinTrajCallback(Callback):
    def __init__(self, save_dir, media_path, nstep, nx, Ts):
        super().__init__()
        self.directory = media_path + save_dir + '/'
        self.nstep = nstep
        self.nx = nx
        self.Ts = Ts
        self.average_vel = 1.0

    def fig8(self, t, A=4, B=4, C=4, Z=-5):
        """ 
        Generate a 3D figure-eight trajectory with full state.
        
        Parameters:
        t: Time variable
        A, B, C: Scaling factors for the x, y and z coordinates respectively
        Z: The z-plane at which the figure-eight lies
        
        Returns:
        x, y, z coordinates, quaternion, velocities, angular velocities, and motor angular velocities
        """

        # accelerate or decelerate time based on velocity desired
        t *= self.average_vel

        # Position
        x = A * np.cos(t)
        y = B * np.sin(2*t) / 2
        z = C * np.sin(t) + Z  # z oscillates around the plane Z

        # Velocities
        xdot = -A * np.sin(t)
        ydot = B * np.cos(2*t)
        zdot = C * np.cos(t)

        return ptu.from_numpy(np.array([x, xdot, y, ydot, z, zdot]))

    def begin_eval(self, trainer, output):

        n = 10

        # need a reference to track that we care about - lets get the pringle
        # Calculate the total time
        T_total = self.nstep * self.Ts
        t = 0.0
        r = []
        while t <= T_total:
            r.append(self.fig8(t))
            t += self.Ts
        r = torch.vstack(r).unsqueeze(0)
        R = torch.cat([r]*n, dim=0)

        X = torch.zeros(n, 1, self.nx, dtype=torch.float32)
        
        # Generating n random radii and thetas for x and y
        radii = 4 * torch.sqrt(torch.rand(n))  # sqrt is used to ensure uniform sampling within the circle
        thetas = 2 * np.pi * torch.rand(n)
        
        X[:, 0, 0] = radii * torch.cos(thetas)  # x
        X[:, 0, 1] = torch.zeros(n)  # xdot, assuming velocity as 0
        X[:, 0, 2] = radii * torch.sin(thetas)  # y
        X[:, 0, 3] = torch.zeros(n)  # ydot, assuming velocity as 0
        X[:, 0, 4] = (-1 * torch.rand(n)) * 8  # z, random between [-12, 0]
        X[:, 0, 5] = torch.zeros(n)  # zdot, assuming velocity as 0

        P = torch.cat([torch.concat([torch.tensor([[[4,4,4,-5]]])] * (self.nstep + 1), dim=1)]*n, dim=0)

        data = {
            'X': X, 
            'R': R,
            'P': P
        }

        trajectories = trainer.model.nodes[0](data)

        plot_fig8_traj(output, trajectories, save_path=self.directory)
        plt.close()

    def animate(self):
        # Gather all the PNG files
        filenames = sorted([f for f in os.listdir(self.directory) if f.endswith('.png')])

        # Convert the PNGs to GIF
        with imageio.get_writer(self.directory + 'animation.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.directory + filename)
                writer.append_data(image)

    def delete_all_but_last_image(self):
        # List all .png files in the directory
        all_files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.png')]

        # Sort the files (they'll be sorted by date/time due to the filename format)
        sorted_files = sorted(all_files)

        # Delete all but the last one
        for file in sorted_files[:-1]:  # Exclude the last file
            os.remove(os.path.join(self.directory, file))

class LinTrajCallback(Callback):
    def __init__(self, save_dir, media_path, nstep, nx, Ts):
        super().__init__()
        self.directory = media_path + save_dir + '/'
        self.nstep = nstep
        self.nx = nx
        self.Ts = Ts

    def begin_eval(self, trainer, output):

        # lets call the current trained model on the data we are interested in
        ref_start_point = torch.tensor([-2., 0, 2, 0, -2, 0])
        ref_end_point = torch.tensor([2., 0, 2, 0, 2, 0])

        # Extract displacements in x, y, and z
        delta_x = ref_end_point[0] - ref_start_point[0]
        delta_y = ref_end_point[2] - ref_start_point[2]
        delta_z = ref_end_point[4] - ref_start_point[4]

        # Calculate the total time
        T_total = self.nstep * self.Ts

        # Calculate constant velocities in x, y, and z directions
        xdot = delta_x / T_total
        ydot = delta_y / T_total
        zdot = delta_z / T_total    

        ref_start_point[1], ref_start_point[3], ref_start_point[5] = xdot, ydot, zdot
        ref_end_point[1], ref_end_point[3], ref_end_point[5] = xdot, ydot, zdot

        # Construct a linspace for each dimension and stack them together
        linspace_tensors = [torch.linspace(start, end, steps=self.nstep+1) for start, end in zip(ref_start_point, ref_end_point)]
        linspace_result = torch.stack(linspace_tensors, dim=-1)  # Shape: (self.nstep+1, 6)

        # Add the necessary dimensions to match the shape of original tensor
        linspace_result = linspace_result.unsqueeze(0)  # Shape: (1, self.nstep+1, 6)

        data = {
            'X': torch.zeros(1, 1, self.nx, dtype=torch.float32), 
            'R': linspace_result 
        }
        
        data = {key: value.to(ptu.device) for key, value in data.items()}

        trajectories = trainer.model.nodes[0](data)

        plot_p2p_traj(output, trajectories, save_path=self.directory)
        plt.close()

    def animate(self):
        # Gather all the PNG files
        filenames = sorted([f for f in os.listdir(self.directory) if f.endswith('.png')])

        # Convert the PNGs to GIF
        with imageio.get_writer(self.directory + 'animation.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.directory + filename)
                writer.append_data(image)

    def delete_all_but_last_image(self):
        # List all .png files in the directory
        all_files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.png')]

        # Sort the files (they'll be sorted by date/time due to the filename format)
        sorted_files = sorted(all_files)

        # Delete all but the last one
        for file in sorted_files[:-1]:  # Exclude the last file
            os.remove(os.path.join(self.directory, file))

class WP_Callback(Callback):
    def __init__(self, save_dir, media_path, nstep, nx):
        super().__init__()
        self.directory = media_path + save_dir + '/'
        self.nstep = nstep
        self.nx = nx

    def begin_eval(self, trainer, output):

        # lets call the current trained model on the data we are interested in
        x_lower = -1  # Replace with your actual lower bound
        x_upper = 1   # Replace with your actual upper bound
        n = 9  # Replace with your actual number of points

        # Generate n linearly spaced x values between x_lower and x_upper
        x_values = torch.linspace(x_lower, x_upper, n)
        # Calculate corresponding y values using the line equation y = -x
        y_values = -x_values
        z_values = torch.zeros(n)
        xdot_values = torch.zeros(n)
        ydot_values = torch.zeros(n)
        zdot_values = torch.zeros(n)
        points = torch.stack((x_values, xdot_values, y_values, ydot_values, z_values, zdot_values), dim=-1)
        X = points.unsqueeze(1)  # Example to make it (1, 1, n, 2)
        
        R = torch.cat([torch.cat([torch.tensor([[[2, 0, 2, 0, 1, 0]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Cyl = torch.cat([torch.cat([torch.tensor([[[1,1]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Idx = torch.cat([torch.vstack([torch.tensor([0.0])]).unsqueeze(1)]*n, dim=0)
        M = torch.cat([torch.ones([1, 1, 1])]*n, dim=0)

        data = {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': Idx,
            'M': M
        }

        data = {key: value.to(ptu.device) for key, value in data.items()}

        test_trajectory = trainer.model.nodes[0](data)

        # plot_wp_p2p(trajectories, save_path=self.directory)]        
        plot_wp_p2p_train(output, test_trajectory, save_path=self.directory)
        plt.close()
    
    def animate(self):
        # Gather all the PNG files
        filenames = sorted([f for f in os.listdir(self.directory) if f.endswith('.png')])

        # Convert the PNGs to GIF
        with imageio.get_writer(self.directory + 'animation.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.directory + filename)
                writer.append_data(image)

    def delete_all_but_last_image(self):
        # List all .png files in the directory
        all_files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.png')]

        # Sort the files (they'll be sorted by date/time due to the filename format)
        sorted_files = sorted(all_files)

        # Delete all but the last one
        for file in sorted_files[:-1]:  # Exclude the last file
            os.remove(os.path.join(self.directory, file))
