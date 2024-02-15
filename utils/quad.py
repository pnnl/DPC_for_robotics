"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!

adaptations by: John Viljoen
email: johnviljoen2@gmail.com
"""

from datetime import datetime
import os 

import numpy as np
import torch
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib import animation

import utils.pytorch as ptu
import utils.rotation as rotation

class makeMixerFM:

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    
    @staticmethod
    def pytorch(params):
        dxm = params["dxm"]
        dym = params["dym"]
        kTh = params["kTh"]
        kTo = params["kTo"] 

        return ptu.tensor([[    kTh,      kTh,      kTh,      kTh],
                            [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                            [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                            [   -kTo,      kTo,     -kTo,      kTo]])
    
class applyMixerFM:

    @staticmethod
    def numpy(params, thr, moment):   
        t = np.array([thr, moment[0], moment[1], moment[2]])
        w_cmd = np.sqrt(np.clip(np.dot(params["mixerFMinv"], t), params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd

    @staticmethod
    def pytorch(params, thr, moment):
        t = torch.stack([thr, moment[0], moment[1], moment[2]])
        w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd
    
    @staticmethod
    def pytorch_vectorized(params, thr, moment):
        t = torch.stack([thr, moment[:,0], moment[:,1], moment[:,2]], dim=1)

        # Perform batched matrix multiplication
        w_cmd_sq = torch.bmm(params["mixerFMinv"].unsqueeze(0).expand(t.size(0), -1, -1), t.unsqueeze(-1)).squeeze(-1)

        # Clip and take square root
        w_cmd = torch.sqrt(torch.clamp(w_cmd_sq, min=params["minWmotor"]**2, max=params["maxWmotor"]**2))
        
        # w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
        return w_cmd
    
    @staticmethod
    def casadi(params, thr, moment):
        t = ca.vertcat(thr, moment[:,0], moment[:,1], moment[:,2])

        mixer = ca.DM(ptu.to_numpy(params["mixerFMinv"]))
        # Perform matrix multiplication
        w_cmd_sq = ca.mtimes(mixer, t)

        # Clamp and take square root
        w_cmd_sq_clamped = ca.fmax(ca.fmin(w_cmd_sq, params["maxWmotor"]**2), params["minWmotor"]**2)
        w_cmd = ca.sqrt(w_cmd_sq_clamped)
        
        return w_cmd

def draw_cylinder(ax, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
        z = np.linspace(z_center - depth, z_center + depth, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + x_center
        y_grid = radius*np.sin(theta_grid) + y_center
        return ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')

def calculate_mpc_cost(x_history, u_history, r_history):
    Q = np.diag([1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0])
    R = np.diag([1,1,1,1])
    x_e = r_history - x_history
    return (x_e @ Q @ x_e.T).trace() + (u_history @ R @ u_history.T).trace()

def plot_mujoco_trajectories_wp_p2p(outputs, filename='data/paper/mujoco_trajectories.svg'):
    # Create a new figure for 3D plotting
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    for output in outputs:
        # Plot trajectories
        x = ptu.to_numpy(output['X'][0,1:,0])
        y = ptu.to_numpy(output['X'][0,1:,1])
        z = ptu.to_numpy(output['X'][0,1:,2]) * -1
        ax1.plot(x, y, z, label='Trajectory')
        ax2.plot(x, y, label='Top-down view')

    draw_cylinder(ax1, 1, 1, 0, 0.5, 1, 100)

    # Setting labels and title
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    # ax.set_title('3D Trajectories with Cylinder')

    # Add a red dot at (2, 2, -1)
    ax1.scatter(2, 2, -1, color='red', s=30)

    circle = plt.Circle((1, 1), 0.5, color='r', fill=False)
    ax2.add_patch(circle)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    # ax2.set_title('Top-Down View with Circle')
    ax2.set_aspect('equal', 'box')
    ax2.grid()

    # Add a red dot at (2, 2) for the 2D view
    ax2.scatter(2, 2, color='red', s=30)

    # Adjust spacing between the subplots
    plt.subplots_adjust(wspace=0.65)  # Increase the width spacing

    # Save the figure
    plt.savefig(filename, format='svg')
    plt.close(fig)

    print('fin')

def plot_mujoco_trajectories_wp_traj(outputs, filename='data/paper/mujoco_trajectories.svg'):
    # Create a new figure for 3D plotting
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    for output in outputs:
        # Plot trajectories
        x = ptu.to_numpy(output['X'][0,1:,0])
        y = -ptu.to_numpy(output['X'][0,1:,1])
        z = -ptu.to_numpy(output['X'][0,1:,2])

        xr = ptu.to_numpy(output['R'][0,1:,0])
        yr = -ptu.to_numpy(output['R'][0,1:,1])
        zr = -ptu.to_numpy(output['R'][0,1:,2])

        ax1.plot(x, y, z, label='Trajectory')
        ax1.plot(xr, yr, zr, 'm--',label='Reference')
        ax2.plot(x, y, label='Trajectory')
        ax2.plot(xr, yr, 'm--', label='Reference')

    # Setting labels and title
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.legend()
    # ax.set_title('3D Trajectories with Cylinder')
    ax1.scatter(3, 2, -1, color='red', s=30)
    ax1.scatter(0, 0, 0, color='green', s=30)

    ax2.scatter(3, 2, color='red', s=30)
    ax2.scatter(0, 0, color='green', s=30)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    # ax2.set_title('Top-Down View with Circle')
    ax2.set_aspect('equal', 'box')
    ax2.grid()

    # Adjust spacing between the subplots
    plt.subplots_adjust(wspace=0.65)  # Increase the width spacing

    # Save the figure
    plt.savefig(filename, format='svg')
    plt.close(fig)

    print('fin')

def plot_high_level_trajectories(outputs):

    # Create a new figure for 3D plotting
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    for output in outputs:
        # Plot trajectories
        x = ptu.to_numpy(output['X'][0,:,0])
        y = ptu.to_numpy(output['X'][0,:,2])
        z = ptu.to_numpy(output['X'][0,:,4])
        ax1.plot(x, y, z, label='Trajectory')
        ax2.plot(x, y, label='Top-down view')

    draw_cylinder(ax1, 1, 1, 0, 0.5, 1, 100)

    # Setting labels and title
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    # ax.set_title('3D Trajectories with Cylinder')

    # Add a red dot at (2, 2, 1)
    ax1.scatter(2, 2, 1, color='red', s=30)

    circle = plt.Circle((1, 1), 0.5, color='r', fill=False)
    ax2.add_patch(circle)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    # ax2.set_title('Top-Down View with Circle')
    ax2.set_aspect('equal', 'box')
    ax2.grid()

    # Add a red dot at (2, 2) for the 2D view
    ax2.scatter(2, 2, color='red', s=30)

    # Adjust spacing between the subplots
    plt.subplots_adjust(wspace=0.65)  # Increase the width spacing


    # Save the figure
    plt.savefig('data/paper/high_level_trajectories.svg', format='svg')
    plt.close(fig)

    print('fin')

class Animator:
    def __init__(
            self,
            states,
            times,
            reference_history,
            # reference,
            max_frames = 500,
            drawCylinder = True,
            drawTermSet = False,
            termSetRad = None,
            num_frames = 1,
            dt = 0.1,
            ifsave=True,
            reference_type='wp_traj', # 'wp_traj', 'wp_p2p', 'fig8'
            state_prediction=None,
            save_path='data/media',
            perspective='normal', # 'normal', 'topdown'
        ) -> None:

        num_steps = len(times)
        max_frames = 500
        def compute_render_interval(num_steps, max_frames):
            render_interval = 1  # Start with rendering every frame.
            # While the number of frames using the current render interval exceeds max_frames, double the render interval.
            while num_steps / render_interval > max_frames:
                render_interval *= 2
            return render_interval
        render_interval = compute_render_interval(num_steps, max_frames)
        
        self.cylinder = None
        self.sphere = None
        self.preds = None
        self.cwp = None
        self.termSetRad = termSetRad
        self.drawCylinder = drawCylinder
        self.drawTermSet = drawTermSet
        self.reference_type = reference_type
        if state_prediction is not None:
            self.state_prediction = state_prediction[::render_interval,:]
        else:
            self.state_prediction = state_prediction

        self.states = states[::render_interval,:]
        self.times = times[::render_interval]
        self.reference_history = reference_history[::render_interval,:]
        # self.reference = reference
        self.num_frames = num_frames
        self.dt = dt
        self.ifsave = ifsave
        self.save_path = save_path

        # Unpack States for readability
        # -----------------------------

        self.x = self.states[:,0]
        self.y = self.states[:,1]
        self.z = -self.states[:,2]

        self.q0 = self.states[:,3]
        self.q1 = self.states[:,4]
        self.q2 = self.states[:,5]
        self.q3 = self.states[:,6]

        # Instantiate the figure with title, time, limits...
        # --------------------------------------------------

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        if perspective == 'topdown':
            self.ax.view_init(elev=90, azim=270)

        # draw the lines between the waypoints
        self.xDes = self.reference_history[:, 0]
        self.yDes = self.reference_history[:, 1]
        self.zDes = -self.reference_history[:, 2]
        self.ax.plot(self.xDes, self.yDes, self.zDes, ':', lw=1.3, color='green')

        # these are the lines that draw the quadcopter
        self.line1, = self.ax.plot([], [], [], lw=2, color='red')
        self.line2, = self.ax.plot([], [], [], lw=2, color='blue')
        self.line3, = self.ax.plot([], [], [], '--', lw=1, color='blue')

        # Setting the axes properties
        extraEachSide = 0.5

        # Setting the axes properties
        extraEachSide = 0.5

        x_min = min(np.min(self.x), np.min(self.xDes))
        y_min = min(np.min(self.y), np.min(self.yDes))
        z_min = min(np.min(self.z), np.min(self.zDes))
        x_max = max(np.max(self.x), np.max(self.xDes))
        y_max = max(np.max(self.y), np.max(self.yDes))
        z_max = max(np.max(self.z), np.max(self.zDes))

        maxRange = 0.5*np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() + extraEachSide
        mid_x = 0.5*(x_max+x_min)
        mid_y = 0.5*(y_max+y_min)
        mid_z = 0.5*(z_max+z_min)
        
        self.ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange]) # NED

        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        self.ax.set_zlabel('Altitude')

        self.titleTime = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

        # Unpack Reference for readability
        # --------------------------------

        if reference_type == 'wp_traj' or reference_type == 'wp_p2p':
            self.wp = np.array([
                [0,0,0],
                [2,2,1],
                [-2,3,-3],
                [-2,-1,-3],
                [3,-2,1]
            ])
            x_wp = self.wp[:,0]
            y_wp = self.wp[:,1]
            z_wp = -self.wp[:,2]

            # draw the reference waypoints
            self.ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker = 'o', s = 25)

        title = self.ax.text2D(0.95, 0.95, 'Trajectory', transform=self.ax.transAxes, horizontalalignment='right')

#     def draw_cylinder(self, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
#         z = np.linspace(z_center - depth, z_center + depth, resolution)
#         theta = np.linspace(0, 2*np.pi, resolution)
#         theta_grid, z_grid=np.meshgrid(theta, z)
#         x_grid = radius*np.cos(theta_grid) + x_center
#         y_grid = radius*np.sin(theta_grid) + y_center
#         return self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')
    
    def draw_sphere(self, x_center=0, y_center=0, z_center=0, radius=1, resolution=100):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x_grid = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_grid = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_grid = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center
        return self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')        
        
    def draw_predictions(self, i, state_prediction):
        # Plot the line
        # ax.plot(state_prediction[0,:], state_prediction[1,:], state_prediction[2,:])
        # ax.scatter(state_prediction[0,0], state_prediction[1,0], state_prediction[2,0], marker='x', color='red', label='predicted start point')
        # ax.scatter(state[0], state[1], state[2], marker='x', color='green', label='actual start point')
        # ax.scatter(ref[0], ref[1], ref[2], marker='x', color='black', label='reference point')
        # ax.legend()
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # predicted_x is in the form of (prediction idx, state, timestep idx)
        return self.ax.plot(state_prediction[i,0,:], state_prediction[i,1,:], -state_prediction[i,2,:], color='black')

    def update_lines(self, i):

        # we draw this every self.num_frames
        time = self.times[i * self.num_frames]

        x = self.x[i * self.num_frames]
        y = self.y[i * self.num_frames]
        z = self.z[i * self.num_frames]

        # to draw the history of the line so far we need to retrieve all of it
        x_from0 = self.x[0:i * self.num_frames]
        y_from0 = self.y[0:i * self.num_frames]
        z_from0 = self.z[0:i * self.num_frames]

        # cannot be bothered to import motor positions for this animation
        dxm, dym, dzm = 0.16, 0.16, 0.01

        # retrieve quat in NED frame
        quat = np.array([
            self.q0,
            -self.q1,
            -self.q2,
            self.q3
        ])[:,i * self.num_frames]

        # orient the motors correctly
        R = rotation.quaternion_to_dcm.numpy(quat)
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 

        # if using predictive control, plot the predictions
        if self.state_prediction is not None:

            if self.preds is not None:
                self.preds[0].remove()

            self.preds = self.draw_predictions(i, self.state_prediction)


        # draw red dot on current times waypoint
        if self.cwp is not None:
            self.cwp.remove()

        # retrieve current desired position
        xDes = self.xDes[i * self.num_frames]
        yDes = self.yDes[i * self.num_frames]
        zDes = self.zDes[i * self.num_frames]
        self.cwp = self.ax.scatter(xDes, yDes, zDes, color='magenta', alpha=1, marker = 'o', s = 25)
        
        # remove the old cylinder if it exists
        if self.cylinder is not None:
            self.cylinder.remove()

        # Draw a cylinder at a given location
        if self.drawCylinder:
            self.cylinder = draw_cylinder(self.ax, x_center=1, y_center=1, radius=0.5, depth=2)

        # remove the old cylinder if it exists
        if self.sphere is not None:
            self.sphere.remove()

        # Draw a cylinder at a given location
        if self.drawTermSet:
            self.sphere = self.draw_sphere(x_center=2, y_center=2, z_center=-1, radius=self.termSetRad)

        self.line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        self.line1.set_3d_properties(motorPoints[2,0:3])
        self.line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        self.line2.set_3d_properties(motorPoints[2,3:6])
        self.line3.set_data(x_from0, y_from0)
        self.line3.set_3d_properties(z_from0)
        self.titleTime.set_text(u"Time = {:.2f} s".format(time))

        return self.line1, self.line2
    
    def ini_plot(self):

        self.line1.set_data(np.empty([1]), np.empty([1]))
        self.line1.set_3d_properties(np.empty([1]))
        self.line2.set_data(np.empty([1]), np.empty([1]))
        self.line2.set_3d_properties(np.empty([1]))
        self.line3.set_data(np.empty([1]), np.empty([1]))
        self.line3.set_3d_properties(np.empty([1]))

        return self.line1, self.line2, self.line3     
    
    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig, 
            self.update_lines, 
            init_func=self.ini_plot, 
            # frames=len(self.times[0:-2:self.num_frames]), 
            frames=len(self.times)-1, 
            interval=(self.dt*10), 
            blit=False)

        if (self.ifsave):
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"save path: {os.path.abspath(self.save_path)}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            line_ani.save(f'{self.save_path}/{current_datetime}.gif', dpi=120, fps=25)
            # Update the figure with the last frame of animation
            self.update_lines(len(self.times[1:])-1)
            # Save the final frame as an SVG for good paper plots
            self.fig.savefig(f'{self.save_path}/{current_datetime}_final_frame.svg', format='svg')

        # plt.close(self.fig)            
        # plt.show()
        return line_ani

