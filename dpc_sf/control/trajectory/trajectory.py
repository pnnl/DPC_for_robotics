import numpy as np
import matplotlib.pyplot as plt
import dpc_sf.utils as utils # for quaternion conversions

class waypoint_reference:
    def __init__(
            self, 
            type='wp_p2p', # 'wp_traj', 'wp_p2p'
            average_vel=1.6,
            include_actuators=True,
            set_vel_zero = True,
        ) -> None:
        
        self.include_actuators=include_actuators
        self.type = type
        self.set_vel_zero = set_vel_zero

        self.wp = np.array([
            [0,0,0],
            [2,2,1],
            [-2,3,-3],
            [-2,-1,-3],
            [3,-2,1]
        ])

        self.desired_yaw_at_wp = np.array([
            0,
            np.arctan2((self.wp[1,1]-self.wp[0,1]),(self.wp[1,0]-self.wp[0,0])),
            np.arctan2((self.wp[2,1]-self.wp[1,1]),(self.wp[2,0]-self.wp[1,0])),
            np.arctan2((self.wp[3,1]-self.wp[2,1]),(self.wp[3,0]-self.wp[2,0])),
            np.arctan2((self.wp[4,1]-self.wp[3,1]),(self.wp[4,0]-self.wp[3,0])),
        ])

        distances = np.array([
            np.linalg.norm(self.wp[1,:] - self.wp[0,:]),
            np.linalg.norm(self.wp[2,:] - self.wp[1,:]),
            np.linalg.norm(self.wp[3,:] - self.wp[2,:]),
            np.linalg.norm(self.wp[4,:] - self.wp[3,:])
        ])

        self.delta_times = np.array([
            distances[0]/average_vel,
            distances[1]/average_vel,
            distances[2]/average_vel,
            distances[3]/average_vel,
        ])

    def __call__(self, t):
        if self.type == 'wp_p2p':
            return self.wp_p2p_ref(t)
        elif self.type == 'wp_traj':
            return self.wp_traj_ref(t)

    def wp_traj_ref(self, t):
        # Calculate cumulative times at each waypoint
        cumulative_times = np.cumsum(np.append(0, self.delta_times))

        # Identify the current waypoint and next waypoint
        current_waypoint_index = np.searchsorted(cumulative_times, t) - 1
        next_waypoint_index = current_waypoint_index + 1

        # Special case: when t is past the last waypoint time, use the last waypoint as both current and next
        if next_waypoint_index >= len(self.wp):
            current_waypoint_index = next_waypoint_index = len(self.wp) - 1

        # Calculate x, y, z, yaw
        x = np.interp(t, cumulative_times, self.wp[:, 0])
        y = np.interp(t, cumulative_times, self.wp[:, 1])
        z = np.interp(t, cumulative_times, self.wp[:, 2])
        yaw = np.interp(t, cumulative_times, self.desired_yaw_at_wp)

        # Calculate xdot, ydot, zdot (velocities)
        # handle special case at end of the timeseries where time_diff becomes zero as we made the indices the same earlier lol
        # there is almost certainly a better way to do this but this way makes sense to me now.
        time_diff = cumulative_times[next_waypoint_index] - cumulative_times[current_waypoint_index]
        if time_diff >= 1e-05:
            xdot = (self.wp[next_waypoint_index, 0] - self.wp[current_waypoint_index, 0]) / time_diff
            ydot = (self.wp[next_waypoint_index, 1] - self.wp[current_waypoint_index, 1]) / time_diff
            zdot = (self.wp[next_waypoint_index, 2] - self.wp[current_waypoint_index, 2]) / time_diff
        else:
            # for when time_diff is zero we assume end of run - velocities = 0
            xdot = 0
            ydot = 0
            zdot = 0

        if self.set_vel_zero is True:
            xdot = 0
            ydot = 0
            zdot = 0

        # quaternion conversion based on yaw. Pitch, roll desired to be 0
        q0, q1, q2, q3 = utils.YPRToQuat_np(yaw, 0, 0)

        # desired angular velocities are all 0
        p, q, r = 0, 0, 0

        # the desired angular velocities of the motors are the same as hover
        wM1, wM2, wM3, wM4 = [522.9847140714692]*4

        if self.include_actuators is True:
            # desired state at the specified time is therefore
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r,
                wM1, wM2, wM3, wM4
            ])
        else:
            # desired state at the specified time is therefore
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r
            ])

        return state

    def wp_p2p_ref(self, t):
        # Calculate cumulative times at each waypoint
        cumulative_times = np.cumsum(np.append(0, self.delta_times))

        # Find which waypoint the time t belongs to
        waypoint_index = np.digitize(t, cumulative_times)

        # Clamp the index to the array size
        waypoint_index = np.clip(waypoint_index, 0, len(self.wp) - 1)

        # Get the position and yaw from the waypoints
        x, y, z = self.wp[waypoint_index]
        try:
            # yaw = self.desired_yaw_at_wp[waypoint_index + 1]
            yaw = self.desired_yaw_at_wp[waypoint_index + 1]
        except:
            yaw = 0
        # yaw = 2 # kill yaw at all times

        # quaternion conversion based on yaw. Pitch, roll desired to be 0
        q0, q1, q2, q3 = utils.YPRToQuat_np(yaw, 0, 0)

        # The velocity and angular velocities are all 0
        xdot, ydot, zdot = 0, 0, 0
        p, q, r = 0, 0, 0

        # the desired angular velocities of the motors are the same as hover
        wM1, wM2, wM3, wM4 = [522.9847140714692]*4

        # desired state at the specified time is therefore
        if self.include_actuators is True:
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r,
                wM1, wM2, wM3, wM4
            ])
        else:
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r
            ])            

        return state

class equation_reference:
    def __init__(
            self,
            type='fig8', # 'fig8', ''
            average_vel=1.6,
            set_vel_zero=False,
            include_actuators=True,
            Ts = 0.001,
        ) -> None:
        
        self.type = type
        self.average_vel = average_vel
        self.set_vel_zero = set_vel_zero
        self.include_actuators=include_actuators
        self.Ts = Ts

    def __call__(self, t):
        if self.type == 'fig8':
            return self.fig8(t)

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
        # z *= -1

        xkp1 = A * np.cos(t+self.Ts)
        ykp1 = B * np.sin(2*(t+self.Ts)) / 2
        zkp1 = C * np.sin(t+self.Ts) + Z
        # zkp1 *= -1

        # Velocities
        xdot = -A * np.sin(t)
        ydot = B * np.cos(2*t)
        zdot = C * np.cos(t)
        # xdot = (xkp1 - x)/self.Ts
        # ydot = (ykp1 - y)/self.Ts
        # zdot = (zkp1 - z)/self.Ts

        # compute the yaw from the velocities
        yaw = np.arctan2(ydot, xdot)

        # Quaternion conversion based on yaw. Pitch, roll desired to be 0
        q0, q1, q2, q3 = utils.YPRToQuat_np(yaw, 0, 0)

        # Angular velocities are all 0
        p, q, r = 0, 0, 0

        # The desired angular velocities of the motors are the same as hover
        wM1, wM2, wM3, wM4 = [522.9847140714692]*4

        # desired state at the specified time is therefore
        # state = np.array([
        #     x, y, z,
        #     q0, q1, q2, q3,
        #     xdot, ydot, zdot,
        #     p, q, r,
        #     wM1, wM2, wM3, wM4
        # ])

        if self.set_vel_zero:
            xdot *= 0
            ydot *= 0
            zdot *= 0

        if self.include_actuators is True:
            # desired state at the specified time is therefore
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r,
                wM1, wM2, wM3, wM4
            ])
        else:
            # desired state at the specified time is therefore
            state = np.array([
                x, y, z,
                q0, q1, q2, q3,
                xdot, ydot, zdot,
                p, q, r
            ])

        return state

    def plot(self, t_range=np.linspace(0, 2*np.pi, 1000), plot_interval=100):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize lists to store the trajectory data
        x, y, z, xdot, ydot, zdot = [], [], [], [], [], []

        # Call the function with one time instance at a time
        for t in t_range:
            state = self.fig8(t)
            x.append(state[0])
            y.append(state[1])
            z.append(state[2])
            xdot.append(state[7])
            ydot.append(state[8])
            zdot.append(state[9])

        # Plot the whole trajectory
        ax.plot(x, y, z, 'b-')

        # Plot markers and arrows at specific points
        for i in range(0, len(t_range), plot_interval):
            ax.scatter(x[i], y[i], z[i], color='red')
            ax.quiver(x[i], y[i], z[i], xdot[i], ydot[i], zdot[i], color='green', length=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


# fig8_ref = equation_reference(type='fig8')
# fig8_ref.plot()

# Call the function to plot
# plot_figure_eight_trajectory()

# print('fin')