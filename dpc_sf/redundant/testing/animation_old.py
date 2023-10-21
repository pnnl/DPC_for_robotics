# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime

import utils

numFrames = 8

import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

# def draw_cylinder(ax, x_center, y_center, radius, depth, resolution=100):
#     x = np.linspace(-radius, radius, resolution) + x_center
#     z = np.linspace(-depth, depth, resolution)
#     X, Z = np.meshgrid(x, z)
#     Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center
# 
#     ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100)
#     ax.plot_surface(X, 2*y_center-Y, Z, alpha=0.5, facecolors='r', rstride=100, cstride=100)
# 
#     floor = Circle((x_center, y_center), radius, color='b')
#     ax.add_patch(floor)
#     art3d.pathpatch_2d_to_3d(floor, z=-depth, zdir="z")
# 
#     ceiling = Circle((x_center, y_center), radius, color='b')
#     ax.add_patch(ceiling)
#     art3d.pathpatch_2d_to_3d(ceiling, z=depth, zdir="z")

def draw_cylinder(ax, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
    z = np.linspace(z_center, z_center + depth, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + x_center
    y_grid = radius*np.sin(theta_grid) + y_center
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)


def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave, obstacle=None):

    x = pos_all[:,0]
    y = pos_all[:,1]
    z = pos_all[:,2]

    xDes = sDes_tr_all[:,0]
    yDes = sDes_tr_all[:,1]
    zDes = sDes_tr_all[:,2]

    x_wp = waypoints[:,0]
    y_wp = waypoints[:,1]
    z_wp = waypoints[:,2]

    # NED rather than ENU
    z = -z
    zDes = -zDes
    z_wp = -z_wp

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    line1, = ax.plot([], [], [], lw=2, color='red')
    line2, = ax.plot([], [], [], lw=2, color='blue')
    line3, = ax.plot([], [], [], '--', lw=1, color='blue')

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange]) # NED

    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    if (xyzType == 0):
        trajType = 'Hover'
    else:
        ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker = 'o', s = 25)
        if (xyzType == 1 or xyzType == 12):
            trajType = 'Simple Waypoints'
        else:
            ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')
            if (xyzType == 2):
                trajType = 'Simple Waypoint Interpolation'
            elif (xyzType == 3):
                trajType = 'Minimum Velocity Trajectory'
            elif (xyzType == 4):
                trajType = 'Minimum Acceleration Trajectory'
            elif (xyzType == 5):
                trajType = 'Minimum Jerk Trajectory'
            elif (xyzType == 6):
                trajType = 'Minimum Snap Trajectory'
            elif (xyzType == 7):
                trajType = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType == 8):
                trajType = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType == 9):
                trajType = 'Minimum Snap Trajectory - Stop'
            elif (xyzType == 10):
                trajType = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType == 1):
                trajType = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'

    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = pos_all[0:i*numFrames,0]
        y_from0 = pos_all[0:i*numFrames,1]
        z_from0 = pos_all[0:i*numFrames,2]
    
        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]
        
        quat = quat_all[i*numFrames]
    
        # NED
        z = -z
        z_from0 = -z_from0
        quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
    
        R = utils.quat2Dcm(quat)    
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 

        # Draw a cylinder at a given location
        draw_cylinder(ax, x_center=1, y_center=1, radius=0.5, depth=2)
        
        line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        line1.set_3d_properties(motorPoints[2,0:3])
        line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        line2.set_3d_properties(motorPoints[2,3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        
        return line1, line2


    def ini_plot():

        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))

        return line1, line2, line3

        
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    if (ifsave):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        line_ani.save(f'gifs/{current_datetime}.gif', dpi=120, fps=25)
        
    # plt.show()
    return line_ani