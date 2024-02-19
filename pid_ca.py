"""
Adapting classical_control2_func to recieve (batch, state)

COMPLETE - asides from the velocity saturation
"""

import math
# import dpc_sf.utils as utils
import utils.rotation
import utils.quad
import casadi as ca
# from dpc_sf.dynamics.eom_pt import state_dot_nm, state_dot_pt
# from dpc_sf.dynamics.params import params as quad_params
# from dpc_sf.control.trajectory.trajectory import waypoint_reference

from dynamics import get_quad_params
quad_params = get_quad_params()

rad2deg = 180.0/math.pi
deg2rad = math.pi/180.0

ctrl_params = {}

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
ctrl_params["Py"]    = 1.0
ctrl_params["Px"]    = ctrl_params["Py"]
ctrl_params["Pz"]    = 1.0

ctrl_params["pos_P_gain"] = ca.DM([ctrl_params["Px"], ctrl_params["Py"], ctrl_params["Pz"]])

# Velocity P-D gains
ctrl_params["Pxdot"] = 5.0
ctrl_params["Dxdot"] = 0.5
ctrl_params["Ixdot"] = 5.0

ctrl_params["Pydot"] = ctrl_params["Pxdot"]
ctrl_params["Dydot"] = ctrl_params["Dxdot"]
ctrl_params["Iydot"] = ctrl_params["Ixdot"]

ctrl_params["Pzdot"] = 4.0
ctrl_params["Dzdot"] = 0.5
ctrl_params["Izdot"] = 5.0

ctrl_params["vel_P_gain"] = ca.DM([ctrl_params["Pxdot"], ctrl_params["Pydot"], ctrl_params["Pzdot"]])
ctrl_params["vel_D_gain"] = ca.DM([ctrl_params["Dxdot"], ctrl_params["Dydot"], ctrl_params["Dzdot"]])
ctrl_params["vel_I_gain"] = ca.DM([ctrl_params["Ixdot"], ctrl_params["Iydot"], ctrl_params["Izdot"]])

# Attitude P gains
ctrl_params["Pphi"] = 8.0
ctrl_params["Ptheta"] = ctrl_params["Pphi"]
ctrl_params["Ppsi"] = 1.5
ctrl_params["PpsiStrong"] = 8

ctrl_params["att_P_gain"] = ca.DM([ctrl_params["Pphi"], ctrl_params["Ptheta"], ctrl_params["Ppsi"]])

# Rate P-D gains
ctrl_params["Pp"] = 1.5
ctrl_params["Dp"] = 0.04
ctrl_params["Pq"] = ctrl_params["Pp"]
ctrl_params["Dq"] = ctrl_params["Dp"] 
ctrl_params["Pr"] = 1.0
ctrl_params["Dr"] = 0.1

ctrl_params["rate_P_gain"] = ca.DM([ctrl_params["Pp"], ctrl_params["Pq"], ctrl_params["Pr"]])
ctrl_params["rate_D_gain"] = ca.DM([ctrl_params["Dp"], ctrl_params["Dq"], ctrl_params["Dr"]])

# Max Velocities
ctrl_params["uMax"] = 5.0
ctrl_params["vMax"] = 5.0
ctrl_params["wMax"] = 5.0

ctrl_params["velMax"] = ca.DM([ctrl_params["uMax"], ctrl_params["vMax"], ctrl_params["wMax"]])
ctrl_params["velMaxAll"] = 5.0

ctrl_params["saturateVel_separetely"] = False

# Max tilt
ctrl_params["tiltMax"] = 50.0*deg2rad

# Max Rate
ctrl_params["pMax"] = 200.0*deg2rad
ctrl_params["qMax"] = 200.0*deg2rad
ctrl_params["rMax"] = 150.0*deg2rad

ctrl_params["rateMax"] = ca.DM([ctrl_params["pMax"], ctrl_params["qMax"], ctrl_params["rMax"]])
roll_pitch_gain = 0.5*(ctrl_params["att_P_gain"][0] + ctrl_params["att_P_gain"][1])

# assumed yaw_w to be 1 to allow for much better gradients
# ctrl_params["yaw_w"] = np.clip(ctrl_params["att_P_gain"][2]/roll_pitch_gain, 0.0, 1.0)
ctrl_params["yaw_w"] = ca.fmin(ca.fmax(ctrl_params["att_P_gain"][2] / roll_pitch_gain, 0.0), 1.0)
ctrl_params["att_P_gain"][2] = roll_pitch_gain

# yaw rate feedforward term and clip it
ctrl_params["yawFF"] = ca.DM(0.0)

# add the calculated rateMax term clip to yawFF
# ctrl_params["yawFF"] = np.clip(ctrl_params["yawFF"], -ctrl_params["rateMax"][2], ctrl_params["rateMax"][2])
ctrl_params["yawFF"] = ca.fmin(ca.fmax(ctrl_params["yawFF"], -ctrl_params["rateMax"][2]), ctrl_params["rateMax"][2])


