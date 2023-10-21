import torch
import numpy as np

import dpc_sf.utils as utils
import dpc_sf.utils.pytorch_utils as ptu

rad2deg = 180.0/torch.pi
deg2rad = torch.pi/180.0

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
Py    = 1.0
Px    = Py
Pz    = 1.0

pos_P_gain = ptu.from_numpy(np.array([Px, Py, Pz]))

# Velocity P-D gains
Pxdot = 5.0
Dxdot = 0.5
Ixdot = 5.0

Pydot = Pxdot
Dydot = Dxdot
Iydot = Ixdot

Pzdot = 4.0
Dzdot = 0.5
Izdot = 5.0

vel_P_gain = ptu.from_numpy(np.array([Pxdot, Pydot, Pzdot]))
vel_D_gain = ptu.from_numpy(np.array([Dxdot, Dydot, Dzdot]))
vel_I_gain = ptu.from_numpy(np.array([Ixdot, Iydot, Izdot]))

# Attitude P gains
Pphi = 8.0
Ptheta = Pphi
Ppsi = 1.5
PpsiStrong = 8

att_P_gain = ptu.from_numpy(np.array([Pphi, Ptheta, Ppsi]))

# Rate P-D gains
Pp = 1.5
Dp = 0.04

Pq = Pp
Dq = Dp 

Pr = 1.0
Dr = 0.1

rate_P_gain = ptu.from_numpy(np.array([Pp, Pq, Pr]))
rate_D_gain = ptu.from_numpy(np.array([Dp, Dq, Dr]))

# Max Velocities
uMax = 5.0
vMax = 5.0
wMax = 5.0

velMax = ptu.from_numpy(np.array([uMax, vMax, wMax]))
velMaxAll = 5.0

saturateVel_separetely = False

# Max tilt
tiltMax = 50.0*deg2rad

# Max Rate
pMax = 200.0*deg2rad
qMax = 200.0*deg2rad
rMax = 150.0*deg2rad

rateMax = ptu.from_numpy(np.array([pMax, qMax, rMax]))

class Control:
    def __init__(self, quad, Ts=0.1):

        self.Ts = Ts
        self.quad = quad

        self.sDesCalc   = ptu.from_numpy(np.zeros(16))
        self.w_cmd      = ptu.from_numpy(np.ones(4)*quad.params["w_hover"])
        self.thr_int    = ptu.from_numpy(np.zeros(3))
        self.pos_des     = ptu.from_numpy(np.zeros(3))
        self.vel_des     = ptu.from_numpy(np.zeros(3))
        self.acc_des     = ptu.from_numpy(np.zeros(3))
        self.thr_des  = ptu.from_numpy(np.zeros(3))
        self.eul_des     = ptu.from_numpy(np.zeros(3))
        self.pqr_des     = ptu.from_numpy(np.zeros(3))
        self.yawFF      = ptu.from_numpy(np.zeros(3))

    def controller(self, x, r):
        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_des[:]     = r[0:3]
        self.vel_des[:]     = r[3:6]
        self.acc_des[:]     = r[6:9]
        self.thr_des[:]     = r[9:12]
        self.eul_des[:]     = r[12:15]
        self.pqr_des[:]     = r[15:18]
        self.yawFF[:]       = r[18]
        
        # Select Controller - xyz velocity is desired
        # ---------------------------
        self.saturateVel()
        self.z_vel_control(self.quad, self.Ts)
        self.xy_vel_control(self.quad, self.Ts)
        self.thrustToAttitude(self.quad, self.Ts)
        self.attitude_control(self.quad, self.Ts)
        self.rate_control(self.quad, self.Ts)

        # Mixer
        # --------------------------- 
        self.w_cmd = utils.mixerFM(self.quad, torch.linalg.norm(self.thr_des), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_des
        self.sDesCalc[3:6] = self.vel_des
        self.sDesCalc[6:9] = self.thr_des
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_des

    def z_vel_control(self, x, Ts):
        
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul

        # need z dot and 2dot
        z_d = None
        z_dd = None

        vel_z_error = self.vel_des[2] - z_d
        thrust_z_sp = vel_P_gain[2]*vel_z_error - vel_D_gain[2]*z_dd + self.quad.params["mB"]*(self.acc_des[2] - self.quad.params["g"]) + self.thr_int[2]

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad.params["minThr"]
        uMin = -self.quad.params["maxThr"]


        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += vel_I_gain[2]*vel_z_error*Ts * self.quad.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), self.quad.params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thr_des[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, x):
        
        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------

        # need xy dot and 2 dot
        xy_d = None
        xy_dd = None

        vel_xy_error = self.vel_sp[0:2] - xy_d
        thrust_xy_sp = vel_P_gain[0:2]*vel_xy_error - vel_D_gain[0:2]*xy_dd + self.quad.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thr_des[2])*np.tan(tiltMax)
        thrust_max_xy = torch.sqrt(self.quad.params["maxThr"]**2 - self.thr_des[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thr_des[0:2] = thrust_xy_sp
        if (np.dot(self.thr_des[0:2].T, self.thr_des[0:2]) > thrust_max_xy**2):
            mag = torch.linalg.norm(self.thr_des[0:2])
            self.thr_des[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thr_des[0:2])*arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2]*vel_err_lim*self.Ts * self.quad.params["useIntergral"]

    def thrustToAttitude(self, x):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_des = self.eul_des[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thr_des)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        # y_C = np.array([-torch.sin(yaw_des), torch.cos(yaw_des), 0.0])
        y_C = torch.stack([-torch.sin(yaw_des), torch.cos(yaw_des), 0.0])
        
        # Desired body_x axis direction
        body_x = torch.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = torch.cross(body_z, body_x)

        # Desired rotation matrix
        R_des = torch.stack([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = utils.RotToQuat(R_des)
        
    def attitude_control(self, x):

        # need direction cosine matrix (quad.dcm original location), and current quaternion
        dcm = None
        quat = None

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = dcm[:,2]
        e_z_d = -utils.vectNormalize(self.thr_des)

        # Quaternion error between the 2 vectors
        qe_red = ptu.from_numpy(np.zeros(4))
        qe_red[0] = torch.dot(e_z, e_z_d) + torch.sqrt(torch.linalg.norm(e_z)**2 * torch.linalg.norm(e_z_d)**2)
        qe_red[1:4] = torch.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = utils.quatMultiply(qe_red, quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = torch.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = torch.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, np.array([torch.cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, torch.sin(self.yaw_w*np.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_des = (2.0*torch.sign(self.qe[0])*self.qe[1:4])*att_P_gain
        
        # Limit yawFF
        self.yawFF = torch.clip(self.yawFF, -rateMax[2], rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_des += utils.quat2Dcm(utils.inverse(quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_des = torch.clip(self.rate_des, -rateMax, rateMax)

    def rate_control(self, quad, Ts):

        # need quad "omega", this is a rate, not the rotor rotational rates. (quad.omega)
        omega = None
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_des - omega
        self.rateCtrl = rate_P_gain*rate_error - rate_D_gain*quad.omega_dot     # Be sure it is right sign for the D part
        
    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain