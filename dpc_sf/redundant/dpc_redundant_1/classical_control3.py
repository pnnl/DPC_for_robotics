"""
Adaptation of working classical controller 2 - aims to be used with
vectorised inputs like neuromancer inputs now
"""

import torch

import dpc_sf.utils as utils
import dpc_sf.utils.pytorch_utils as ptu

from dpc_sf.dynamics.eom_pt import state_dot_nm, state_dot_pt
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference

rad2deg = 180.0/torch.pi
deg2rad = torch.pi/180.0

ctrl_params = {}

# Set PID Gains and Max Values
# ---------------------------

# Position P gains
ctrl_params["Py"]    = 1.0
ctrl_params["Px"]    = ctrl_params["Py"]
ctrl_params["Pz"]    = 1.0

ctrl_params["pos_P_gain"] = ptu.create_tensor([ctrl_params["Px"], ctrl_params["Py"], ctrl_params["Pz"]])

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

ctrl_params["vel_P_gain"] = ptu.create_tensor([ctrl_params["Pxdot"], ctrl_params["Pydot"], ctrl_params["Pzdot"]])
ctrl_params["vel_D_gain"] = ptu.create_tensor([ctrl_params["Dxdot"], ctrl_params["Dydot"], ctrl_params["Dzdot"]])
ctrl_params["vel_I_gain"] = ptu.create_tensor([ctrl_params["Ixdot"], ctrl_params["Iydot"], ctrl_params["Izdot"]])

# Attitude P gains
ctrl_params["Pphi"] = 8.0
ctrl_params["Ptheta"] = ctrl_params["Pphi"]
ctrl_params["Ppsi"] = 1.5
ctrl_params["PpsiStrong"] = 8

ctrl_params["att_P_gain"] = ptu.create_tensor([ctrl_params["Pphi"], ctrl_params["Ptheta"], ctrl_params["Ppsi"]])

# Rate P-D gains
ctrl_params["Pp"] = 1.5
ctrl_params["Dp"] = 0.04
ctrl_params["Pq"] = ctrl_params["Pp"]
ctrl_params["Dq"] = ctrl_params["Dp"] 
ctrl_params["Pr"] = 1.0
ctrl_params["Dr"] = 0.1

ctrl_params["rate_P_gain"] = ptu.create_tensor([ctrl_params["Pp"], ctrl_params["Pq"], ctrl_params["Pr"]])
ctrl_params["rate_D_gain"] = ptu.create_tensor([ctrl_params["Dp"], ctrl_params["Dq"], ctrl_params["Dr"]])

# Max Velocities
ctrl_params["uMax"] = 5.0
ctrl_params["vMax"] = 5.0
ctrl_params["wMax"] = 5.0

ctrl_params["velMax"] = ptu.create_tensor([ctrl_params["uMax"], ctrl_params["vMax"], ctrl_params["wMax"]])
ctrl_params["velMaxAll"] = 5.0

ctrl_params["saturateVel_separetely"] = False

# Max tilt
ctrl_params["tiltMax"] = 50.0*deg2rad

# Max Rate
ctrl_params["pMax"] = 200.0*deg2rad
ctrl_params["qMax"] = 200.0*deg2rad
ctrl_params["rMax"] = 150.0*deg2rad

ctrl_params["rateMax"] = ptu.create_tensor([ctrl_params["pMax"], ctrl_params["qMax"], ctrl_params["rMax"]])

class Control(torch.nn.Module):
    def __init__(
            self,
            R = waypoint_reference(type='wp_p2p', average_vel=1.6),
            Ts = 0.1,
            ctrl_params = ctrl_params,
            quad_params = quad_params,
            include_actuators=False
        ) -> None:
        super().__init__()

        # state dot function = X_d, reference function = R
        self.R = R
        self.Ts = Ts
        self.ctrl_params = ctrl_params
        self.quad_params = quad_params
        self.yawFF = ptu.create_tensor(0.0)
        self.include_actuators = include_actuators

        # we require memory for set points
        self.vel_sp = ptu.create_tensor([0,0,0])
        self.thrust_sp = ptu.create_tensor([0,0,0])
        self.thr_int = ptu.create_tensor([0,0,0])
        
        self.w_cmd = ptu.create_tensor([quad_params["w_hover"]] * 4)

        # instantiate things
        self.vel_old = ptu.create_tensor([0,0,0])
        self.omega_old = ptu.create_tensor([0,0,0])

        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(self.ctrl_params["att_P_gain"][0] + self.ctrl_params["att_P_gain"][1])
        self.yaw_w = torch.clip(self.ctrl_params["att_P_gain"][2]/roll_pitch_gain, 0.0, 1.0)

        self.ctrl_params["att_P_gain"][2] = roll_pitch_gain


    def __call__(self, x, vel_sp):

        """
        The algorithm:
        --------------

        - We provide a setpoint for velocity that we hope to achieve in x, y, z directions
        - in xyz_vel_control we have a PI controller calculating the thrust required to get there, which is the thrust_sp
        - in thrustToAttitude we calculate the orientation for the quad required to achieve this x, y, z thrust, which is the self.qd_full
        - in attitude_control we have a P controller calculating the angular velocities to get to this orientation, which is self.rate_sp
        - in rate_control we have a P controller minimising error of angular velocities, whilst penalising angular accelerations, generating rateCtrl (desired moment)

        - in mixerFM we calculate the w inputs which would result in total thrust: norm(self.thrust_sp), and moments: rateCtrl
        
        """
        # need convert to NED
        x[:,2] *= -1
        x[:,9] *= -1
        # unpack the state and reference
        # pos = x[0:3]
        quat = x[:,3:7]
        vel = x[:,7:10]
        omega = x[:,10:13]
        # w = x[13:17]
        self.vel_sp = vel_sp
        # pos_sp = r[0:3]
        eul_sp = ptu.create_tensor([0,0,0])
        acc_sp = ptu.create_tensor([0,0,0])
        # self.thrust_sp = -ptu.create_tensor([0,0,self.quad_params["kTh"]*self.quad_params["w_hover"] ** 2 * 4])

        dcm = utils.quat2Dcm(quat)

        # find state dot for control using non NED coords
        # x_standard[2] *= -1
        # x_standard[9] *= -1
        # x_d = self.X_d(x, self.w_cmd, self.quad_params, include_actuators=self.include_actuators)
        # x_d[2] *= -1
        # x_d[9] *= -1
        vel_dot = vel - self.vel_old
        omega_dot = omega - self.omega_old
        # vel_dot = x_d[7:10]
        # vel_dot[-1] *= -1
        # omega_dot = x_d[10:13]

        # manual control of velocity set point
        # self.vel_sp = ptu.create_tensor([-1,0,-1])

        # modifies self.thrust_sp, self.thr_int for antiwindup VALIDATED
        self.thrust_sp, self.thr_int = self.xyz_vel_control(vel=vel, vel_dot=vel_dot, vel_sp=self.vel_sp, acc_sp=acc_sp)

        # creates qd_full
        self.qd_full = self.thrustToAttitude(eul_sp=eul_sp, thrust_sp=self.thrust_sp)

        # only modifies self.rate_sp
        self.rate_sp = self.attitude_control(dcm=dcm, quat=quat)

        # only modifies self.rateCtrl
        rateCtrl = self.rate_control(omega=omega, omega_dot=omega_dot, rate_sp=self.rate_sp)

        # find the w commands to generate the desired rotational rate and thrusts
        self.w_cmd = utils.mixerFM(self.quad_params, torch.linalg.norm(self.thrust_sp), rateCtrl)

        # find the command to achieve the desired w in the next timestep
        # u = self.w_control(w=w, w_cmd=w_cmd)

        print(f"rate_sp: {self.rate_sp}")
        print(f"thrust_sp: {self.thrust_sp}")
        print(f"w_cmd: {self.w_cmd}")
        print(f"thr_int: {self.thr_int}")
        # save old values we need for memory
        self.vel_old = vel
        self.omega_old = omega

        return self.w_cmd

    def saturate_vel(self):
        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (self.ctrl_params["saturateVel_separetely"]):
            self.vel_sp = torch.clip(self.vel_sp, -self.ctrl_params["velMax"], self.ctrl_params["velMax"])
        else:
            totalVel_sp = torch.linalg.norm(self.vel_sp)
            if (totalVel_sp > self.ctrl_params["velMaxAll"]):
                self.vel_sp = self.vel_sp / totalVel_sp * self.ctrl_params["velMaxAll"]


    def thrustToAttitude(self, eul_sp, thrust_sp):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(thrust_sp)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = ptu.create_tensor([-torch.sin(yaw_sp), torch.cos(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = torch.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = torch.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = torch.vstack([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        qd_full = utils.RotToQuat(R_sp)

        return qd_full

    def xyz_vel_control(self, vel, vel_dot, vel_sp, acc_sp):

        # calculate the thrust set point based on the velocity set point defined in self.vel_sp

        assert isinstance(vel, torch.Tensor)
        assert isinstance(vel_dot, torch.Tensor)
        assert isinstance(vel_sp, torch.Tensor)

        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_z_error = - vel_sp[:,2] + vel[:,2]
        thrust_z_sp = self.ctrl_params["vel_P_gain"][2]*vel_z_error - self.ctrl_params["vel_D_gain"][2]*vel_dot[2] + self.quad_params["mB"]*(acc_sp[2] - self.quad_params["g"]) + self.thr_int[:,2]

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad_params["minThr"]
        uMin = -self.quad_params["maxThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[:,2] += self.ctrl_params["vel_I_gain"][2]*vel_z_error*self.Ts * self.quad_params["useIntegral"]
            # Limit thrust integral
            self.thr_int[:,2] = torch.min(torch.abs(self.thr_int[:,2]), ptu.create_tensor([self.quad_params["maxThr"]]))*torch.sign(self.thr_int[:,2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = torch.clip(thrust_z_sp, uMin, uMax)

        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = vel_sp[0:2] - vel[0:2]
        thrust_xy_sp = self.ctrl_params["vel_P_gain"][0:2]*vel_xy_error - self.ctrl_params["vel_D_gain"][0:2]*vel_dot[0:2] + self.quad_params["mB"]*(acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = torch.abs(self.thrust_sp[2])*torch.tan(ptu.create_tensor(self.ctrl_params["tiltMax"]))
        thrust_max_xy = torch.sqrt(self.quad_params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (torch.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = torch.linalg.norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/self.ctrl_params["vel_P_gain"][0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += self.ctrl_params["vel_I_gain"][0:2]*vel_err_lim*self.Ts * self.quad_params["useIntegral"]

        return self.thrust_sp, self.thr_int

    def attitude_control(self, dcm, quat):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = dcm[:,2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)

        # Quaternion error between the 2 vectors
        qe_red = ptu.create_tensor([0,0,0,0])
        qe_red[0] = torch.dot(e_z, e_z_d) + torch.sqrt(torch.linalg.norm(e_z)**2 * torch.linalg.norm(e_z_d)**2)
        qe_red[1:4] = torch.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = utils.quatMultiply(qe_red, quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*torch.sign(q_mix[0])
        q_mix[0] = torch.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = torch.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, torch.hstack([torch.cos(self.yaw_w*torch.arccos(q_mix[0])), ptu.create_tensor(0), ptu.create_tensor(0), torch.sin(self.yaw_w*torch.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0*torch.sign(self.qe[0])*self.qe[1:4])*self.ctrl_params["att_P_gain"]
        
        # Limit yawFF
        self.yawFF = torch.clip(self.yawFF, -self.ctrl_params["rateMax"][2], self.ctrl_params["rateMax"][2])

        # Add Yaw rate feed-forward
        self.rate_sp += utils.quat2Dcm(utils.inverse(quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = torch.clip(self.rate_sp, -self.ctrl_params["rateMax"], self.ctrl_params["rateMax"])

        return self.rate_sp


    def rate_control(self, omega, omega_dot, rate_sp):
        
        # Rate Control
        # ---------------------------
        rate_error = rate_sp - omega
        rateCtrl = self.ctrl_params["rate_P_gain"]*rate_error - self.ctrl_params["rate_D_gain"]*omega_dot     # Be sure it is right sign for the D part
        
        return rateCtrl
    
if __name__ == "__main__":

    from dpc_sf.dynamics.eom_pt import QuadcopterPT
    from tqdm import tqdm
    import numpy as np
    from dpc_sf.utils.animation import Animator
    import copy
    import matplotlib.pyplot as plt

    class Visualiser():
        def __init__(self, reference=waypoint_reference('wp_p2p', average_vel=1.6)) -> None:
            self.reference=reference
            self.x = []
            self.u = []
            self.r = []
            self.t = []

        def save(self, x, u, r, t):
            self.x.append(copy.deepcopy(x))
            self.u.append(np.copy(u))
            self.r.append(np.copy(r))
            self.t.append(t)

        def animate(self, x_p=None, drawCylinder=False):
            animator = Animator(
                states=np.vstack(self.x), 
                times=np.array(self.t), 
                reference_history=np.vstack(self.r), 
                reference=self.reference, 
                reference_type=self.reference.type, 
                drawCylinder=drawCylinder,
                state_prediction=x_p
            )
            animator.animate() # does not contain plt.show()      

    vis = Visualiser()

    print("testing the classical control system from the github repo")
    Ts = 0.01
    ctrl = Control(Ts=Ts)
    quad = QuadcopterPT(Ts=Ts, include_actuators=False)
    R = waypoint_reference()
    
    for i in tqdm(range(800)):
        t = i*Ts
        x = ptu.from_numpy(copy.deepcopy(quad.get_state()))
        r = R(t)
        vel_sp = torch.sin(torch.tensor([t*1, t*1, t*2])) * 5

        u = ptu.to_numpy(ctrl(x, vel_sp))
        if i % 20 == 0:
            x[2] *= -1
            x[9] *= -1

            vis.save(x,u,r,t)
        quad.step(u)
        print(f"x: {x}")

    vis.animate()
    plt.show()

    print('fin')
