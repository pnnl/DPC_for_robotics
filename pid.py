"""
Adapting classical_control2_func to recieve (batch, state)

COMPLETE - asides from the velocity saturation
"""

import casadi as ca
import numpy as np
import torch
import utils.pytorch as ptu
import utils.rotation
import utils.quad

rad2deg = 180.0/torch.pi
deg2rad = torch.pi/180.0

def get_ctrl_params():

    ctrl_params = {}

    # Set PID Gains and Max Values
    # ---------------------------

    # Position P gains
    ctrl_params["Py"]    = 1.0
    ctrl_params["Px"]    = ctrl_params["Py"]
    ctrl_params["Pz"]    = 1.0

    ctrl_params["pos_P_gain"] = ptu.tensor([ctrl_params["Px"], ctrl_params["Py"], ctrl_params["Pz"]])

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

    ctrl_params["vel_P_gain"] = ptu.tensor([ctrl_params["Pxdot"], ctrl_params["Pydot"], ctrl_params["Pzdot"]])
    ctrl_params["vel_D_gain"] = ptu.tensor([ctrl_params["Dxdot"], ctrl_params["Dydot"], ctrl_params["Dzdot"]])
    ctrl_params["vel_I_gain"] = ptu.tensor([ctrl_params["Ixdot"], ctrl_params["Iydot"], ctrl_params["Izdot"]])

    # Attitude P gains
    ctrl_params["Pphi"] = 8.0
    ctrl_params["Ptheta"] = ctrl_params["Pphi"]
    ctrl_params["Ppsi"] = 1.5
    ctrl_params["PpsiStrong"] = 8

    ctrl_params["att_P_gain"] = ptu.tensor([ctrl_params["Pphi"], ctrl_params["Ptheta"], ctrl_params["Ppsi"]])

    # Rate P-D gains
    ctrl_params["Pp"] = 1.5
    ctrl_params["Dp"] = 0.04
    ctrl_params["Pq"] = ctrl_params["Pp"]
    ctrl_params["Dq"] = ctrl_params["Dp"] 
    ctrl_params["Pr"] = 1.0
    ctrl_params["Dr"] = 0.1

    ctrl_params["rate_P_gain"] = ptu.tensor([ctrl_params["Pp"], ctrl_params["Pq"], ctrl_params["Pr"]])
    ctrl_params["rate_D_gain"] = ptu.tensor([ctrl_params["Dp"], ctrl_params["Dq"], ctrl_params["Dr"]])

    # Max Velocities
    ctrl_params["uMax"] = 5.0
    ctrl_params["vMax"] = 5.0
    ctrl_params["wMax"] = 5.0

    ctrl_params["velMax"] = ptu.tensor([ctrl_params["uMax"], ctrl_params["vMax"], ctrl_params["wMax"]])
    ctrl_params["velMaxAll"] = 5.0

    ctrl_params["saturateVel_separetely"] = False

    # Max tilt
    ctrl_params["tiltMax"] = 50.0*deg2rad

    # Max Rate
    ctrl_params["pMax"] = 200.0*deg2rad
    ctrl_params["qMax"] = 200.0*deg2rad
    ctrl_params["rMax"] = 150.0*deg2rad

    ctrl_params["rateMax"] = ptu.tensor([ctrl_params["pMax"], ctrl_params["qMax"], ctrl_params["rMax"]])
    roll_pitch_gain = 0.5*(ctrl_params["att_P_gain"][0] + ctrl_params["att_P_gain"][1])

    # assumed yaw_w to be 1 to allow for much better gradients
    ctrl_params["yaw_w"] = torch.clip(ctrl_params["att_P_gain"][2]/roll_pitch_gain, 0.0, 1.0)
    ctrl_params["att_P_gain"][2] = roll_pitch_gain

    # yaw rate feedforward term and clip it
    ctrl_params["yawFF"] = ptu.tensor(0.0)

    # add the calculated rateMax term clip to yawFF
    ctrl_params["yawFF"] = torch.clip(ctrl_params["yawFF"], -ctrl_params["rateMax"][2], ctrl_params["rateMax"][2])

    return ctrl_params

class PID(torch.nn.Module):

    """
    A cascade PI controller that recieves reference:
        xdot, ydot, zdot
    and outputs desired rotor angular velocities:
        w (1x4)

    The algorithm:
    --------------

        - We provide a setpoint for thrust that we hope to achieve in x, y, z directions
        - in xyz_thr_control we have a PI controller limit the thrust requested, which is the thrust_sp
        - in thrustToAttitude we calculate the orientation for the quad required to achieve this x, y, z thrust, which is the self.qd_full
        - in attitude_control we have a P controller calculating the angular velocities to get to this orientation, which is self.rate_sp
        - in rate_control we have a P controller minimising error of angular velocities, whilst penalising angular accelerations, generating rateCtrl (desired moment)

        - in mixerFM we calculate the w inputs which would result in total thrust: norm(self.thrust_sp), and moments: rateCtrl
    """

    def __init__(
            self,
            Ts,
            bs,
            ctrl_params,
            quad_params,
            include_actuators=True,
            verbose=False,
            input="xyz_thr", # 'xyz_vel', 'xyz_thr'
        ) -> None:
        super().__init__()

        # STUFF I AGREE WITH
        # ------------------
        self.Ts = Ts
        self.bs = bs
        self.ctrl_params = ctrl_params
        self.quad_params = quad_params
        self.include_actuators = include_actuators
        self.verbose = verbose
        self.input = input

        # the internal states of the integral components:
        self.thr_int = ptu.zeros([bs,3])

        # STUFF I KIND OF AGREE WITH BUT COULD BE FIXED (TODO s:)
        # -------------------------------------------------------
        
        # we currently calculate the statedot by just taking the delta between timesteps
        # this becomes inaccurate at slower timesteps, and so is a candidate to be upgraded
        # with the true dynamics in the future
        self.vel_old = ptu.zeros([bs, 3])
        self.omega_old = ptu.zeros([bs, 3])

    def __call__(self, x, sp):

        quat = x[:,3:7]
        vel = x[:,7:10]
        omega = x[:,10:13]

        eul_sp = ptu.zeros(vel.shape)
        acc_sp = ptu.zeros(vel.shape)

        dcm = utils.rotation.quaternion_to_dcm.pytorch_vectorized(quat)

        # should be replaceable by the known dynamics of the system TODO
        vel_dot = vel - self.vel_old
        omega_dot = omega - self.omega_old


        if self.input == "xyz_vel":

            # saturate the vel_sp
            vel_sp = self.saturate_vel_batched(vel_sp=sp)

            # modifies self.thrust_sp, self.thr_int for antiwindup VALIDATED
            thrust_sp = self.xyz_vel_control(vel=vel, vel_dot=vel_dot, vel_sp=vel_sp, acc_sp=acc_sp)

        elif self.input == "xyz_thr":

            # simply limits thrust setpoint dynamically
            thrust_sp = self.xyz_thr_control(sp)

        # creates qd_full
        qd_full = self.thrustToAttitude(eul_sp=eul_sp, thrust_sp=thrust_sp)

        # only modifies self.rate_sp
        rate_sp = self.attitude_control(dcm=dcm, quat=quat, thrust_sp=thrust_sp, qd_full=qd_full)

        # only modifies self.rateCtrl
        rateCtrl = self.rate_control(omega=omega, omega_dot=omega_dot, rate_sp=rate_sp)

        # find the w commands to generate the desired rotational rate and thrusts
        w_cmd = utils.quad.applyMixerFM.pytorch_vectorized(self.quad_params, torch.linalg.norm(thrust_sp, dim=1), rateCtrl)

        if self.verbose:
            print(f"rate_sp: {rate_sp[0,:]}")
            print(f"thrust_sp: {thrust_sp[0,:]}")
            print(f"w_cmd: {w_cmd[0,:]}")
            print(f"thr_int: {self.thr_int[0,:]}")

        # save states
        self.vel_old = vel
        self.omega_old = omega

        if w_cmd.isnan().any():
            print('fin')

        pid_x = self.get_pid_x()

        if self.include_actuators:
            # find the command to achieve the desired w in the next timestep
            w = x[:,13:17]
            u = self.w_control(w=w, w_cmd=w_cmd)
            return u, pid_x
        else:
            # 
            return w_cmd, pid_x
        
    def get_pid_x(self):
        return torch.cat([self.thr_int, self.vel_old, self.omega_old], dim=1)
        # return {'thr_int': self.thr_int, 'vel_old': self.vel_old, 'omega_old': self.omega_old}
    
    def reset(self, pid_x):
        if pid_x is None:
            # call to reset integrators and states and whatnot
            self.thr_int = ptu.zeros([self.bs, 3])
            self.vel_old = ptu.zeros([self.bs, 3])
            self.omega_old = ptu.zeros([self.bs, 3])

        else:
            self.thr_int = pid_x[:,:3]
            self.vel_old = pid_x[:,3:6]
            self.omega_old = pid_x[:,6:]

    def saturate_vel_batched(self, vel_sp):
        # UNTESTED
        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if (self.ctrl_params["saturateVel_separetely"]):
            vel_sp = torch.clip(vel_sp, -self.ctrl_params["velMax"], self.ctrl_params["velMax"])
        else:
            totalVel_sp = torch.linalg.norm(vel_sp, dim=1, keepdim=True)
            mask = totalVel_sp > self.ctrl_params["velMaxAll"]
            scaling_factors = torch.where(mask, self.ctrl_params["velMaxAll"] / totalVel_sp, torch.tensor(1.0))
            vel_sp = vel_sp * scaling_factors
            # if (totalVel_sp > self.ctrl_params["velMaxAll"]):
            #     vel_sp = vel_sp / totalVel_sp * self.ctrl_params["velMaxAll"]
        return vel_sp


    def thrustToAttitude(self, eul_sp, thrust_sp):
        
        """Marked to remain the same in the DPC conversion"""

        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = eul_sp[:,2]

        # Desired body_z axis direction
        body_z = -utils.rotation.normalize_vector.pytorch_vectorized(thrust_sp)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = torch.vstack([-torch.sin(yaw_sp), torch.cos(yaw_sp), ptu.zeros(self.bs)]).T
        
        # Desired body_x axis direction
        body_x = torch.cross(y_C, body_z)
        body_x = utils.rotation.normalize_vector.pytorch_vectorized(body_x)
        
        # Desired body_y axis direction
        body_y = torch.cross(body_z, body_x)

        # Desired rotation matrix - permute does the transpose on the batched 3x3 matrices
        R_sp = torch.stack([body_x, body_y, body_z], dim=1).permute(0,2,1)
        # R_sp = torch.vstack([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        qd_full = utils.rotation.rot_matrix_to_quaternion.pytorch_vectorized(R_sp)

        if qd_full.isnan().any():
            print('fin')

        return qd_full
    
    def xyz_thr_control(self, thr_sp):

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad_params["minThr"] - 1e-8
        uMin = -self.quad_params["maxThr"] + 1e-8

        thrust_xy_sp = thr_sp[:,0:2]
        thrust_z_sp = thr_sp[:,2]

        # Saturate thrust setpoint in D-direction
        thrust_z_sp = torch.clip(thrust_z_sp, uMin + 1e-3, uMax - 1e-3)

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = torch.abs(thrust_z_sp)*torch.tan(ptu.tensor(self.ctrl_params["tiltMax"]))
        thrust_max_xy = torch.sqrt(self.quad_params["maxThr"]**2 - thrust_z_sp**2)
        thrust_max_xy_min = torch.min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        mask = ((thrust_xy_sp ** 2).sum(dim=1) > thrust_max_xy_min**2)

        # Calculate norms for each row where the condition is True
        mags = torch.linalg.norm(thrust_xy_sp[mask,:], dim=1)

        # Update rows of thrust_xy_sp where the condition is True
        # thrust_xy_sp[mask] = (thrust_xy_sp[mask,:].T / mags * thrust_max_xy_min[mask]).T

        # we do this instead to avoid in place operations
        # Calculate norms for each row where the condition is True
        mags = torch.linalg.norm(thrust_xy_sp, dim=1)
    
        # Update rows of thrust_xy_sp where the condition is True
        # print(f"thrust_xy_sp nan?: {thrust_xy_sp.isnan().any()}")
        # print(f"thrust_max_xy_min nan?: {thrust_max_xy_min.isnan().any()}")
        # print(f"mags nan?: {mags.isnan().any()}")

        conditioned_values = (thrust_xy_sp.T / mags * thrust_max_xy_min).T
        thrust_xy_sp_masked = torch.where(mask.unsqueeze(-1), conditioned_values, thrust_xy_sp)

        # concatenate the z and xy thrust set points together 
        thrust_sp = torch.hstack([thrust_xy_sp_masked, thrust_z_sp.unsqueeze(1)])

        return thrust_sp

    # VALIDATED
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
        thrust_z_sp = self.ctrl_params["vel_P_gain"][2]*vel_z_error - self.ctrl_params["vel_D_gain"][2]*vel_dot[:,2] + self.quad_params["mB"]*(acc_sp[:,2] - self.quad_params["g"]) + self.thr_int[:,2]

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad_params["minThr"] - 1e-8
        uMin = -self.quad_params["maxThr"] + 1e-8

        # Apply Anti-Windup in D-direction
        stop_int_D = ((thrust_z_sp >= uMax) & (vel_z_error >= 0.0)) | ((thrust_z_sp <= uMin) & (vel_z_error <= 0.0))

        # Mask for applying anti-windup to only the correct parts of the batch
        mask = ~stop_int_D

        # Calculate integral part
        # Update only those parts of self.thr_int where stop_int_D is False
        self.thr_int[mask, 2] += self.ctrl_params["vel_I_gain"][2] * vel_z_error[mask] * self.Ts * self.quad_params["useIntegral"]

        # Apply the limitation only on the relevant parts
        limited = torch.min(torch.abs(self.thr_int[mask, 2]), ptu.tensor([self.quad_params["maxThr"]]))

        # replace this operation in place changes with non in-place
        # self.thr_int[mask, 2] = limited * torch.sign(self.thr_int[mask, 2])

        updated_thr_int = self.thr_int.clone()
        updated_thr_int[mask, 2] = limited * torch.sign(self.thr_int[mask, 2])
        self.thr_int = updated_thr_int

        # Calculate integral part
        # if not (stop_int_D):
        #     self.thr_int[:,2] += self.ctrl_params["vel_I_gain"][2]*vel_z_error*self.Ts * self.quad_params["useIntegral"]
        #     # Limit thrust integral
        #     self.thr_int[:,2] = torch.min(torch.abs(self.thr_int[:,2]), ptu.tensor([self.quad_params["maxThr"]]))*torch.sign(self.thr_int[:,2])

        # Saturate thrust setpoint in D-direction
        thrust_z_sp = torch.clip(thrust_z_sp, uMin + 1e-3, uMax - 1e-3)

        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = vel_sp[:,0:2] - vel[:,0:2]
        thrust_xy_sp = self.ctrl_params["vel_P_gain"][0:2]*vel_xy_error - self.ctrl_params["vel_D_gain"][0:2]*vel_dot[:,0:2] + self.quad_params["mB"]*(acc_sp[:,0:2]) + self.thr_int[:,0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = torch.abs(thrust_z_sp)*torch.tan(ptu.tensor(self.ctrl_params["tiltMax"]))
        thrust_max_xy = torch.sqrt(self.quad_params["maxThr"]**2 - thrust_z_sp**2)
        thrust_max_xy_min = torch.min(thrust_max_xy, thrust_max_xy_tilt)

        if thrust_max_xy_min.isnan().any():
            print('we have NaNs in xyz_vel_control')

        # Saturate thrust in NE-direction
        mask = ((thrust_xy_sp ** 2).sum(dim=1) > thrust_max_xy_min**2)

        # Calculate norms for each row where the condition is True
        mags = torch.linalg.norm(thrust_xy_sp[mask,:], dim=1)

        # Update rows of thrust_xy_sp where the condition is True
        # thrust_xy_sp[mask] = (thrust_xy_sp[mask,:].T / mags * thrust_max_xy_min[mask]).T

        # we do this instead to avoid in place operations
        # Calculate norms for each row where the condition is True
        mags = torch.linalg.norm(thrust_xy_sp, dim=1)
    
        # Update rows of thrust_xy_sp where the condition is True
        # print(f"thrust_xy_sp nan?: {thrust_xy_sp.isnan().any()}")
        # print(f"thrust_max_xy_min nan?: {thrust_max_xy_min.isnan().any()}")
        # print(f"mags nan?: {mags.isnan().any()}")

        conditioned_values = (thrust_xy_sp.T / mags * thrust_max_xy_min).T
        thrust_xy_sp_masked = torch.where(mask.unsqueeze(-1), conditioned_values, thrust_xy_sp)

        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0/self.ctrl_params["vel_P_gain"][0:2]
        vel_err_lim = vel_xy_error # - (thrust_xy_sp - thrust_xy_sp)*arw_gain
        self.thr_int[:,0:2] += self.ctrl_params["vel_I_gain"][0:2]*vel_err_lim*self.Ts * self.quad_params["useIntegral"]

        # concatenate the z and xy thrust set points together 
        thrust_sp = torch.hstack([thrust_xy_sp_masked, thrust_z_sp.unsqueeze(1)])

        return thrust_sp

    def attitude_control(self, dcm, quat, thrust_sp, qd_full):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = dcm[:,:,2]
        e_z_d = -utils.rotation.normalize_vector.pytorch_vectorized(thrust_sp)

        # Quaternion error between the 2 vectors - TODO get torch.dot to work properly with batched
        qe_red = ptu.zeros(quat.shape)
        qe_red[:,0] = torch.sum(e_z * e_z_d, dim=1) + torch.sqrt(torch.linalg.norm(e_z, dim=1)**2 * torch.linalg.norm(e_z_d, dim=1)**2)

        # qe_red[0] = torch.dot(e_z, e_z_d) + torch.sqrt(torch.linalg.norm(e_z, dim=1)**2 * torch.linalg.norm(e_z_d, dim=1)**2)
        qe_red[:,1:4] = torch.cross(e_z, e_z_d)
        qe_red = utils.rotation.normalize_vector.pytorch_vectorized(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        qd_red = utils.rotation.quaternion_multiply.pytorch_vectorized(qe_red, quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.rotation.quaternion_multiply.pytorch_vectorized(utils.rotation.quaternion_inverse.pytorch_vectorized(qd_red), qd_full)
        q_mix = q_mix*torch.sign(q_mix[:,0:1])
        # q_mix[:,0] = torch.clip(q_mix[:,0], -1.0, 1.0)
        # q_mix[:,3] = torch.clip(q_mix[:,3], -1.0, 1.0)

        # print(f"q_mix nan: {q_mix.isnan().any()}")
        # print(f"q_mix abs max (near 1 leads to poor gradients/nans): {q_mix[:,0].abs().max()}")

        # q0ac = torch.arccos(q_mix[:,0])
        q0 = torch.clip(q_mix[:,0], -1.0, 1.0)
        # q0 = torch.cos(self.ctrl_params["yaw_w"]*q0ac)
        q1 = ptu.zeros(self.bs)
        q2 = ptu.zeros(self.bs)
        # q3 = torch.sin(self.ctrl_params["yaw_w"]*torch.arcsin(q_mix[:,3]))
        q3 = torch.clip(q_mix[:,3], -1.0, 1.0)
        multiplier = torch.vstack([q0, q1, q2, q3]).T

        qd = utils.rotation.quaternion_multiply.pytorch_vectorized(qd_red, multiplier)

        # Resulting error quaternion
        qe = utils.rotation.quaternion_multiply.pytorch_vectorized(utils.rotation.quaternion_inverse.pytorch_vectorized(quat), qd)

        # Create rate setpoint from quaternion error
        rate_sp = (2.0*torch.sign(qe[:,0:1])*qe[:,1:4])*self.ctrl_params["att_P_gain"]
        
        # Add Yaw rate feed-forward term clipped by rate limits calculated by yaw rate weighting in ctrl_params
        rate_sp += utils.rotation.quaternion_to_dcm.pytorch_vectorized(utils.rotation.quaternion_inverse.pytorch_vectorized(quat))[:,:,2]*self.ctrl_params["yawFF"]

        # Limit rate setpoint
        rate_sp = torch.clip(rate_sp, -self.ctrl_params["rateMax"], self.ctrl_params["rateMax"])

        return rate_sp


    def rate_control(self, omega, omega_dot, rate_sp):
        
        # Rate Control
        # ---------------------------
        rate_error = rate_sp - omega
        rateCtrl = self.ctrl_params["rate_P_gain"]*rate_error - self.ctrl_params["rate_D_gain"]*omega_dot     # Be sure it is right sign for the D part
        
        return rateCtrl
    
    def w_control(self, w, w_cmd):

        # the above is the commanded omega
        w_error = w_cmd - w
        p_gain = self.quad_params["IRzz"] / self.Ts
        motor_inputs = w_error * p_gain
        return motor_inputs
    
class PID_CA():

    """
    A cascade PI controller that recieves reference:
        xdot, ydot, zdot
    and outputs desired rotor angular velocities:
        w (1x4)

    The algorithm:
    --------------

        - We provide a setpoint for velocity that we hope to achieve in x, y, z directions
        - in xyz_vel_control we have a PI controller calculating the thrust required to get there, which is the thrust_sp
        - in thrustToAttitude we calculate the orientation for the quad required to achieve this x, y, z thrust, which is the self.qd_full
        - in attitude_control we have a P controller calculating the angular velocities to get to this orientation, which is self.rate_sp
        - in rate_control we have a P controller minimising error of angular velocities, whilst penalising angular accelerations, generating rateCtrl (desired moment)

        - in mixerFM we calculate the w inputs which would result in total thrust: norm(self.thrust_sp), and moments: rateCtrl
    """
    def __init__(
            self,
            Ts,
            bs,
            ctrl_params,
            quad_params,
            include_actuators=False,
            verbose=False,
            input="xyz_vel", # 'xyz_vel', 'xyz_thr'
        ) -> None:
        super().__init__()

        # STUFF I AGREE WITH
        # ------------------
        self.Ts = Ts
        self.bs = bs

        convert_dict = lambda input_dict: {k: ptu.to_numpy(v) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

        self.ctrl_params = convert_dict(ctrl_params)
        self.quad_params = convert_dict(quad_params)
        self.include_actuators = include_actuators
        self.verbose = verbose
        self.input = input

        # the internal states of the integral components:
        # self.thr_int = np.zeros([bs,3])
        self.thr_int = ca.MX.zeros(bs, 3)

        # STUFF I KIND OF AGREE WITH BUT COULD BE FIXED (TODO s:)
        # -------------------------------------------------------
        
        # we currently calculate the statedot by just taking the delta between timesteps
        # this becomes inaccurate at slower timesteps, and so is a candidate to be upgraded
        # with the true dynamics in the future
        # self.vel_old = np.zeros([bs, 3])
        # self.omega_old = np.zeros([bs, 3])
        self.vel_old = ca.MX.zeros(bs, 3)
        self.omega_old = ca.MX.zeros(bs, 3)

    def __call__(self, x, sp):

        # x = x.T
        sp = sp.T

        quat = x[:,3:7]
        vel = x[:,7:10]
        omega = x[:,10:13]

        eul_sp = ca.DM.zeros(vel.shape)
        acc_sp = ca.DM.zeros(vel.shape)

        dcm = utils.rotation.quaternion_to_dcm.casadi(quat)
        # dcm = utils.quat2Dcm_batched_ca(quat)

        # should be replaceable by the known dynamics of the system TODO
        # vel_dot = vel - self.vel_old
        omega_dot = omega - self.omega_old

        # simply limits thrust setpoint dynamically
        thrust_sp = self.xyz_thr_control(sp)

        # creates qd_full
        qd_full = self.thrustToAttitude(eul_sp=eul_sp, thrust_sp=thrust_sp)

        # only modifies self.rate_sp
        rate_sp = self.attitude_control(dcm=dcm, quat=quat, thrust_sp=thrust_sp, qd_full=qd_full)

        # only modifies self.rateCtrl
        rateCtrl = self.rate_control(omega=omega, omega_dot=omega_dot, rate_sp=rate_sp)

        # find the w commands to generate the desired rotational rate and thrusts
        # norms = [ca.norm_2(thrust_sp[i,:]) for i in range(thrust_sp.shape[0])]
        norms = ca.norm_2(thrust_sp[0,:])
        # w_cmd = utils.mixerFM_batched_ca(self.quad_params, norms, rateCtrl)
        w_cmd = utils.quad.applyMixerFM.casadi(self.quad_params, norms, rateCtrl)

        if self.verbose:
            print(f"rate_sp: {rate_sp[0,:]}")
            print(f"thrust_sp: {thrust_sp[0,:]}")
            print(f"w_cmd: {w_cmd[0,:]}")
            print(f"thr_int: {self.thr_int[0,:]}")

        # save states
        self.vel_old = vel
        self.omega_old = omega


        if self.include_actuators:
            # find the command to achieve the desired w in the next timestep
            w = x[:,13:17]
            u = self.w_control(w=w, w_cmd=w_cmd)
            return u
        else:
            # 
            return w_cmd
    
    def reset(self):
        # call to reset integrators and states and whatnot
        self.thr_int = ca.DM.zeros(self.bs, 3)
        self.vel_old = ca.DM.zeros(self.bs, 3)
        self.omega_old = ca.DM.zeros(self.bs, 3)

    # def saturate_vel_batched(self, vel_sp):
    #     # UNTESTED
    #     # Saturate Velocity Setpoint
    #     # --------------------------- 
    #     # Either saturate each velocity axis separately, or total velocity (prefered)
    #     if (self.ctrl_params["saturateVel_separetely"]):
    #         vel_sp = np.clip(vel_sp, -self.ctrl_params["velMax"], self.ctrl_params["velMax"])
    #     else:
    #         totalVel_sp = np.linalg.norm(vel_sp, dim=1, keepdim=True)
    #         mask = totalVel_sp > self.ctrl_params["velMaxAll"]
    #         scaling_factors = np.where(mask, self.ctrl_params["velMaxAll"] / totalVel_sp, np.array(1.0))
    #         vel_sp = vel_sp * scaling_factors
    #         # if (totalVel_sp > self.ctrl_params["velMaxAll"]):
    #         #     vel_sp = vel_sp / totalVel_sp * self.ctrl_params["velMaxAll"]
    #     return vel_sp


    def thrustToAttitude(self, eul_sp, thrust_sp):
        
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = eul_sp[:,2]

        # Desired body_z axis direction
        body_z = -utils.rotation.normalize_vector.casadi(thrust_sp)

        # body_z = -utils.vectNormalize_batched_ca(thrust_sp)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        # y_C = np.vstack([-np.sin(yaw_sp), np.cos(yaw_sp), np.zeros(self.bs)]).T
        y_C = ca.vertcat(-ca.sin(yaw_sp), ca.cos(yaw_sp), ca.MX.zeros(self.bs)).T

        # Desired body_x axis direction
        # body_x = np.cross(y_C, body_z)
        # body_x = utils.vectNormalize_batched(body_x)
        body_x = ca.cross(y_C, body_z)
        # body_x = utils.vectNormalize_batched_ca(body_x)  # Assuming this is the CasADi version of the function
        body_x = utils.rotation.normalize_vector.casadi(body_x)

        # Desired body_y axis direction
        # body_y = np.cross(body_z, body_x)
        body_y = ca.cross(body_z, body_x)

        # Desired rotation matrix - permute does the transpose on the batched 3x3 matrices
        # R_sp = np.stack([body_x, body_y, body_z], dim=1).permute(0,2,1)
        # R_sp = np.vstack([body_x, body_y, body_z]).T
        R_sp = ca.vertcat(body_x, body_y, body_z).T  # This will create the matrix using the three vectors as its columns

        # Full desired quaternion (full because it considers the desired Yaw angle)
        # qd_full = utils.RotToQuat_batched_ca(R_sp)
        qd_full = utils.rotation.rot_matrix_to_quaternion.casadi_vectorized(R_sp)

        # if qd_full.isnan().any():
        #     print('fin')

        return qd_full
    
    def xyz_thr_control(self, thr_sp):

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad_params["minThr"] - 1e-8
        uMin = -self.quad_params["maxThr"] + 1e-8

        thrust_xy_sp = thr_sp[:,0:2]
        thrust_z_sp = thr_sp[:,2]

        # Saturate thrust setpoint in D-direction
        # thrust_z_sp = np.clip(thrust_z_sp, uMin + 1e-3, uMax - 1e-3)
        thrust_z_sp = ca.fmin(ca.fmax(thrust_z_sp, uMin + 1e-3), uMax - 1e-3)

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = ca.mtimes(ca.fabs(thrust_z_sp), ca.tan(self.ctrl_params["tiltMax"]))
        thrust_max_xy = ca.sqrt(self.quad_params["maxThr"]**2 - thrust_z_sp**2)
        thrust_max_xy_min = ca.fmin(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        # Here we use a matrix multiplication to calculate squared sum over last dimension
        sum_squared = ca.mtimes(thrust_xy_sp, thrust_xy_sp.T)
        mask = sum_squared > thrust_max_xy_min**2

        # Calculate norms for each row
        mags = [ca.norm_2(thrust_xy_sp[i, :]) for i in range(thrust_xy_sp.shape[0])]
        mags = ca.vertcat(*mags)

        conditioned_values = thrust_xy_sp / mags * thrust_max_xy_min

        # Replicating the mask for each dimension (similar to unsqueeze operation)
        mask_repeated = ca.repmat(mask, 1, thrust_xy_sp.shape[1])

        # Using if_else for element-wise conditional operation
        thrust_xy_sp_masked = ca.if_else(mask_repeated, conditioned_values, thrust_xy_sp)
        # # concatenate the z and xy thrust set points together 
        # thrust_sp = np.hstack([thrust_xy_sp_masked, thrust_z_sp.unsqueeze(1)])
        thrust_sp = ca.vertcat(thrust_xy_sp_masked.T, thrust_z_sp).T
        return thrust_sp

    def attitude_control(self, dcm, quat, thrust_sp, qd_full):

        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = dcm[:,2]
        # e_z_d = -utils.vectNormalize_batched_ca(thrust_sp)
        e_z_d = -utils.rotation.normalize_vector.casadi(thrust_sp)


        qe_red = ca.MX.zeros(quat.shape)
        qe_red[:,0] = ca.mtimes(e_z_d, e_z) + ca.sqrt(ca.norm_2(e_z)**2 * ca.norm_2(e_z_d)**2)

        # qe_red[:,1:4] = np.cross(e_z, e_z_d)
        # qe_red = utils.vectNormalize_batched_ca(qe_red)
        # Transpose e_z_d to make it a column vector
        e_z_d_col = e_z_d.T

        # Compute the cross product
        cross_prod = ca.vertcat(
            e_z[1] * e_z_d_col[2] - e_z[2] * e_z_d_col[1],
            e_z[2] * e_z_d_col[0] - e_z[0] * e_z_d_col[2],
            e_z[0] * e_z_d_col[1] - e_z[1] * e_z_d_col[0]
        )

        # Assign to qe_red
        qe_red[:, 1:4] = cross_prod
                
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        # qd_red = utils.quatMultiply_batched_ca(qe_red, quat)
        qd_red = utils.rotation.quaternion_multiply.casadi_vectorized(qe_red, quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        # q_mix = utils.quatMultiply_batched_ca(utils.inverse_batched_ca(qd_red), qd_full)
        q_mix = utils.rotation.quaternion_multiply.casadi_vectorized(utils.rotation.quaternion_inverse.casadi_vectorized(qd_red), qd_full)

        # q_mix = q_mix*np.sign(q_mix[:,0:1])
        q_mix = q_mix * ca.sign(q_mix[:,0])

        q0 = ca.fmax(ca.fmin(q_mix[:,0], 1.0), -1.0)
        q1 = ca.MX.zeros(self.bs)
        q2 = ca.MX.zeros(self.bs)
        q3 = ca.fmax(ca.fmin(q_mix[:,3], 1.0), -1.0)
        multiplier = ca.horzcat(q0, q1, q2, q3)
        # qd = utils.quatMultiply_batched_ca(qd_red, multiplier)
        qd = utils.rotation.quaternion_multiply.casadi_vectorized(qd_red, multiplier)

        # Resulting error quaternion
        # qe = utils.quatMultiply_batched_ca(utils.inverse_batched_ca(quat), qd)
        qe = utils.rotation.quaternion_multiply.casadi_vectorized(utils.rotation.quaternion_inverse.casadi_vectorized(quat), qd)

        # Create rate setpoint from quaternion error
        # rate_sp = (2.0*np.sign(qe[:,0:1])*qe[:,1:4])*self.ctrl_params["att_P_gain"]
        rate_sp = (2.0 * ca.sign(qe[:,0]) * qe[:,1:4]) * self.ctrl_params["att_P_gain"][None,:]

        # Add Yaw rate feed-forward term clipped by rate limits calculated by yaw rate weighting in ctrl_params
        # rate_sp += utils.quat2Dcm_batched_ca(utils.inverse_batched_ca(quat))[:,:,2]*self.ctrl_params["yawFF"]
        # rate_sp = rate_sp + ca.mtimes(utils.quat2Dcm_batched_ca(utils.inverse_batched_ca(quat))[:,2], self.ctrl_params["yawFF"]).T
        rate_sp += utils.rotation.quaternion_to_dcm.casadi(utils.rotation.quaternion_inverse.casadi_vectorized(quat))[:,2].T*self.ctrl_params["yawFF"]

        # Limit rate setpoint
        # rate_sp = np.clip(rate_sp, -self.ctrl_params["rateMax"], self.ctrl_params["rateMax"])
        rate_sp = ca.fmax(ca.fmin(rate_sp, self.ctrl_params["rateMax"][None,:]), -self.ctrl_params["rateMax"][None,:])

        return rate_sp


    def rate_control(self, omega, omega_dot, rate_sp):
        
        # Rate Control
        # ---------------------------
        rate_error = rate_sp - omega
        rateCtrl = self.ctrl_params["rate_P_gain"][None,:]*rate_error - self.ctrl_params["rate_D_gain"][None,:]*omega_dot     # Be sure it is right sign for the D part
        
        return rateCtrl
    
    def w_control(self, w, w_cmd):

        # the above is the commanded omega
        w_error = w_cmd.T - w
        p_gain = self.quad_params["IRzz"] / self.Ts
        motor_inputs = w_error * p_gain
        return motor_inputs
    
    
if __name__ == "__main__":

    from tqdm import tqdm
    import numpy as np
    from utils.quad import Animator
    import copy
    import matplotlib.pyplot as plt
    from dynamics import state_dot, get_quad_params
    import reference
    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()

    class Visualiser():
        def __init__(self, reference=reference.waypoint('wp_p2p', average_vel=1.6)) -> None:
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

            animator = Animator(np.vstack(self.x),
                                np.array(self.t), 
                                np.vstack(self.x), 
                                max_frames=10, 
                                save_path='data')
            animator.animate()

    vis = Visualiser()  # Assuming a CasADi compatible visualizer

    Ts = 0.01
    ctrl = PID_CA(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params, input="xyz_thr")
    R = reference.waypoint('wp_p2p', average_vel=1.6)  # Assuming a way to convert or use reference in CasADi context
    state = np.array(quad_params["default_init_state_pt"])

    # initial conditions
    x = ca.DM(state).T  # Ensure x is column vector if needed


    x_sym = ca.MX.sym('x', len(state))  # Symbolic variable for state
    u_sym = ca.MX.sym('u', 4)  # Adjust the size according to your control input dimension

    x_dot_sym = state_dot.casadi(x_sym, u_sym, params=quad_params)  # Symbolic state derivative

    # Create a CasADi function for numerical evaluation
    x_dot_fun = ca.Function('x_dot_fun', [x_sym, u_sym], [x_dot_sym])

    # Now in your simulation loop, instead of x += x_dot * Ts symbolically:
    for i in tqdm(range(1000)):
        t = i * Ts
        r = R(t)  # Adjust to CasADi-compatible reference retrieval
        thr_sp = ca.DM(np.array([0, 0, -quad_params["hover_thr"]]))  # Adjust thrust setpoint as needed

        u = ctrl(x, thr_sp)
        
        # Evaluate the state derivative numerically
        x_dot_num = x_dot_fun(x, u)
        
        # Update state numerically
        x += x_dot_num * Ts

        if i % 20 == 0:
            vis.save(x.full().flatten(), u.full().flatten(), r, t)  # Adapt to ensure correct data handling

    vis.animate()
    plt.show()

    vis = Visualiser()

    print("testing the classical control system from the github repo")
    Ts = 0.01
    ctrl = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params, input="xyz_thr")
    R = reference.waypoint('wp_p2p', average_vel=1.6)
    state = quad_params["default_init_state_pt"]#[0:13]
    batch_size = 1

    # initial conditions 2D
    x = torch.vstack([state]*batch_size)

    # need convert to NED
    x[:,2] *= -1
    x[:,9] *= -1

    for i in tqdm(range(1000)):
        t = i*Ts
        r = ptu.from_numpy(R(t))
        # vel_sp = torch.vstack([torch.cos(torch.tensor([t*1, t*1, t*2])) * 5] * batch_size) * 1
        thr_sp = torch.vstack([torch.cos(torch.tensor([0, 0, 0])) * 0] * batch_size) * 1
        gravity_offset = torch.vstack([torch.tensor([0,0,-quad_params["hover_thr"]])] * batch_size)
        thr_sp += gravity_offset
        u, pid_x = ctrl(x, thr_sp)

        x[0,2] *= -1
        x[0,9] *= -1
        r[2] *= -1
        r[9] *= -1

        if i % 20 == 0:
            if x[0,:].isnan().any() or u[0,:].isnan().any():
                print(f"x: {x[0,:]}")
                print(f"u: {u[0,:]}")
                vis.animate()
                plt.show()   

            vis.save(x[0,:],u[0,:],r,t)

        x += state_dot.neuromancer(x, u, params=quad_params) * Ts
        print(f"x: {x[0,:]}")

        x[0,2] *= -1
        x[0,9] *= -1
        r[2] *= -1
        r[9] *= -1

    vis.animate()
    plt.show()

    print('fin')
