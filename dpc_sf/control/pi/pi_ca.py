"""
Adapting classical_control2_func to recieve (batch, state)

COMPLETE - asides from the velocity saturation
"""

import math
import dpc_sf.utils as utils
import casadi as ca
from dpc_sf.dynamics.eom_pt import state_dot_nm, state_dot_pt
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference

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


class XYZ_Vel():

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
            Ts = 0.1,
            bs = 100,
            ctrl_params = ctrl_params,
            quad_params = quad_params,
            include_actuators=False,
            verbose=False,
            input="xyz_vel", # 'xyz_vel', 'xyz_thr'
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

        x = x.T
        sp = sp.T

        quat = x[:,3:7]
        vel = x[:,7:10]
        omega = x[:,10:13]

        eul_sp = ca.DM.zeros(vel.shape)
        acc_sp = ca.DM.zeros(vel.shape)

        dcm = utils.quat2Dcm_batched_ca(quat)

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
        w_cmd = utils.mixerFM_batched_ca(self.quad_params, norms, rateCtrl)

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
        self.thr_int = ca.MX.zeros(self.bs, 3)
        self.vel_old = ca.MX.zeros(self.bs, 3)
        self.omega_old = ca.MX.zeros(self.bs, 3)

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
        body_z = -utils.vectNormalize_batched_ca(thrust_sp)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        # y_C = np.vstack([-np.sin(yaw_sp), np.cos(yaw_sp), np.zeros(self.bs)]).T
        y_C = ca.vertcat(-ca.sin(yaw_sp), ca.cos(yaw_sp), ca.MX.zeros(self.bs)).T

        # Desired body_x axis direction
        # body_x = np.cross(y_C, body_z)
        # body_x = utils.vectNormalize_batched(body_x)
        body_x = ca.cross(y_C, body_z)
        body_x = utils.vectNormalize_batched_ca(body_x)  # Assuming this is the CasADi version of the function

        # Desired body_y axis direction
        # body_y = np.cross(body_z, body_x)
        body_y = ca.cross(body_z, body_x)

        # Desired rotation matrix - permute does the transpose on the batched 3x3 matrices
        # R_sp = np.stack([body_x, body_y, body_z], dim=1).permute(0,2,1)
        # R_sp = np.vstack([body_x, body_y, body_z]).T
        R_sp = ca.vertcat(body_x, body_y, body_z).T  # This will create the matrix using the three vectors as its columns

        # Full desired quaternion (full because it considers the desired Yaw angle)
        qd_full = utils.RotToQuat_batched_ca(R_sp)

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
        # thrust_max_xy_tilt = np.abs(thrust_z_sp)*np.tan(self.ctrl_params["tiltMax"])
        # thrust_max_xy = np.sqrt(self.quad_params["maxThr"]**2 - thrust_z_sp**2)
        # thrust_max_xy_min = np.min(thrust_max_xy, thrust_max_xy_tilt)

        # # Saturate thrust in NE-direction
        # mask = ((thrust_xy_sp ** 2).sum(dim=1) > thrust_max_xy_min**2)

        # # Calculate norms for each row where the condition is True
        # mags = np.linalg.norm(thrust_xy_sp[mask,:], dim=1)

        # # Update rows of thrust_xy_sp where the condition is True
        # # thrust_xy_sp[mask] = (thrust_xy_sp[mask,:].T / mags * thrust_max_xy_min[mask]).T

        # # we do this instead to avoid in place operations
        # # Calculate norms for each row where the condition is True
        # mags = np.linalg.norm(thrust_xy_sp, dim=1)
    
        # # Update rows of thrust_xy_sp where the condition is True
        # # print(f"thrust_xy_sp nan?: {thrust_xy_sp.isnan().any()}")
        # # print(f"thrust_max_xy_min nan?: {thrust_max_xy_min.isnan().any()}")
        # # print(f"mags nan?: {mags.isnan().any()}")

        # conditioned_values = (thrust_xy_sp.T / mags * thrust_max_xy_min).T
        # thrust_xy_sp_masked = np.where(mask.unsqueeze(-1), conditioned_values, thrust_xy_sp)
        
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
        e_z_d = -utils.vectNormalize_batched_ca(thrust_sp)

        # Quaternion error between the 2 vectors - TODO get np.dot to work properly with batched
        # qe_red = np.zeros(quat.shape)
        # qe_red[:,0] = np.sum(e_z * e_z_d, dim=1) + np.sqrt(np.linalg.norm(e_z, dim=1)**2 * np.linalg.norm(e_z_d, dim=1)**2)
        
        # qe_red = ca.MX.zeros(quat.shape)
        # qe_red[:,0] = ca.mtimes(e_z, e_z_d).diagonal() + ca.sqrt(ca.mtimes(ca.norm_2(e_z, "rows"), ca.norm_2(e_z_d, "rows")))
        # qe_red[:,1:4] = ca.cross(e_z, e_z_d)
        # qe_red = utils.vectNormalize_batched_ca(qe_red)

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
        qd_red = utils.quatMultiply_batched_ca(qe_red, quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply_batched_ca(utils.inverse_batched_ca(qd_red), qd_full)
        # q_mix = q_mix*np.sign(q_mix[:,0:1])
        q_mix = q_mix * ca.sign(q_mix[:,0])

        # q_mix[:,0] = np.clip(q_mix[:,0], -1.0, 1.0)
        # q_mix[:,3] = np.clip(q_mix[:,3], -1.0, 1.0)

        # print(f"q_mix nan: {q_mix.isnan().any()}")
        # print(f"q_mix abs max (near 1 leads to poor gradients/nans): {q_mix[:,0].abs().max()}")

        # q0ac = np.arccos(q_mix[:,0])
        # q0 = np.clip(q_mix[:,0], -1.0, 1.0)
        # # q0 = np.cos(self.ctrl_params["yaw_w"]*q0ac)
        # q1 = np.zeros(self.bs)
        # q2 = np.zeros(self.bs)
        # # q3 = np.sin(self.ctrl_params["yaw_w"]*np.arcsin(q_mix[:,3]))
        # q3 = np.clip(q_mix[:,3], -1.0, 1.0)
        # multiplier = np.vstack([q0, q1, q2, q3]).T
        # q0 = ca.mmax(ca.mmin(q_mix[:,0], 1.0), -1.0)
        # q1 = ca.MX.zeros(self.bs)
        # q2 = ca.MX.zeros(self.bs)
        # q3 = ca.mmax(ca.mmin(q_mix[:,3], 1.0), -1.0)
        # multiplier = ca.horzcat(q0, q1, q2, q3)
        # qd = utils.quatMultiply_batched_ca(qd_red, multiplier)

        q0 = ca.fmax(ca.fmin(q_mix[:,0], 1.0), -1.0)
        q1 = ca.MX.zeros(self.bs)
        q2 = ca.MX.zeros(self.bs)
        q3 = ca.fmax(ca.fmin(q_mix[:,3], 1.0), -1.0)
        multiplier = ca.horzcat(q0, q1, q2, q3)
        qd = utils.quatMultiply_batched_ca(qd_red, multiplier)

        # Resulting error quaternion
        qe = utils.quatMultiply_batched_ca(utils.inverse_batched_ca(quat), qd)

        # Create rate setpoint from quaternion error
        # rate_sp = (2.0*np.sign(qe[:,0:1])*qe[:,1:4])*self.ctrl_params["att_P_gain"]
        rate_sp = (2.0 * ca.sign(qe[:,0]) * qe[:,1:4]) * self.ctrl_params["att_P_gain"].T

        # Add Yaw rate feed-forward term clipped by rate limits calculated by yaw rate weighting in ctrl_params
        # rate_sp += utils.quat2Dcm_batched_ca(utils.inverse_batched_ca(quat))[:,:,2]*self.ctrl_params["yawFF"]
        rate_sp = rate_sp + ca.mtimes(utils.quat2Dcm_batched_ca(utils.inverse_batched_ca(quat))[:,2], self.ctrl_params["yawFF"]).T

        # Limit rate setpoint
        # rate_sp = np.clip(rate_sp, -self.ctrl_params["rateMax"], self.ctrl_params["rateMax"])
        rate_sp = ca.fmax(ca.fmin(rate_sp, self.ctrl_params["rateMax"].T), -self.ctrl_params["rateMax"].T)

        return rate_sp


    def rate_control(self, omega, omega_dot, rate_sp):
        
        # Rate Control
        # ---------------------------
        rate_error = rate_sp - omega
        rateCtrl = self.ctrl_params["rate_P_gain"].T*rate_error - self.ctrl_params["rate_D_gain"].T*omega_dot     # Be sure it is right sign for the D part
        
        return rateCtrl
    
    def w_control(self, w, w_cmd):

        # the above is the commanded omega
        w_error = w_cmd.T - w
        p_gain = self.quad_params["IRzz"] / self.Ts
        motor_inputs = w_error * p_gain
        return motor_inputs
    
