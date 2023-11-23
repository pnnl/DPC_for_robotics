import torch
import numpy as np
import casadi as ca
import utils.pytorch as ptu

class euler_to_quaternion:

    @staticmethod
    def pytorch_batched(euler_angles: torch.Tensor):
        # euler_angles is a tensor of shape (batch_size, 3)
        # where each row is (roll, pitch, yaw)
        rolls = euler_angles[:, 0]
        pitches = euler_angles[:, 1]
        yaws = euler_angles[:, 2]

        cys = torch.cos(yaws * 0.5)
        sys = torch.sin(yaws * 0.5)
        cps = torch.cos(pitches * 0.5)
        sps = torch.sin(pitches * 0.5)
        crs = torch.cos(rolls * 0.5)
        srs = torch.sin(rolls * 0.5)

        q0s = crs * cps * cys + srs * sps * sys
        q1s = srs * cps * cys - crs * sps * sys
        q2s = crs * sps * cys + srs * cps * sys
        q3s = crs * cps * sys - srs * sps * cys

        quaternions = torch.stack((q0s, q1s, q2s, q3s), dim=1)
        # Normalize each quaternion
        norms = torch.norm(quaternions, p=2, dim=1, keepdim=True)
        quaternions_normalized = quaternions / norms
        
        return quaternions_normalized
    
    @staticmethod
    def pytorch(euler_angles: np.ndarray):
        raise NotImplementedError
    
    @staticmethod
    def casadi(euler_angles: ca.MX):
        raise NotImplementedError

    @staticmethod
    def numpy_batched(euler_angles: np.ndarray):
        raise NotImplementedError
    
    @staticmethod
    def numpy(euler_angles: np.ndarray):
        # euler_angles is a tensor of shape (batch_size, 3)
        # where each row is (roll, pitch, yaw)
        rolls = euler_angles[0]
        pitches = euler_angles[1]
        yaws = euler_angles[2]

        cys = np.cos(yaws * 0.5)
        sys = np.sin(yaws * 0.5)
        cps = np.cos(pitches * 0.5)
        sps = np.sin(pitches * 0.5)
        crs = np.cos(rolls * 0.5)
        srs = np.sin(rolls * 0.5)

        q0s = crs * cps * cys + srs * sps * sys
        q1s = srs * cps * cys - crs * sps * sys
        q2s = crs * sps * cys + srs * cps * sys
        q3s = crs * cps * sys - srs * sps * cys

        quaternions = np.stack((q0s, q1s, q2s, q3s))
        # Normalize each quaternion
        norms = np.linalg.norm(quaternions, ord=2)
        quaternions_normalized = quaternions / norms
        
        return quaternions_normalized

class quaternion_to_euler:

    @staticmethod
    def pytorch_batched(quaternions: torch.Tensor):
        # quaternions is a tensor of shape (batch_size, 4)
        # where each row is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        
        # Extracting the values from each quaternion
        q0 = quaternions[:, 0]
        q1 = quaternions[:, 1]
        q2 = quaternions[:, 2]
        q3 = quaternions[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1.pow(2) + q2.pow(2))
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2.pow(2) + q3.pow(2))
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.vstack([roll, pitch, yaw]).T
    
    @staticmethod
    def pytorch(quaternion: np.ndarray):
        # quaternion is a tensor of shape (4,)
        # where the array is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternion = quaternion / quaternion.norm()
        
        # Extracting the values from the quaternion
        q0, q1, q2, q3 = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = torch.asin(sinp) if torch.abs(sinp) < 1 else torch.sign(sinp) * torch.pi / 2

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.tensor([roll, pitch, yaw])
    
    @staticmethod
    def casadi(quaternion: None):
        raise NotImplementedError

    @staticmethod
    def numpy_batched(quaternions: np.ndarray):
        # quaternions is a numpy array of shape (batch_size, 4)
        # where each row is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Extracting the values from each quaternion
        q0 = quaternions[:, 0]
        q1 = quaternions[:, 1]
        q2 = quaternions[:, 2]
        q3 = quaternions[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = np.where(
            np.abs(sinp) >= 1,
            np.sign(sinp) * np.pi / 2,
            np.arcsin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.vstack([roll, pitch, yaw]).T
    
    @staticmethod
    def numpy(quaternion: np.ndarray):
        # quaternion is a numpy array of shape (4,)
        # where the array is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Extracting the values from the quaternion
        q0, q1, q2, q3 = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = np.arcsin(sinp) if np.abs(sinp) < 1 else np.sign(sinp) * np.pi / 2

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

class euler_to_rot_matrix:

    @staticmethod
    def pytorch_batched(roll, pitch, yaw):
        # Assumes the angles are in radians and the order is 'XYZ'
        cos = torch.cos
        sin = torch.sin

        # Pre-calculate cosines and sines
        cos_roll = cos(roll)
        sin_roll = sin(roll)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)

        # Calculate rotation matrix components
        R_x = torch.stack([
            torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
            torch.zeros_like(roll), cos_roll, -sin_roll,
            torch.zeros_like(roll), sin_roll, cos_roll
        ], dim=1).reshape(-1, 3, 3)

        R_y = torch.stack([
            cos_pitch, torch.zeros_like(pitch), sin_pitch,
            torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
            -sin_pitch, torch.zeros_like(pitch), cos_pitch
        ], dim=1).reshape(-1, 3, 3)

        R_z = torch.stack([
            cos_yaw, -sin_yaw, torch.zeros_like(yaw),
            sin_yaw, cos_yaw, torch.zeros_like(yaw),
            torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        ], dim=1).reshape(-1, 3, 3)

        # The final rotation matrix combines rotations around all axes
        R = torch.bmm(torch.bmm(R_z, R_y), R_x)

        return R
    
    @staticmethod
    def pytorch(roll, pitch, yaw):
        raise NotImplementedError
    
    @staticmethod
    def casadi(roll, pitch, yaw):
        raise NotImplementedError    
    
    @staticmethod
    def numpy_batched(roll, pitch, yaw):
        raise NotImplementedError    
    
    @staticmethod
    def numpy(roll, pitch, yaw):
        # Assumes the angles are in radians and the order is 'XYZ'
        cos = np.cos
        sin = np.sin

        # Pre-calculate cosines and sines
        cos_roll = cos(roll)
        sin_roll = sin(roll)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)

        # Calculate rotation matrix components
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])

        R_y = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])

        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # The final rotation matrix combines rotations around all axes
        R = R_z @ R_y @ R_x

        return R

class rot_matrix_to_quaternion:

    @staticmethod
    def casadi_vectorized(R: ca.MX) -> ca.MX:
        
        R11 = R[0, 0]
        R12 = R[0, 1]
        R13 = R[0, 2]
        R21 = R[1, 0]
        R22 = R[1, 1]
        R23 = R[1, 2]
        R31 = R[2, 0]
        R32 = R[2, 1]
        R33 = R[2, 2]

        tr = R11 + R22 + R33

        # We will calculate the values for all the conditions and then use if_else to select the correct one
        e0_cond1 = 0.5 * ca.sqrt(1 + tr)
        r_cond1 = 0.25 / e0_cond1
        e1_cond1 = (R32 - R23) * r_cond1
        e2_cond1 = (R13 - R31) * r_cond1
        e3_cond1 = (R21 - R12) * r_cond1

        e1_cond2 = 0.5 * ca.sqrt(1 - tr + 2*R11)
        r_cond2 = 0.25 / e1_cond2
        e0_cond2 = (R32 - R23) * r_cond2
        e2_cond2 = (R12 + R21) * r_cond2
        e3_cond2 = (R13 + R31) * r_cond2

        e2_cond3 = 0.5 * ca.sqrt(1 - tr + 2*R22)
        r_cond3 = 0.25 / e2_cond3
        e0_cond3 = (R13 - R31) * r_cond3
        e1_cond3 = (R12 + R21) * r_cond3
        e3_cond3 = (R23 + R32) * r_cond3

        e3_cond4 = 0.5 * ca.sqrt(1 - tr + 2*R33)
        r_cond4 = 0.25 / e3_cond4
        e0_cond4 = (R21 - R12) * r_cond4
        e1_cond4 = (R13 + R31) * r_cond4
        e2_cond4 = (R23 + R32) * r_cond4

        # Masking using if_else
        # Define common conditions for reuse
        condition_1 = ca.logic_and(ca.logic_and(tr > R11, tr > R22), tr > R33)
        condition_2 = ca.logic_and(R11 > R22, R11 > R33)
        condition_3 = R22 > R33

        # Masking using if_else
        e0 = ca.if_else(condition_1, e0_cond1, 
            ca.if_else(condition_2, e0_cond2,
            ca.if_else(condition_3, e0_cond3, e0_cond4)))

        e1 = ca.if_else(condition_1, e1_cond1, 
            ca.if_else(condition_2, e1_cond2,
            ca.if_else(condition_3, e1_cond3, e1_cond4)))

        e2 = ca.if_else(condition_1, e2_cond1, 
            ca.if_else(condition_2, e2_cond2,
            ca.if_else(condition_3, e2_cond3, e2_cond4)))

        e3 = ca.if_else(condition_1, e3_cond1, 
            ca.if_else(condition_2, e3_cond2,
            ca.if_else(condition_3, e3_cond3, e3_cond4)))

        # Concatenating the elements to create the quaternion
        q = ca.horzcat(e0, e1, e2, e3)

        # Making sure the scalar part of quaternion is non-negative
        q = q * ca.sign(e0[0])

        # Normalize the quaternion
        magnitude = ca.sqrt(ca.sum1(q*q))
        q_norm = q / magnitude

        return q_norm

    @staticmethod
    def pytorch_vectorized(R: torch.Tensor) -> torch.Tensor:
    
        R11 = R[:, 0, 0]
        R12 = R[:, 0, 1]
        R13 = R[:, 0, 2]
        R21 = R[:, 1, 0]
        R22 = R[:, 1, 1]
        R23 = R[:, 1, 2]
        R31 = R[:, 2, 0]
        R32 = R[:, 2, 1]
        R33 = R[:, 2, 2]
        # From page 68 of MotionGenesis book
        tr = R11 + R22 + R33

        e0 = ptu.zeros(tr.shape)
        e1 = ptu.zeros(tr.shape)
        e2 = ptu.zeros(tr.shape)
        e3 = ptu.zeros(tr.shape)

        # mask1 = (tr > R11) & (tr > R22) & (tr > R33)
        # mask2 = (R11 > R22) & (R11 > R33)
        # mask3 = R22 > R33
        # mask4 = ~mask1 & ~mask2 & ~mask3  # This is equivalent to else

        mask1 = (tr > R11) & (tr > R22) & (tr > R33)
        mask2 = (R11 > R22) & (R11 > R33) & ~mask1
        mask3 = (R22 > R33) & ~mask1 & ~mask2
        mask4 = ~mask1 & ~mask2 & ~mask3
        
        # Condition 1
        e0[mask1] = 0.5 * torch.sqrt(1 + tr[mask1])
        r = 0.25 / e0[mask1]
        e1[mask1] = (R32[mask1] - R23[mask1]) * r
        e2[mask1] = (R13[mask1] - R31[mask1]) * r
        e3[mask1] = (R21[mask1] - R12[mask1]) * r

        # Condition 2
        e1[mask2] = 0.5 * torch.sqrt(1 - tr[mask2] + 2*R11[mask2])
        r = 0.25 / e1[mask2]
        e0[mask2] = (R32[mask2] - R23[mask2]) * r
        e2[mask2] = (R12[mask2] + R21[mask2]) * r
        e3[mask2] = (R13[mask2] + R31[mask2]) * r

        # Condition 3
        e2[mask3] = 0.5 * torch.sqrt(1 - tr[mask3] + 2*R22[mask3])
        r = 0.25 / e2[mask3]
        e0[mask3] = (R13[mask3] - R31[mask3]) * r
        e1[mask3] = (R12[mask3] + R21[mask3]) * r
        e3[mask3] = (R23[mask3] + R32[mask3]) * r

        # Condition 4 (else)
        e3[mask4] = 0.5 * torch.sqrt(1 - tr[mask4] + 2*R33[mask4])
        r = 0.25 / e3[mask4]
        e0[mask4] = (R21[mask4] - R12[mask4]) * r
        e1[mask4] = (R13[mask4] + R31[mask4]) * r
        e2[mask4] = (R23[mask4] + R32[mask4]) * r

        q = torch.stack([e0, e1, e2, e3], dim=1)
        q_sign = q * torch.sign(q[:, 0]).unsqueeze(1)  # unsqueeze to make it [100, 1] for broadcasting
        magnitude = torch.sqrt(torch.sum(q_sign**2, dim=1)).unsqueeze(1)
        q_norm = q_sign / magnitude

        # if tr > R11 and tr > R22 and tr > R33:
        #     e0 = 0.5 * torch.sqrt(1 + tr)
        #     r = 0.25 / e0
        #     e1 = (R32 - R23) * r
        #     e2 = (R13 - R31) * r
        #     e3 = (R21 - R12) * r
        # elif R11 > R22 and R11 > R33:
        #     e1 = 0.5 * torch.sqrt(1 - tr + 2*R11)
        #     r = 0.25 / e1
        #     e0 = (R32 - R23) * r
        #     e2 = (R12 + R21) * r
        #     e3 = (R13 + R31) * r
        # elif R22 > R33:
        #     e2 = 0.5 * torch.sqrt(1 - tr + 2*R22)
        #     r = 0.25 / e2
        #     e0 = (R13 - R31) * r
        #     e1 = (R12 + R21) * r
        #     e3 = (R23 + R32) * r
        # else:
        #     e3 = 0.5 * torch.sqrt(1 - tr + 2*R33)
        #     r = 0.25 / e3
        #     e0 = (R21 - R12) * r
        #     e1 = (R13 + R31) * r
        #     e2 = (R23 + R32) * r

        # e0,e1,e2,e3 = qw,qx,qy,qz
        # q = torch.stack([e0,e1,e2,e3])
        # q = q*torch.sign(e0)
        
        # q = q/torch.sqrt(torch.sum(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2))

        if q.isnan().any():
            print('fin')
        
        return q_norm

class quaternion_derivative:
    # Functions to calculate the quaternion derivative given quaternion and angular velocity

    @staticmethod
    def pytorch_batched(q: torch.Tensor, omega: torch.Tensor):
        # Assuming q is of shape [batch, 4] with q = [q0, q1, q2, q3]
        # omega is of shape [batch, 3] with omega = [p, q, r]
        
        # Quaternion multiplication matrix
        Q_mat = torch.cat([
            -q[:, 1:2], -q[:, 2:3], -q[:, 3:4],
            q[:, 0:1], -q[:, 3:4],  q[:, 2:3],
            q[:, 3:4],  q[:, 0:1], -q[:, 1:2],
            -q[:, 2:3],  q[:, 1:2],  q[:, 0:1]
        ], dim=1).view(-1, 4, 3)

        # Multiply by the angular velocity
        q_dot = 0.5 * torch.bmm(Q_mat, omega.unsqueeze(-1)).squeeze(-1)
        return q_dot
    
    @staticmethod
    def pytorch(q, omega):
        raise NotImplementedError
    
    @staticmethod
    def casadi(q, omega):
        # Assuming q is a column vector [4, 1] with q = [q0, q1, q2, q3]
        # omega is a column vector [3, 1] with omega = [p, q, r]
        
        # Quaternion multiplication matrix
        Q_mat = ca.vertcat(
            ca.horzcat(-q[1], -q[2], -q[3]),
            ca.horzcat( q[0], -q[3],  q[2]),
            ca.horzcat( q[3],  q[0], -q[1]),
            ca.horzcat(-q[2],  q[1],  q[0])
        )

        # Multiply by the angular velocity to get the quaternion derivative
        q_dot = 0.5 * ca.mtimes(Q_mat, omega)

        return q_dot 
    
    @staticmethod
    def numpy_batched(q, omega):
        raise NotImplementedError  

class quaternion_conjugate:

    @staticmethod
    def casadi(q):
        return ca.vertcat(q[0,:], -q[1,:], -q[2,:], -q[3,:])
    
    @staticmethod
    def pytorch_vectorized(q):
        return torch.hstack([q[:,:1], -q[:,1:]])
    
    @staticmethod
    def neuromancer(q):
        return torch.stack([q[:,:,:1], -q[:,:,1:]], dim=2)
    
class quaternion_multiply:

    @staticmethod
    def casadi(q1, q2):
        # Quaternion multiplication (Hamilton product)
        w1, x1, y1, z1 = q1[0,:], q1[1,:], q1[2,:], q1[3,:]
        w2, x2, y2, z2 = q2[0,:], q2[1,:], q2[2,:], q2[3,:]
        return ca.vertcat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    @staticmethod
    def casadi_vectorized(q, p: ca.MX) -> ca.MX:

        row0 = ca.horzcat(q[:,0], -q[:,1], -q[:,2], -q[:,3])
        row1 = ca.horzcat(q[:,1],  q[:,0], -q[:,3],  q[:,2])
        row2 = ca.horzcat(q[:,2],  q[:,3],  q[:,0], -q[:,1])
        row3 = ca.horzcat(q[:,3], -q[:,2],  q[:,1],  q[:,0])

        Q = ca.vertcat(row0, row1, row2, row3)

        # In CasADi, we perform matrix multiplication using mtimes
        mult = ca.mtimes(Q, p.T).T

        return mult
    
    @staticmethod
    def pytorch(q, p: torch.Tensor) -> torch.Tensor:

        row0 = torch.stack([q[0], -q[1], -q[2], -q[3]])
        row1 = torch.stack([q[1],  q[0], -q[3],  q[2]])
        row2 = torch.stack([q[2],  q[3],  q[0], -q[1]])
        row3 = torch.stack([q[3], -q[2],  q[1],  q[0]])

        Q = torch.vstack([row0,row1,row2,row3])

        # Q = np.array([[q[0], -q[1], -q[2], -q[3]],
        #               [q[1],  q[0], -q[3],  q[2]],
        #               [q[2],  q[3],  q[0], -q[1]],
        #               [q[3], -q[2],  q[1],  q[0]]])

        return Q@p

    @staticmethod
    def pytorch_vectorized(q, p: torch.Tensor) -> torch.Tensor:

        row0 = torch.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]], dim=1)
        row1 = torch.stack([q[:,1],  q[:,0], -q[:,3],  q[:,2]], dim=1)
        row2 = torch.stack([q[:,2],  q[:,3],  q[:,0], -q[:,1]], dim=1)
        row3 = torch.stack([q[:,3], -q[:,2],  q[:,1],  q[:,0]], dim=1)

        Q = torch.stack([row0,row1,row2,row3], dim=1)

        # Q = np.array([[q[0], -q[1], -q[2], -q[3]],
        #               [q[1],  q[0], -q[3],  q[2]],
        #               [q[2],  q[3],  q[0], -q[1]],
        #               [q[3], -q[2],  q[1],  q[0]]])
        mult = torch.bmm(Q, p.unsqueeze(-1)).squeeze(-1)
        return mult
    
    @staticmethod
    def neuromancer(q, p):
        row0 = torch.stack([q[:,:,0], -q[:,:,1], -q[:,:,2], -q[:,:,3]], dim=-1)
        row1 = torch.stack([q[:,:,1],  q[:,:,0], -q[:,:,3],  q[:,:,2]], dim=-1)
        row2 = torch.stack([q[:,:,2],  q[:,:,3],  q[:,:,0], -q[:,:,1]], dim=-1)
        row3 = torch.stack([q[:,:,3], -q[:,:,2],  q[:,:,1],  q[:,:,0]], dim=-1)

        Q = torch.stack([row0, row1, row2, row3], dim=-2)

        mult = torch.squeeze(torch.bmm(Q, torch.unsqueeze(p, dim=-1)), dim=-1)
        return mult 

class quaternion_error:

    @staticmethod
    def casadi(q, q_desired):
        # Calculate the quaternion error
        q_desired_conjugate = quaternion_conjugate.casadi(q_desired)
        return quaternion_multiply.casadi(q, q_desired_conjugate)

    @staticmethod
    def pytorch_vectorized(q, q_desired):
        # Calculate the quaternion error
        q_desired_conjugate = quaternion_conjugate.pytorch_vectorized(q_desired)
        return quaternion_multiply.pytorch_vectorized(q, q_desired_conjugate)

    @staticmethod
    def neuromancer(q, q_desired):
        # Calculate the quaternion error
        q_desired_conjugate = quaternion_conjugate.neuromancer(q_desired)
        return quaternion_multiply.neuromancer(q, q_desired_conjugate)

class quaternion_to_dcm:

    @staticmethod
    def numpy(q):
        dcm = np.zeros([3,3])

        dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
        dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
        dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
        dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
        dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
        dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
        dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

        return dcm
    
    @staticmethod
    def pytorch_vectorized(q: torch.Tensor) -> torch.Tensor:

        # expects q first dimension to be batch
        bs = q.shape[0]

        dcm = ptu.zeros([bs,3,3])

        dcm[:,0,0] = q[:,0]**2 + q[:,1]**2 - q[:,2]**2 - q[:,3]**2
        dcm[:,0,1] = 2.0*(q[:,1]*q[:,2] - q[:,0]*q[:,3])
        dcm[:,0,2] = 2.0*(q[:,1]*q[:,3] + q[:,0]*q[:,2])
        dcm[:,1,0] = 2.0*(q[:,1]*q[:,2] + q[:,0]*q[:,3])
        dcm[:,1,1] = q[:,0]**2 - q[:,1]**2 + q[:,2]**2 - q[:,3]**2
        dcm[:,1,2] = 2.0*(q[:,2]*q[:,3] - q[:,0]*q[:,1])
        dcm[:,2,0] = 2.0*(q[:,1]*q[:,3] - q[:,0]*q[:,2])
        dcm[:,2,1] = 2.0*(q[:,2]*q[:,3] + q[:,0]*q[:,1])
        dcm[:,2,2] = q[:,0]**2 - q[:,1]**2 - q[:,2]**2 + q[:,3]**2

        return dcm

class quaternion_inverse:

    @staticmethod
    def pytorch(q: torch.Tensor) -> torch.Tensor:
        qinv = torch.stack([q[0], -q[1], -q[2], -q[3]])/torch.linalg.norm(q)
        return qinv
    
    @staticmethod
    def pytorch_vectorized(q: torch.Tensor) -> torch.Tensor:
        qinv = torch.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]])/torch.linalg.norm(q, axis=1)
        return qinv.T

    # def inverse_batched_ca(q: ca.MX) -> ca.MX:
    #     qnorm = ca.sqrt(ca.mtimes(q, q.T).diagonal())
    #     qinv = ca.horzcat([q[:,0], -q[:,1], -q[:,2], -q[:,3]]) / ca.mtimes(qnorm, ca.MX.ones(1,4))
    #     return qinv

    @staticmethod
    def casadi_vectorized(q: ca.MX) -> ca.MX:
        # Compute the squared norm of each quaternion (each row)
        qnorm2 = ca.mtimes(q, q.T)

        # Take the square root to get the norm
        qnorm = ca.sqrt(qnorm2)

        # Compute the inverse for each quaternion
        qinv = ca.MX.zeros(q.size1(), q.size2())
        qinv[:, 0] = q[:, 0] / qnorm
        qinv[:, 1] = -q[:, 1] / qnorm
        qinv[:, 2] = -q[:, 2] / qnorm
        qinv[:, 3] = -q[:, 3] / qnorm

        return qinv

class normalize_vector:

    @staticmethod
    def pytorch(q: torch.Tensor) -> torch.Tensor:
        return q/torch.linalg.norm(q)

    @staticmethod
    def pytorch_vectorized(q: torch.Tensor, norm_dim=1) -> torch.Tensor:
        normalised_vecs = torch.linalg.norm(q, dim=norm_dim).unsqueeze(1)
        return q/normalised_vecs

class quaternion_derivative_to_angular_velocity:
    """
    Convert a batch of quaternion derivatives to angular velocities (p, q, r) in a vectorized manner.
    
    :param q: Batch of quaternions, shape [batch_size, 4]
    :param q_dot: Batch of quaternion derivatives, shape [batch_size, 4]
    :return: Batch of angular velocities [p, q, r], shape [batch_size, 3]
    """
    @staticmethod
    def pytorch_vectorized(q, q_dot):
        p = 2 * (q_dot[:, 1] * q[:, 0] - q_dot[:, 0] * q[:, 1] - q_dot[:, 2] * q[:, 3] + q_dot[:, 3] * q[:, 2])
        q = 2 * (q_dot[:, 2] * q[:, 0] - q_dot[:, 0] * q[:, 2] - q_dot[:, 3] * q[:, 1] + q_dot[:, 1] * q[:, 3])
        r = 2 * (q_dot[:, 3] * q[:, 0] - q_dot[:, 0] * q[:, 3] - q_dot[:, 1] * q[:, 2] + q_dot[:, 2] * q[:, 1])

        return torch.stack([p, q, r], dim=1)


if __name__ == "__main__":
    
    import utils.pytorch as ptu

    # Example usage:
    # Define a batch of Euler angles (roll, pitch, yaw) in radians
    euler_angles_batch = ptu.tensor([
        [30, 45, 60],
        [10, 20, 30],
        # ... more angles
    ]) * torch.pi/180  # Convert degrees to radians

    # Convert to quaternions
    quaternions_batch = euler_to_quaternion.pytorch_batched(euler_angles_batch)

    # Print the quaternions
    print("Quaternions:\n", quaternions_batch)

    # Convert to Euler angles
    eul_batch = quaternion_to_euler.pytorch_batched(quaternions_batch)

    # Print the Euler angles in radians
    print(f"eul_batch: {eul_batch * 180/torch.pi}")