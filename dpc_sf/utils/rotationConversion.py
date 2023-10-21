# -*- coding: utf-8 -*-

"""First pass on pytorch conversion complete"""

import torch
from numpy import sin, cos
from numpy.linalg import norm
import numpy as np
import dpc_sf.utils.pytorch_utils as ptu
import casadi as ca

def quatToYPR_ZYX(q: torch.Tensor) -> torch.Tensor:
    # [q0 q1 q2 q3] = [w x y z]
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    YPR = threeaxisrot( 2.0*(q1*q2 + q0*q3), \
                        q0**2 + q1**2 - q2**2 - q3**2, \
                        -2.0*(q1*q3 - q0*q2), \
                        2.0*(q2*q3 + q0*q1), \
                        q0**2 - q1**2 - q2**2 + q3**2)

    # YPR = [Yaw, pitch, roll] = [psi, theta, phi]
    return YPR

def quatToYPR_ZYX_nm(q: torch.Tensor) -> torch.Tensor:

    # normalize the quaternion
    q = q / torch.sqrt(torch.sum(q**2, dim=1, keepdim=True))

    # [q0 q1 q2 q3] = [w x y z]
    q0 = q[:,0]
    q1 = q[:,1]
    q2 = q[:,2]
    q3 = q[:,3]
    
    YPR = threeaxisrot( 2.0*(q1*q2 + q0*q3), \
                        q0**2 + q1**2 - q2**2 - q3**2, \
                        -2.0*(q1*q3 - q0*q2), \
                        2.0*(q2*q3 + q0*q1), \
                        q0**2 - q1**2 - q2**2 + q3**2)

    # YPR = [Yaw, pitch, roll] = [psi, theta, phi]
    return YPR.T

def quatToYPR_ZYX_np(q: np.ndarray) -> np.ndarray:
    # [q0 q1 q2 q3] = [w x y z]
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    YPR = threeaxisrot_np( 2.0*(q1*q2 + q0*q3), \
                        q0**2 + q1**2 - q2**2 - q3**2, \
                        -2.0*(q1*q3 - q0*q2), \
                        2.0*(q2*q3 + q0*q1), \
                        q0**2 - q1**2 - q2**2 + q3**2)

    # YPR = [Yaw, pitch, roll] = [psi, theta, phi]
    return YPR

def threeaxisrot(r11, r12, r21, r31, r32: torch.Tensor) -> torch.Tensor:
    r1 = torch.arctan2(r11, r12)
    r2 = torch.arcsin(r21)
    # r2 = torch.arcsin(torch.clamp(r21, min=-1.0+1e-7, max=1.0-1e-7)) # add a clamp here

    r3 = torch.arctan2(r31, r32)

    return torch.stack([r1, r2, r3])

def threeaxisrot_np(r11, r12, r21, r31, r32: np.ndarray) -> np.ndarray:
    r1 = np.arctan2(r11, r12)
    r2 = np.arcsin(r21)
    r3 = np.arctan2(r31, r32)

    return np.stack([r1, r2, r3])

def YPRToQuat_nm(r1, r2, r3: np.array) -> np.array:
    # For ZYX, Yaw-Pitch-Roll
    # psi   = RPY[0] = r1
    # theta = RPY[1] = r2
    # phi   = RPY[2] = r3
    
    cr1 = torch.cos(0.5*r1)
    cr2 = torch.cos(0.5*r2)
    cr3 = torch.cos(0.5*r3)
    sr1 = torch.sin(0.5*r1)
    sr2 = torch.sin(0.5*r2)
    sr3 = torch.sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = torch.vstack([q0,q1,q2,q3])
    # q = q*np.sign(e0)
    
    q = q/torch.linalg.norm(q)
    
    return q.T

def YPRToQuat_np(r1, r2, r3: np.array) -> np.array:
    # For ZYX, Yaw-Pitch-Roll
    # psi   = RPY[0] = r1
    # theta = RPY[1] = r2
    # phi   = RPY[2] = r3
    
    cr1 = cos(0.5*r1)
    cr2 = cos(0.5*r2)
    cr3 = cos(0.5*r3)
    sr1 = sin(0.5*r1)
    sr2 = sin(0.5*r2)
    sr3 = sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = np.array([q0,q1,q2,q3])
    # q = q*np.sign(e0)
    
    q = q/norm(q)
    
    return q

def quat2Dcm_batched(q: torch.Tensor) -> torch.Tensor:

    # expects q first dimension to be batch
    bs = q.shape[0]

    dcm = ptu.create_zeros([bs,3,3])

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

def quat2Dcm_batched_ca(q):

    # expects q first dimension to be batch

    dcm = ca.MX.zeros(3,3)

    dcm[0,0] = q[:,0]**2 + q[:,1]**2 - q[:,2]**2 - q[:,3]**2
    dcm[0,1] = 2.0*(q[:,1]*q[:,2] - q[:,0]*q[:,3])
    dcm[0,2] = 2.0*(q[:,1]*q[:,3] + q[:,0]*q[:,2])
    dcm[1,0] = 2.0*(q[:,1]*q[:,2] + q[:,0]*q[:,3])
    dcm[1,1] = q[:,0]**2 - q[:,1]**2 + q[:,2]**2 - q[:,3]**2
    dcm[1,2] = 2.0*(q[:,2]*q[:,3] - q[:,0]*q[:,1])
    dcm[2,0] = 2.0*(q[:,1]*q[:,3] - q[:,0]*q[:,2])
    dcm[2,1] = 2.0*(q[:,2]*q[:,3] + q[:,0]*q[:,1])
    dcm[2,2] = q[:,0]**2 - q[:,1]**2 - q[:,2]**2 + q[:,3]**2

    return dcm

def quat2Dcm(q: torch.Tensor) -> torch.Tensor: 
    dcm = ptu.from_numpy(np.zeros([3,3]))

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

def quat2Dcm_np(q): 
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

def RotToQuat_batched_ca(R: ca.MX) -> ca.MX:
    
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

def RotToQuat_batched(R: torch.Tensor) -> torch.Tensor:
    
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

    e0 = ptu.create_zeros(tr.shape)
    e1 = ptu.create_zeros(tr.shape)
    e2 = ptu.create_zeros(tr.shape)
    e3 = ptu.create_zeros(tr.shape)

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

def RotToQuat(R: torch.Tensor) -> torch.Tensor:
    
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    # From page 68 of MotionGenesis book
    tr = R11 + R22 + R33

    if tr > R11 and tr > R22 and tr > R33:
        e0 = 0.5 * torch.sqrt(1 + tr)
        r = 0.25 / e0
        e1 = (R32 - R23) * r
        e2 = (R13 - R31) * r
        e3 = (R21 - R12) * r
    elif R11 > R22 and R11 > R33:
        e1 = 0.5 * torch.sqrt(1 - tr + 2*R11)
        r = 0.25 / e1
        e0 = (R32 - R23) * r
        e2 = (R12 + R21) * r
        e3 = (R13 + R31) * r
    elif R22 > R33:
        e2 = 0.5 * torch.sqrt(1 - tr + 2*R22)
        r = 0.25 / e2
        e0 = (R13 - R31) * r
        e1 = (R12 + R21) * r
        e3 = (R23 + R32) * r
    else:
        e3 = 0.5 * torch.sqrt(1 - tr + 2*R33)
        r = 0.25 / e3
        e0 = (R21 - R12) * r
        e1 = (R13 + R31) * r
        e2 = (R23 + R32) * r

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = torch.stack([e0,e1,e2,e3])
    q = q*torch.sign(e0)
    
    q = q/torch.sqrt(torch.sum(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2))
    
    return q


# def RPYtoRot_ZYX(RPY):
    
#     phi = RPY[0]
#     theta = RPY[1]
#     psi = RPY[2]
    
# #    R = np.array([[cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
# #                   cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta), 
# #                   -cos(phi)*sin(theta)],
# #                  [-cos(phi)*sin(psi),
# #                   cos(phi)*cos(psi),
# #                   sin(phi)],
# #                  [cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
# #                   sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
# #                   cos(phi)*cos(theta)]])
    
#     r1 = psi
#     r2 = theta
#     r3 = phi
#     # Rotation ZYX from page 277 of MotionGenesis book
#     R = np.array([[cos(r1)*cos(r2),
#                    -sin(r1)*cos(r3) + sin(r2)*sin(r3)*cos(r1), 
#                    sin(r1)*sin(r3) + sin(r2)*cos(r1)*cos(r3)],
#                   [sin(r1)*cos(r2),
#                    cos(r1)*cos(r3) + sin(r1)*sin(r2)*sin(r3),
#                    -sin(r3)*cos(r1) + sin(r1)*sin(r2)*cos(r3)],
#                   [-sin(r2),
#                    sin(r3)*cos(r2),
#                    cos(r2)*cos(r3)]])
    
#     return R

