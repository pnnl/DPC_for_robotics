# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

"""First pass on pytorch conversion complete"""
import casadi as ca
import torch
from torch.linalg import norm

# Normalize quaternion, or any vector
def vectNormalize(q: torch.Tensor) -> torch.Tensor:
    return q/norm(q)

def vectNormalize_batched(q: torch.Tensor, norm_dim=1) -> torch.Tensor:
    normalised_vecs = norm(q, dim=norm_dim).unsqueeze(1)
    return q/normalised_vecs

def vectNormalize_batched_ca(q: ca.MX, norm_dim=1) -> ca.MX:
    # Compute the norms for each row
    mags = [ca.norm_2(q[i, :]) for i in range(q.shape[0])]
    mags = ca.vertcat(*mags)
    
    # Reshape the magnitude array to be compatible for division
    normalised_vecs = ca.reshape(mags, -1, 1)
    
    # Divide the original matrix by the reshaped magnitude matrix
    return q / normalised_vecs

# Quaternion multiplication
def quatMultiply(q, p: torch.Tensor) -> torch.Tensor:

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

# Quaternion multiplication
def quatMultiply_batched(q, p: torch.Tensor) -> torch.Tensor:

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

def quatMultiply_batched_ca(q, p: ca.MX) -> ca.MX:

    row0 = ca.horzcat(q[:,0], -q[:,1], -q[:,2], -q[:,3])
    row1 = ca.horzcat(q[:,1],  q[:,0], -q[:,3],  q[:,2])
    row2 = ca.horzcat(q[:,2],  q[:,3],  q[:,0], -q[:,1])
    row3 = ca.horzcat(q[:,3], -q[:,2],  q[:,1],  q[:,0])

    Q = ca.vertcat(row0, row1, row2, row3)

    # In CasADi, we perform matrix multiplication using mtimes
    mult = ca.mtimes(Q, p.T).T

    return mult

# Inverse quaternion
def inverse(q: torch.Tensor) -> torch.Tensor:
    qinv = torch.stack([q[0], -q[1], -q[2], -q[3]])/norm(q)
    return qinv

def inverse_batched(q: torch.Tensor) -> torch.Tensor:
    qinv = torch.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]])/norm(q, axis=1)
    return qinv.T

# def inverse_batched_ca(q: ca.MX) -> ca.MX:
#     qnorm = ca.sqrt(ca.mtimes(q, q.T).diagonal())
#     qinv = ca.horzcat([q[:,0], -q[:,1], -q[:,2], -q[:,3]]) / ca.mtimes(qnorm, ca.MX.ones(1,4))
#     return qinv

def inverse_batched_ca(q: ca.MX) -> ca.MX:
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