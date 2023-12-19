"""
This script implements the CBF formulation algorithm:

Assumptions:
- All positions that satisfy the known CLF are valid for agents with lower cost
- All points between positions that satisfy the known CLF are valid

Method:
- Retrieve the training rollouts of DPC training
- 
"""

import torch
import numpy as np
import utils.pytorch as ptu
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d
from dpc import posVel2cyl
import matplotlib.pyplot as plt

def generate_cbf(history):
    # 0.18 too little
    # 0.20 too much, or maybe good to be conservative i'm unsure
    cylinder_margin = 0.2 # 0.19
    cylinder_radius = 0.5

    print('analysing training dataset for points that satisfy constraints and CLF...')
    x = torch.stack(history['x'])
    u = torch.stack(history['u'])
    loss = torch.stack(history['loss'])
    c = torch.vstack([ptu.tensor([1.,1.])] * x.shape[2])

    log = {'success': [], 'failure': [], 'success_inputs': [], 'failure_inputs': [], 'minibatch_loss': [], 'minibatch_number': []}
    for i, (minibatch, minibatch_inputs) in enumerate(zip(x, u)):
        for trajectory, inputs in zip(minibatch, minibatch_inputs):
            # state = {x, xdot, y, ydot, z, zdot}
            xy = trajectory[:,0:4:2]

            # we must first rule out all trajectories that under DPC control intersected
            # the cylinder constraint
            distances = torch.norm(xy - c, dim=1)
            if (distances < cylinder_radius).any() == True:
                # trajectory is tossed
                continue

            # next we must check the lie derivatives of all points on the trajectory
            # we know from euler integrator we can just find dynamics dot from subtracting points
            # if this wasnt the case we could record the actual dynamics as the DPC trains
            # Del V.T @ f(x,u) < 0
            f = trajectory[1:,:] - trajectory[:-1,:]
            del_V = trajectory[:-1,:]
            L = torch.einsum('ij,ij->i', del_V, f) # einsum is AMAZING

            # add the successes and failures to a history to be returned
            true_indices = torch.nonzero(L <= 0.0, as_tuple=True)[0]
            false_indices = torch.nonzero(L > 0.0, as_tuple=True)[0]

            log['success'].append(trajectory[true_indices])
            log['failure'].append(trajectory[false_indices])
            log['success_inputs'].append(inputs[true_indices])
            log['failure_inputs'].append(inputs[false_indices])
            log['minibatch_loss'].append(loss[i])
            log['minibatch_number'].append(i)

    successes = torch.vstack(log['success']).numpy()
    # failures = torch.vstack(log['failure']).numpy()
    # success_inputs = torch.vstack(log['success_inputs']).numpy()

    # the adversarial condition, no point in dataset managed to avoid cylinder from this state
    point_to_check = np.array([0,1.5,0,1.5,0,0]) 

    # Construct the Cylinder Constrait Convex Hull
    # --------------------------------------------
    c = np.array([[1.,1.]]*successes.shape[0])
    pv = np.hstack(posVel2cyl(successes, c, cylinder_radius))
    
    # we only care about points moving towards the cylinder
    pv_pos = pv[pv[:, 1] >= 0]

    # we don't want the max vel towards the cylinder to be too high for robustness
    pv_pos_lim = pv_pos # pv_pos[pv_pos[:, 1] <= 4]


    cyl_hull = ConvexHull(pv_pos_lim)
    cyl_delaunay = Delaunay(pv_pos_lim[cyl_hull.vertices])

    # _ = delaunay_plot_2d(delaunay)
    # plt.savefig('data/paper/cylinder_constraint_cvx_hull.svg')

    def cyl_cbf(x):
        pv = np.hstack(posVel2cyl(x[None,:], c[:1,:], cylinder_radius)).flatten()
        # shift position back with the safety margin
        pv[0] += cylinder_margin
        return cyl_delaunay.find_simplex(pv)
    
    cyl_cbf(point_to_check)

    # Construct Box Constraint Convex Hull
    # ------------------------------------
    print('forming the convex hull of successful points during training... (this can take time)')
    hull = ConvexHull(successes)

    print('performing delaunay triangulation of the convex hull... (this can take time)')
    delaunay = Delaunay(successes[hull.vertices])



    box_cbf = lambda x: delaunay.find_simplex(x)

    box_cbf(point_to_check)

    # interesection of the two safe sets form the CBF
    def cbf(x):
        if box_cbf(x) >= 0 and cyl_cbf(x) >= 0:
            return True
        else:
            return False

    print('running the first point check... (this can take time)')
    is_inside = cbf(point_to_check)

    print(is_inside)

    print('fin')


if __name__ == "__main__":

    log = torch.load('large_data/nav_training_data.pt')
    generate_cbf(log)