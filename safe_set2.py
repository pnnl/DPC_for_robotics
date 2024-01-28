"""
"""

import numpy as np
import torch
import utils.pytorch as ptu
import subprocess
import time
from tqdm import tqdm
from dpc import posVel2cyl
from scipy.spatial import ConvexHull as SPConvexHull
from scipy.spatial import Delaunay as SPDelaunay

np.random.seed(0)

# -----------------------------------------------------------------------
class SafeSet:
    def __init__(self, data, generate_delaunay=False) -> None:
        self.cylinder_radius = 0.5
        self.cylinder_position = np.array([[1.,1.]])
        self.cylinder_margin = 0.15
        num_points_considered = 100_000
        self.simplex_guess = 1 # for the initial guess before warm starting of the directed search

        print(f"----------------------------------------------------------------------------------")
        print(f"generating safe set from DPC training dataset using {num_points_considered} points")
        print(f"----------------------------------------------------------------------------------")

        print('preprocessing data...')
        cvx_safe_points, non_cvx_safe_points = self.preprocess_data(data, num_points_considered)
        safe_points = [cvx_safe_points, non_cvx_safe_points]

        print('generating scipy convex hulls...')
        sphulls = [SPConvexHull(p) for p in safe_points]

        print('getting handles to the C hulls...')
        # hulls = [self._init_hull(sphull) for sphull in sphulls]
        self.cvx_hull = sphulls[0]
        self.non_cvx_hull = sphulls[1]

        if generate_delaunay is True:
            print(f'OPTIONAL generating scipy delaunay triangulations...')
            self.cvx_del = SPDelaunay(cvx_safe_points[self.cvx_hull.vertices])

        print('running test...')
        point = np.array([[2.,0.,2.,0.,1.,0.]])
        tic = time.time()
        equation_cvx = self.is_in_cvx_hull_control(point)
        toc = time.time()
        print(f"time taken in test to find cvx simplex: {toc-tic}")

        tic = time.time()
        equation_noncvx = self.is_in_non_cvx_hull(point)
        toc = time.time()
        print(f"time taken in test to find non cvx simplex: {toc-tic}")

        print("__init__ complete.")
    
    def is_in_cvx_hull_control(self, point):
        centroids = np.mean(self.cvx_hull.points[self.cvx_hull.simplices], axis=1)   
        distances = np.linalg.norm(centroids - point, axis=1)
        closest_simplex = np.argmin(distances)
        print(f"Python nearest simplex: {closest_simplex}")
        print(f"Python min distance: {distances[closest_simplex]}")
        equation = self.cvx_hull.equations[closest_simplex]
        return equation
            
    # applying the C to this one will take some rewriting
    def is_in_non_cvx_hull(self, point):
        x = np.hstack(posVel2cyl.numpy_vectorized(point, self.cylinder_position, self.cylinder_radius)).flatten()
        distances = np.linalg.norm(self.non_cvx_hull.points - x, axis=1)
        closest_point_index = np.argmin(distances)
        closest_point = self.non_cvx_hull.points[closest_point_index]
        simplex_centroids = np.mean(self.non_cvx_hull.points[self.non_cvx_hull.simplices], axis=1)   
        distances_to_closest_point = np.linalg.norm(simplex_centroids - closest_point, axis=1)
        closest_simplex_index = np.argmin(distances_to_closest_point)
        # Form the tangential hyperplane
        normal_vector = self.non_cvx_hull.equations[closest_simplex_index][:self.non_cvx_hull.ndim]  # Normal part of the equation
        equation = self.non_cvx_hull.equations[closest_simplex_index]
        normal_vector = equation[:-1]  # A, B, C, D, E, F
        constant_term = equation[-1]  # G
        # result = np.dot(normal_vector, x) + constant_term # return result <= 0  # Inside or on the simplex if true
        return equation
    
    def preprocess_data(self, data, num_points_considered):

        # extract all states, inputs, losses from data
        x = torch.stack(data['x'])
        u = torch.stack(data['u'])
        loss = torch.stack(data['loss'])
        c = torch.vstack([ptu.tensor([1.,1.])] * x.shape[2])

        log = {'success': [], 'failure': [], 'success_inputs': [], 'failure_inputs': [], 'minibatch_loss': [], 'minibatch_number': []}
        for i, (minibatch, minibatch_inputs) in enumerate(zip(x, u)):
            for trajectory, inputs in zip(minibatch, minibatch_inputs):
                # state = {x, xdot, y, ydot, z, zdot}
                xy = trajectory[:,0:4:2]

                # we must first rule out all trajectories that under DPC control intersected
                # the cylinder constraint
                distances = torch.norm(xy - c, dim=1)
                if (distances < self.cylinder_radius).any() == True:
                    # trajectory is tossed
                    continue

                # next we check the lie derivatives of all points on the trajectory
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

        all_safe_points = torch.vstack(log['success']).numpy()

        # Ensure you don't try to select more points than exist
        num_points_considered = min(num_points_considered, all_safe_points.shape[0])

        # Generate random unique indices
        random_indices = np.random.choice(all_safe_points.shape[0], size=num_points_considered, replace=False)

        # Select the points using the random indices
        cvx_safe_points = all_safe_points[random_indices, :]

        # transform these safe points into non-cvx space
        cylinder_positions = np.vstack([self.cylinder_position]*cvx_safe_points.shape[0])
        non_cvx_safe_points = np.hstack(posVel2cyl.numpy_vectorized(cvx_safe_points, cylinder_positions, self.cylinder_radius))

        return cvx_safe_points, non_cvx_safe_points
    
if __name__ == "__main__":

    data = torch.load('large_data/nav_training_data.pt')
    ss = SafeSet(data, generate_delaunay=True)

    point = np.array([[-1.00000000e+00, -0.00000000e+00,  1.00000000e+00,  1.25224449e-22,
       -1.30335494e-13, -2.60670985e-10]])
    
    simplex_idx = 1
    centroids = np.mean(ss.cvx_hull.points[ss.cvx_hull.simplices], axis=1)   
    distances = np.linalg.norm(centroids - point, axis=1)
    neighbors = ss.cvx_hull.neighbors

    # lets use the delaunay
    centroids = np.mean(ss.cvx_del.points[ss.cvx_del.simplices], axis=1)   
    distances = np.linalg.norm(centroids - point, axis=1)
    neighbors = ss.cvx_del.neighbors

    bf = python_brute_force(centroids, point)
    ds = python_directed_search2(simplex_idx, centroids, neighbors, point, depth=8)

    print(f"brute force distance: {distances[bf]}")
    print(f"directed search distance: {distances[ds]}")
    print('fin')