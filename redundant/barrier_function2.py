import torch
import numpy as np
import utils.pytorch as ptu
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d, distance
from dpc import posVel2cyl
import matplotlib.pyplot as plt
import time
from utils.time import time_function
np.random.seed(0)

class BarrierFunction:
    def __init__(self, data) -> None:
        
        self.cylinder_radius = 0.5
        self.cylinder_position = np.array([[1.,1.]])
        self.cylinder_margin = 0.15
        num_points_considered = 100_000

        print('---------------------------------------')
        print('GENERATING SAFE SET, TAKES ~ 10 SECONDS')
        print('---------------------------------------')

        print('preprocessing data...')
        cvx_safe_points, non_cvx_safe_points = self.preprocess_data(data, num_points_considered)
        safe_points = [cvx_safe_points, non_cvx_safe_points]

        print('generating convex hulls...')
        self.hulls = [ConvexHull(p) for p in safe_points]

        print('finding the centroids of every simplex of every hull...')
        self.simplex_centroids = []
        for hull in self.hulls:
            self.simplex_centroids.append(np.mean(hull.points[hull.simplices], axis=1))

        print('initialising starting directed simplex search index...')
        self.start_simplex_idx = [0, 0]

        dummy_point = np.zeros([6])
        pv_dummy_point = np.hstack(posVel2cyl.numpy_vectorized(dummy_point[None,:], self.cylinder_position, self.cylinder_radius))

        # test find nearest simplex to point
        self.find_nearest_simplex(pv_dummy_point, self.hulls[1], self.simplex_centroids[1], search_step_depth=2)

        print('fin')

    def find_nearest_simplex(self, point, hull, centroids, search_step_depth):
        """
        point: the point being tested
        hull: the convex hull of the safe set being tested
        centroids: the centroids of the faces of the hull being tested
        search_step_depth: how many neighbours degrees we search per step
        """
        start_simplex_idx = np.array([0])
        neighbors, visited = self.find_new_neighbors(hull, start_simplex_idx, search_step_depth)
        
        # Calculate the Euclidean distance from the point x to each point in the convex hull
        search_space = np.hstack([start_simplex_idx, neighbors.flatten()])
        distances = np.linalg.norm(centroids[search_space] - point, axis=1)

        closest_simplex_idx = np.argmin(distances)
        if closest_simplex_idx == 0: # the starting simplex - we have found solution
            return search_space[closest_simplex_idx], True
        else:
            return search_space[closest_simplex_idx], False


    def find_new_neighbors(self, hull, simplex_idx, search_step_depth, previously_visited=None):
        # Get all neighbors until search_depth is satisfied
        ordered_neighbors = []
        if previously_visited is None:
            previously_visited = set(simplex_idx)
        for _ in range(search_step_depth):
            simplex_idx = hull.neighbors[simplex_idx]
            simplex_idx = np.setdiff1d(simplex_idx, list(previously_visited)) # filter out previously visited
            ordered_neighbors.append(simplex_idx)
            previously_visited.update(simplex_idx)
        neighbors = np.vstack(ordered_neighbors)
        return neighbors, previously_visited

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
    bf = BarrierFunction(data)
    print('fin')