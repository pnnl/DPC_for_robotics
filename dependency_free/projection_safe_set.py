import numpy as np
import torch
from scipy.spatial import ConvexHull as SPConvexHull
from scipy.spatial import Delaunay as SPDelaunay

class posVel2cyl:
   
    @staticmethod
    def numpy_vectorized(state, cyl, radius):
        x = state[:, 0:1]
        y = state[:, 2:3]
        xc = cyl[:, 0:1]
        yc = cyl[:, 1:2]

        dx = x - xc
        dy = y - yc

        # Calculate the Euclidean distance from each point to the center of the cylinder
        distance_to_center = (dx**2 + dy**2) ** 0.5
        
        # Subtract the radius to get the distance to the cylinder surface
        distance_to_cylinder = distance_to_center - radius

        xdot = state[:, 1:2]
        ydot = state[:, 3:4]

        # Normalize the direction vector (from the point to the center of the cylinder)
        dx_normalized = dx / (distance_to_center + 1e-10)  # Adding a small number to prevent division by zero
        dy_normalized = dy / (distance_to_center + 1e-10)

        # Compute the dot product of the normalized direction vector with the velocity vector
        velocity_to_cylinder = dx_normalized * xdot + dy_normalized * ydot

        return distance_to_cylinder, velocity_to_cylinder

def preprocess_data(data, num_points_considered, cylinder_radius, cylinder_position):
    # extract all states, inputs, losses from data
    x = torch.stack(data['x'])
    u = torch.stack(data['u'])
    loss = torch.stack(data['loss'])
    c = torch.vstack([torch.tensor([1.,1.])] * x.shape[2])

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
    cylinder_positions = np.vstack([cylinder_position]*cvx_safe_points.shape[0])
    non_cvx_safe_points = np.hstack(posVel2cyl.numpy_vectorized(cvx_safe_points, cylinder_positions, cylinder_radius))

    return cvx_safe_points, non_cvx_safe_points

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
        cvx_safe_points, non_cvx_safe_points = preprocess_data(data, num_points_considered, self.cylinder_radius, self.cylinder_position)
        safe_points = [cvx_safe_points, non_cvx_safe_points]

        print('generating scipy convex hulls...')
        sphulls = [SPConvexHull(p) for p in safe_points]

        self.cvx_hull = sphulls[0]
        self.non_cvx_hull = sphulls[1]

        if generate_delaunay is True:
            print(f'OPTIONAL generating scipy delaunay triangulations...')
            self.cvx_del = SPDelaunay(cvx_safe_points[self.cvx_hull.vertices])

    def nearest_face(self, point):

        face = self.cvx_hull.points[self.cvx_hull.simplices[0]]
        normal = self.cvx_hull.equations[0][:-1]
        offset = self.cvx_hull.equations[0][-1]

        def project_point_onto_plane(point, normal, offset):
            # Calculate the dot product of the point and the normal vector
            dot_product = np.dot(point, normal)
            # Calculate the projection of the point onto the plane
            projection = point - (dot_product + offset) / np.dot(normal, normal) * normal
            return projection

        test = project_point_onto_plane(point, normal, offset)
        assert normal @ test + offset == 0

        
if __name__ == "__main__":
    ss = SafeSet(torch.load('large_data/nav_training_data.pt'), generate_delaunay=False)
    point = np.array([-1.,0,1,0,0,0])
    ss.nearest_face(point)

