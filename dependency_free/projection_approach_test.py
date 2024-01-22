import casadi as ca
import numpy as np
import torch
from scipy.spatial import ConvexHull
import time

def preprocess_data(data, num_points_considered, cylinder_radius):
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

    return cvx_safe_points

def project_point_onto_plane(point, normal, offset):
    dot_product = np.dot(point, normal)
    projection = point - (dot_product + offset) / np.dot(normal, normal) * normal
    return projection

def project_point_onto_planes(point, normals, offsets):
    """
    Project a point onto multiple planes.

    Parameters:
    point (np.array): The point to be projected, shape (3,).
    normals (np.array): Array of normal vectors of the planes, shape (n, 3).
    offsets (np.array): Array of offsets (distance from origin to the plane along the normal), shape (n,).

    Returns:
    np.array: Array of projected points, shape (n, 3).
    """
    # Calculate dot products between point and each normal
    dot_products = np.dot(normals, point)  # shape (n,)

    # Calculate projections for each plane
    projections = point - ((dot_products + offsets)[:, np.newaxis] / np.sum(normals * normals, axis=1)[:, np.newaxis]) * normals
    
    return projections


def get_neighbors_and_origin(hull, simplex):
    return np.insert(hull.neighbors[simplex], 0, simplex)


def find_normal(vectors):
    """
    Find a vector that is orthogonal to all given vectors.
    """
    dim = vectors.shape[1]
    A = vectors
    b = np.zeros(dim)
    x = np.linalg.lstsq(A, b, rcond=None)[0] # no cross product beyond 3D
    return x / np.linalg.norm(x)

def check_centroid_in_simplex(vertices):
    num_vertices = vertices.shape[1]
    dim = vertices.shape[2]
    centroid = np.mean(vertices, axis=1)
    
    A = np.zeros((num_vertices, dim))
    b = np.zeros(num_vertices)
    
    for i in range(num_vertices):
        v1 = vertices[0, i]
        v2 = vertices[0, (i + 1) % num_vertices]

        # Form vectors that define the face
        face_vectors = np.array([v2 - v1])  # Add more vectors if needed
        
        # Find a normal to the face
        normal = find_normal(face_vectors)
        
        A[i] = normal
        b[i] = np.dot(normal, v1)
    
    return np.all(np.matmul(A, centroid.T) <= b)

def get_edges(hull, search_set):
    points = hull.points[hull.simplices[search_set]]
    origin = points[:1]
    neighbors = points[1:]

    # I will first test if points are cyclic on the origin simplex
    print(check_centroid_in_simplex(origin))
    print('fin')

class SafeSet:
    def __init__(self, data) -> None:
        self.cylinder_radius = 0.5
        self.cylinder_margin = 0.15
        num_points_considered = 100_000
        self.simplex_guess = 1 # for the initial guess before warm starting of the directed search

        print(f"----------------------------------------------------------------------------------")
        print(f"generating safe set from DPC training dataset using {num_points_considered} points")
        print(f"----------------------------------------------------------------------------------")

        print('preprocessing data...')
        safe_points = preprocess_data(data, num_points_considered, self.cylinder_radius)

        print("generating hull")
        self.hull = ConvexHull(safe_points)

    def __call__(self, point):
        search_set = get_neighbors_and_origin(self.hull, self.simplex_guess) 
        equations = self.hull.equations[search_set]
        projections = project_point_onto_planes(point, equations[:,:-1], equations[:,-1])

        # now we have the projections onto the hull at a simplex and its edge neighbors
        # first we must test if the projection is within any of the faces
        
        # first we need the edges of the simplex:
        edges = get_edges(self.hull, search_set)


        print('fin')
        


if __name__ == "__main__":
   
    point = np.array([-1.,0,1,0,0,0])

    # validated
    dummy_plane_equation = np.ones([7])
    project_point_onto_plane(point, dummy_plane_equation[:-1], dummy_plane_equation[-1])

    # validated
    dummy_vec_plane_equation = np.ones([2,7])
    project_point_onto_planes(point, dummy_vec_plane_equation[:,:-1], dummy_vec_plane_equation[:,-1])

    ss = SafeSet(torch.load('large_data/nav_training_data.pt'))

    test = ss(point)

    # ss.nearest_face(point)
    print('fin')
