# dependecy free testbed for directed search algorithm
import numpy as np
import torch
import ctypes
import subprocess
import time
from tqdm import tqdm
from scipy.spatial import ConvexHull as SPConvexHull
from scipy.spatial import Delaunay as SPDelaunay

np.random.seed(0)

# =============================== #
# ======== Utility Funcs ======== #
# =============================== #

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

def find_vertex_neighbors(simplices, simplex):    # find all simplices that share a vertex with ds_vertices
    directed_search_vertices = simplices[simplex]
    shared_v_simplices_idx = []
    for idx, vertices in tqdm(enumerate(simplices)):
        common_elements = np.intersect1d(vertices, directed_search_vertices)

        # Check if there are any common elements
        has_common_elements = len(common_elements) > 0
        if has_common_elements:
            shared_v_simplices_idx.append(idx)
    return shared_v_simplices_idx

def find_distance_to_point(points, simplices, point, search_simplex):
    centroid = np.mean(points[simplices[search_simplex]], axis=1)
    distance = np.linalg.norm(centroid - point, axis=1)
    return distance

def find_min_distance_to_point(points, simplices, point, search_simplices):
    distances = find_distance_to_point(points, simplices, point, search_simplices)
    return distances.min()

# =============================== #
# ======= Directed Search ======= #
# =============================== #
    
def nth_degree_centroidal_search(candidate, centroids, neighbors, vertices, point, depth):

    candidate_vertices = set(vertices[candidate])
    visited = set()  # Set to keep track of visited neighbors
    visited.add(candidate)
    result = set()

    def search_neighbors(current, depth_remaining, is_first_level=True):
        if depth_remaining == 0:
            return current
    
        if is_first_level and depth_remaining == 1:
            result.update(list(neighbors[current]))

        # Include current node's neighbors
        for neighbor in neighbors[current]:

            # Skip neighbors that don't share a vertex with the candidate (except for the first level)
            if not is_first_level and (neighbor in visited or not candidate_vertices.intersection(vertices[neighbor])):
                continue

            visited.add(neighbor)
            result.add(neighbor)

            if depth_remaining > 1:
                search_neighbors(neighbor, depth_remaining - 1, False)

    # Calculate the distance from the candidate centroid to the point
    d_ctrl = np.linalg.norm(centroids[candidate] - point)

    # Recursively search neighbors up to Nth degree
    search_neighbors(candidate, depth) # -> result
    nN = list(result)
    # if depth > 1:
    #     print('fin')

    # Process the results
    if len(nN) == 0:
        return candidate
    nN = np.unique(np.hstack(nN))
    
    mask = np.isin(nN, candidate)
    nNm = nN[~mask]
    c = centroids[nNm]
    d = np.linalg.norm(c - point, axis=1)

    # Find the minimum distance and decide
    i = np.argmin(d)
    return candidate if d_ctrl <= d[i] else nNm[i]

def find_angle(candidate, centroids, equations, point):
    direction = point.flatten() - centroids[candidate]
    normal = equations[candidate][:-1]
    cosine_angle = np.dot(normal, direction) / (np.linalg.norm(normal) * np.linalg.norm(direction)) # normalised
    angle = np.arccos(cosine_angle)
    return angle # in radians    

def find_angles(candidate, centroids, equations, point):
    direction = point - centroids[candidate]
    normal = equations[candidate][:,:-1]
    cosine_angle = np.einsum('ij,ij->i', normal, direction)
    normal_norms = np.linalg.norm(normal, axis=1)
    direction_norms = np.linalg.norm(direction, axis=1)
    cosine_angle_norm = cosine_angle / (normal_norms * direction_norms)
    angle = np.arccos(cosine_angle_norm)
    return angle # in radians

def nth_degree_curvature_search(candidate, centroids, equations, neighbors, vertices, point, depth):

    candidate_vertices = set(vertices[candidate])
    visited = set()  # Set to keep track of visited neighbors
    visited.add(candidate)
    result = set()
    result.add(candidate)

    def search_neighbors(current, depth_remaining, is_first_level=True):
        if depth_remaining == 0:
            return current
    
        if is_first_level and depth_remaining == 1:
            result.update(list(neighbors[current]))

        # Include current node's neighbors
        for neighbor in neighbors[current]:

            # Skip neighbors that don't share a vertex with the candidate (except for the first level)
            if not is_first_level and (neighbor in visited or not candidate_vertices.intersection(vertices[neighbor])):
                continue

            visited.add(neighbor)
            result.add(neighbor)

            if depth_remaining > 1:
                search_neighbors(neighbor, depth_remaining - 1, False)
    
    # Recursively search neighbors up to Nth degree
    search_neighbors(candidate, depth) # -> result
    nN = list(result)

    # Process the results
    if len(nN) == 0:
        return candidate
    nN = np.hstack(nN)
    
    a = find_angles(nN, centroids, equations, point)

    # Find the minimum distance and decide
    i = np.argmax(a)
    # print(f"largest angle found: {a[i]}")
    return nN[i]


def directed_search(simplex_idx, centroids, equations, neighbors, vertices, point, depth):
    current_candidate = simplex_idx

    # do the curvature directed search
    # nth_degree_curvature_search(current_candidate, centroids, equations, neighbors, vertices, point, depth)

    # do the curvature directed search
    found=False
    while not found:
        prior_candidate = current_candidate
        current_candidate = nth_degree_curvature_search(current_candidate, centroids, equations, neighbors, vertices, point, depth=1)
        if current_candidate == prior_candidate:
            current_candidate = nth_degree_curvature_search(current_candidate, centroids, equations, neighbors, vertices, point, depth=depth)
            if current_candidate == prior_candidate:
                print(f"curvature found simplex: {current_candidate}")
                found = True

    
    # do the centroidal directed search
    found=False
    while not found:
        prior_candidate = current_candidate
        current_candidate = nth_degree_centroidal_search(current_candidate, centroids, neighbors, vertices, point, depth=1)
        if current_candidate == prior_candidate:
            current_candidate = nth_degree_centroidal_search(current_candidate, centroids, neighbors, vertices, point, depth=depth)
            if current_candidate == prior_candidate:
                print(f"centroidal found simplex: {current_candidate}")
                return current_candidate
            


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

        self.cvx_hull = sphulls[0]
        self.non_cvx_hull = sphulls[1]

        if generate_delaunay is True:
            print(f'OPTIONAL generating scipy delaunay triangulations...')
            self.cvx_del = SPDelaunay(cvx_safe_points[self.cvx_hull.vertices])

    # def nearest_simplex(self, point, depth=6):
    #     simplex_idx = 1
    #     centroids = np.mean(self.cvx_hull.points[self.cvx_hull.simplices], axis=1)   
    #     distances = np.linalg.norm(centroids - point, axis=1)
    #     closest_simplex = directed_search(simplex_idx, centroids, self.cvx_hull.neighbors, self.cvx_hull.simplices, point, depth=depth)
    #     print(f"directed search nearest simplex: {closest_simplex}")
    #     print(f"directed search min distance: {distances[closest_simplex]}")
    #     equation = self.cvx_hull.equations[closest_simplex]
    #     return equation
    # 
    # def nearest_simplex_control(self, point):
    #     centroids = np.mean(self.cvx_hull.points[self.cvx_hull.simplices], axis=1)   
    #     distances = np.linalg.norm(centroids - point, axis=1)
    #     closest_simplex = np.argmin(distances)
    #     print(f"true nearest simplex: {closest_simplex}")
    #     print(f"true min distance: {distances[closest_simplex]}")
    #     equation = self.cvx_hull.equations[closest_simplex]
    #     return equation

    def is_in_cvx_hull_control(self, point):
        centroids = np.mean(self.cvx_hull.points[self.cvx_hull.simplices], axis=1)   
        distances = np.linalg.norm(centroids - point, axis=1)
        closest_simplex = np.argmin(distances)
        print(f"Python nearest simplex: {closest_simplex}")
        print(f"Python min distance: {distances[closest_simplex]}")
        equation = self.cvx_hull.equations[closest_simplex]
        print(f"In cvx hull: {equation[:-1] @ point.T + equation[-1] <= 0}")
        cvx_hyperplane_distances = self.cvx_del.plane_distance(point)
        satisfy_delaunay = cvx_hyperplane_distances.max() >= 0 and cvx_hyperplane_distances.min() <= 0 # .find_simplex(x) >= 0 
        print(f"In Delaunay hull: {satisfy_delaunay}")
        return equation
            
            
    # def is_in_non_cvx_hull(self, point):
    #     x = np.hstack(posVel2cyl.numpy_vectorized(point, self.cylinder_position, self.cylinder_radius)).flatten()
    #     distances = np.linalg.norm(self.non_cvx_hull.points - x, axis=1)
    #     closest_point_index = np.argmin(distances)
    #     closest_point = self.non_cvx_hull.points[closest_point_index]
    #     simplex_centroids = np.mean(self.non_cvx_hull.points[self.non_cvx_hull.simplices], axis=1)   
    #     distances_to_closest_point = np.linalg.norm(simplex_centroids - closest_point, axis=1)
    #     closest_simplex_index = np.argmin(distances_to_closest_point)
    #     # Form the tangential hyperplane
    #     normal_vector = self.non_cvx_hull.equations[closest_simplex_index][:self.non_cvx_hull.ndim]  # Normal part of the equation
    #     equation = self.non_cvx_hull.equations[closest_simplex_index]
    #     normal_vector = equation[:-1]  # A, B, C, D, E, F
    #     constant_term = equation[-1]  # G
    #     # result = np.dot(normal_vector, x) + constant_term # return result <= 0  # Inside or on the simplex if true
    #     return equation

    def is_in_non_cvx_hull(self, point):
        point = np.hstack(posVel2cyl.numpy_vectorized(point, self.cylinder_position, self.cylinder_radius)).flatten()
        point[0] -= self.cylinder_margin # add margin term to shift safe set
        centroids = np.mean(self.non_cvx_hull.points[self.non_cvx_hull.simplices], axis=1)
        distances = np.linalg.norm(centroids - point, axis=1)
        closest_simplex = np.argmin(distances)
        equation = self.non_cvx_hull.equations[closest_simplex]
        return equation


    def preprocess_data(self, data, num_points_considered):

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
    ss = SafeSet(data, generate_delaunay=False)
    from tqdm import tqdm

    # point = np.array([[-1., 0, 1.0, 0, 0, 0]]) fail
    # point = np.array([[3., 0, -1., 0, 0, 0]]) pass
    # point = np.array([[-2., 0, 2., 0, 0, 0]]) pass
    point = np.array([[-0.9, 0, 0.9, 0, 0, 0]])
    eqn = ss.is_in_cvx_hull_control(point)
    point_in_cvx_hull = eqn[:-1] @ point.T + eqn[-1] <= 0
    print(f"point in cvx hull: {point_in_cvx_hull}")
    assert point_in_cvx_hull == False

    random_guesses = np.random.randint(0, ss.cvx_hull.simplices.shape[0], 100)    
    for simplex_guess in random_guesses:
        hull = ss.cvx_hull
        points = hull.points
        simplices = hull.simplices
        neighbors = hull.neighbors
        centroids = np.mean(points[simplices], axis=1)   
        equations = hull.equations
        distances = np.linalg.norm(centroids - point, axis=1)

        directed_search_simplex = directed_search(simplex_guess, centroids, equations, neighbors, simplices, point, depth=15)
        directed_search_distance = find_distance_to_point(points, simplices, point, directed_search_simplex)

        print(f"directed search nearest simplex: {directed_search_simplex}")
        print(f"directed search min distance: {distances[directed_search_simplex]}")

        if distances[directed_search_simplex] == distances.min():
            continue

        directed_search_vertex_neighbors = find_vertex_neighbors(simplices, directed_search_simplex)
        directed_search_vertex_neighbors_min_distance = find_min_distance_to_point(points, simplices, point, directed_search_vertex_neighbors)

        print(f"shared vertex min distance: {directed_search_vertex_neighbors_min_distance}")
        print(f"true min distance: {distances.min()}")

        if directed_search_vertex_neighbors_min_distance != distances.min():
            print("DISCREPANCY DETECTEDDDDDDDD!!!!!")

    # centroids = np.mean(ss.cvx_hull.points[candidate], axis=1)   
    # distances = np.linalg.norm(centroids - point, axis=1)
    # print(f"directed search nearest simplex: {candidate}")

    print('fin')

    
