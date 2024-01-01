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
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d, distance
from dpc import posVel2cyl
import matplotlib.pyplot as plt
import time
from utils.time import time_function
from sklearn.cluster import KMeans
np.random.seed(0)

class BarrierFunction:
    def __init__(self, data) -> None:

        self.cylinder_radius = 0.5
        self.cylinder_position = np.array([[1.,1.]])
        self.cylinder_margin = 0.15
        num_points_considered = 100_000

        print('-----------------------------------')
        print('GENERATING SAFE SET, TAKES ~ 2 MINS')
        print('-----------------------------------')

        print('preprocessing data...')
        processed_data = self.preprocess_data(data, num_points_considered)

        print('constructing non-cvx constraint safe set...')
        self.non_cvx_delaunay, self.non_cvx_hull = self.construct_non_cvx_constraint_safe_set(processed_data)

        print('constructing cvx constraint safe set (can take a while)...')
        self.cvx_delaunay, self.cvx_hull = self.construct_cvx_constraint_safe_set(processed_data)

        print('evaluating combined safe sets for first time... (can take time - scipy cython functions compiling)')
        _ = self.is_in_safe_set(np.array([0.,0.,0.,0.,0.,0.]))

        print('evaluating for a second time - should be fast now...')
        _ = self.is_in_safe_set(np.array([0.,0.,0.,0.,0.,0.]))

    def __call__(self, x):
        return self.is_in_safe_set(x)
    
    def find_tangential_hyperplane_on_pv_hull_nearest_point(self, x):
        x = np.array(x).flatten()
        # Calculate the Euclidean distance from the point x to each point in the convex hull
        distances = np.linalg.norm(self.non_cvx_hull.points - x, axis=1)

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Extract the closest point
        closest_point = self.non_cvx_hull.points[closest_point_index]

        # Calculate the centroids of each simplex
        simplex_centroids = np.mean(self.non_cvx_hull.points[self.non_cvx_hull.simplices], axis=1)   

        # Compute distances from each centroid to the closest point
        distances_to_closest_point = np.linalg.norm(simplex_centroids - closest_point, axis=1)

        # Find the index of the closest simplex
        closest_simplex_index = np.argmin(distances_to_closest_point)

        # Form the tangential hyperplane
        # closest_simplex = self.cvx_hull.simplices[closest_simplex_index]
        normal_vector = self.non_cvx_hull.equations[closest_simplex_index][:self.non_cvx_hull.ndim]  # Normal part of the equation
        offset = self.non_cvx_hull.equations[closest_simplex_index][-1]  # Offset part of the equation

        # The distance to the closest simplex (the minimum distance from the closest point to simplex centroids)
        distance_to_closest_simplex = distances_to_closest_point[closest_simplex_index]

        print(f'distance to nearest simplex: {distance_to_closest_simplex}')
        print(f'is in nearest halfspace safe set: {normal_vector.T @ x + offset <= 0}')

        return normal_vector, offset

    def find_tangential_hyperplane_on_cvx_hull_nearest_point(self, x):
        x = np.array(x).flatten()
        # Calculate the Euclidean distance from the point x to each point in the convex hull
        distances = np.linalg.norm(self.cvx_hull.points - x, axis=1)

        # Find the index of the closest point
        closest_point_index = np.argmin(distances)

        # Extract the closest point
        closest_point = self.cvx_hull.points[closest_point_index]

        # Calculate the centroids of each simplex
        simplex_centroids = np.mean(self.cvx_hull.points[self.cvx_hull.simplices], axis=1)

        # Compute distances from each centroid to the closest point
        distances_to_closest_point = np.linalg.norm(simplex_centroids - closest_point, axis=1)

        # Find the index of the closest simplex
        closest_simplex_index = np.argmin(distances_to_closest_point)

        # Form the tangential hyperplane
        # closest_simplex = self.cvx_hull.simplices[closest_simplex_index]
        normal_vector = self.cvx_hull.equations[closest_simplex_index][:self.cvx_hull.ndim]  # Normal part of the equation
        offset = self.cvx_hull.equations[closest_simplex_index][-1]  # Offset part of the equation

        # The distance to the closest simplex (the minimum distance from the closest point to simplex centroids)
        distance_to_closest_simplex = distances_to_closest_point[closest_simplex_index]

        print(f'distance to nearest simplex: {distance_to_closest_simplex}')
        print(f'is in nearest halfspace safe set: {normal_vector.T @ x + offset <= 0}')

        return normal_vector, offset

    # @time_function
    def is_in_safe_set(self, x):
        # x: {x, xdot, y, ydot, z, zdot}
        cvx_hyperplane_distances = self.cvx_delaunay.plane_distance(x)
        satisfy_box = cvx_hyperplane_distances.max() >= 0 and cvx_hyperplane_distances.min() <= 0 # .find_simplex(x) >= 0 

        pv = np.hstack(posVel2cyl.numpy_vectorized(x[None,:], self.cylinder_position, self.cylinder_radius)).flatten()
        pv_hyperplane_distances = self.non_cvx_delaunay.plane_distance(pv - np.array([self.cylinder_margin, 0]))
        satisfy_cyl = (pv_hyperplane_distances.max() >= 0 and cvx_hyperplane_distances.min() <= 0) or pv[1] > 0 # .find_simplex(pv) >= 0
        # satisfy_cyl = self.non_cvx_delaunay.find_simplex(pv) >= 0
        print(f"box_satisfied: {satisfy_box}")
        print(f"cyl_satisfied: {satisfy_cyl}")

        if satisfy_box and satisfy_cyl:
            return True
        else:
            return False

    @time_function
    def construct_cvx_constraint_safe_set(self, data):

        print('forming the convex hull of selected successful points during training... (this can take time)')
        time_start = time.time()
        hull = ConvexHull(data) # 10k was successful
        time_end = time.time()
        print(f'time taken: {time_end - time_start}')

        print('performing delaunay triangulation of the convex hull... (this can take time)')
        time_start = time.time()
        delaunay = Delaunay(data[hull.vertices])
        time_end = time.time()
        print(f'time taken: {time_end - time_start}')

        return delaunay, hull

    @time_function
    def construct_non_cvx_constraint_safe_set(self, data):

        c = np.array([[1.,1.]]*data.shape[0])
        pv = np.hstack(posVel2cyl.numpy_vectorized(data, c, self.cylinder_radius))

        cyl_hull = ConvexHull(pv)
        cyl_delaunay = Delaunay(pv[cyl_hull.vertices])
        
        # cyl_delaunay.find_simplex(point_to_check)
        return cyl_delaunay, cyl_hull
    
    @time_function
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

        all_successes = torch.vstack(log['success']).numpy()

        # Ensure you don't try to select more points than exist
        num_points_considered = min(num_points_considered, all_successes.shape[0])

        # Generate random unique indices
        random_indices = np.random.choice(all_successes.shape[0], size=num_points_considered, replace=False)

        # Select the points using the random indices
        successes = all_successes[random_indices, :]

        return successes

def generate_bf(history):

    print('-----------------------------------')
    print('GENERATING SAFE SET, TAKES ~ 2 MINS')
    print('-----------------------------------')

    cylinder_margin = 0.2 # 0.19
    cylinder_radius = 0.5
    num_points_considered = 100_000 # out of ~ 4_000_000

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

    all_successes = torch.vstack(log['success']).numpy()

    # Ensure you don't try to select more points than exist
    num_points_considered = min(num_points_considered, all_successes.shape[0])

    # Generate random unique indices
    random_indices = np.random.choice(all_successes.shape[0], size=num_points_considered, replace=False)

    # Select the points using the random indices
    successes = all_successes[random_indices, :]

    # abandoned clustering idea in favour of random sampling, worked better:

    # print('clustering begins...')
    # time_start = time.time()
    # kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    # # kmeans.fit(successes[-10_000:,:])
    # kmeans.fit(successes[-num_points_considered:,:])
    # successes = kmeans.cluster_centers_  # These are the centroids of the clusters
    # time_end = time.time()
    # print(f'time taken: {time_end - time_start}')

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

    # optional plotting:
    # _ = delaunay_plot_2d(cyl_delaunay)
    # plt.savefig('data/paper/cylinder_constraint_cvx_hull.svg')
    # points = pv_pos_lim[cyl_hull.vertices]
    # simplices = cyl_delaunay.simplices

    # plot_shifted_fs1_bf(points, simplices, delta=1.0)

    def cyl_bf(x):
        pv = np.hstack(posVel2cyl(x[None,:], c[:1,:], cylinder_radius)).flatten()
        # shift position back with the safety margin
        pv[0] += cylinder_margin
        return cyl_delaunay.find_simplex(pv)
    
    cyl_bf(point_to_check)

    # Construct Box Constraint Convex Hull
    # ------------------------------------

    # can do this for x, y, z individually as to simplify the computation - not same hull

    # x_hull = ConvexHull(successes[:,0::3]) # x, xdot
    # y_hull = ConvexHull(successes[:,1::3]) # y, ydot
    # z_hull = ConvexHull(successes[:,2::3]) # z, zdot

    # x_del = Delaunay(successes[x_hull.vertices][:,0::3])
    # y_del = Delaunay(successes[y_hull.vertices][:,1::3])
    # z_del = Delaunay(successes[z_hull.vertices][:,2::3])

    # simplify same problem by taking K-means clustering of convex hull to underapproximate it with less vertices

    print('forming the convex hull of selected successful points during training... (this can take time)')
    time_start = time.time()
    hull = ConvexHull(successes) # 10k was successful
    time_end = time.time()
    print(f'time taken: {time_end - time_start}')

    print('performing delaunay triangulation of the convex hull... (this can take time)')
    time_start = time.time()
    delaunay = Delaunay(successes[hull.vertices])
    time_end = time.time()
    print(f'time taken: {time_end - time_start}')

    box_bf = lambda x: delaunay.find_simplex(x)

    print('performing first simplex search... (this can take time - compiling cython functions in scipy)')
    time_start = time.time()
    print(box_bf(point_to_check))
    time_end = time.time()
    print(f'time taken: {time_end - time_start}')

    # interesection of the two safe sets form the CBF
    def bf(x):
        satisfy_box = box_bf(x) >= 0 
        satisfy_cyl = cyl_bf(x) >= 0
        print(f"box_satisfied: {satisfy_box}")
        print(f"cyl_satisfied: {satisfy_cyl}")
        if satisfy_box and satisfy_cyl:
            return True
        else:
            return False

    print('testing full system, should be fast now, cython functions already compiled...')
    time_start = time.time()
    is_inside = bf(point_to_check)
    time_end = time.time()
    print(f'time taken: {time_end - time_start}')

    # analysis:

    # test = delaunay.find_simplex(successes)
    # test2 = delaunay.find_simplex(all_successes)
    # num_elements_equal_to_minus_one = np.sum(test == -1)
    # num_elements_equal_to_minus_one2 = np.sum(test2 == -1)

    return bf
        
def plot_shifted_fs1_bf(points, simplices, delta, filename='data/paper/cylinder_constraint_convex_hull.svg'):
    
    shifted_points = np.copy(points)
    shifted_points[:, 0] += delta

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot - Original plot
    axs[0].triplot(points[:, 0], points[:, 1], simplices)
    axs[0].plot(points[:, 0], points[:, 1], 'o')
    axs[0].set_xlabel('$a_1$')
    axs[0].set_ylabel('$a_2$')
    axs[0].set_xlim(-0.2, 5)

    # Right subplot - Shifted plot only
    axs[1].triplot(shifted_points[:, 0], shifted_points[:, 1], simplices)
    axs[1].plot(shifted_points[:, 0], shifted_points[:, 1], 'o')
    axs[1].set_xlabel('$a_1$')
    axs[1].set_ylabel('$a_2$')
    axs[1].set_xlim(-0.2, 5)


    # Adding a line on the left-most section of the original plot
    left_most_x = np.min(points[:, 0])
    axs[1].axvline(x=left_most_x, color='green', linestyle='--')

    # Adding a line on the left-most section of the shifted plot
    left_most_shifted_x = np.min(shifted_points[:, 0])
    axs[1].axvline(x=left_most_shifted_x, color='green', linestyle='--')

    # Adding an arrow originating from the line to the shifted points
    arrow_start = (left_most_x, np.mean(points[:, 1]))
    arrow_end = (left_most_x + delta, np.mean(shifted_points[:, 1]))
    axs[1].annotate('', xy=arrow_end, xytext=arrow_start, 
                    arrowprops=dict(facecolor='black', shrink=0.05))

    # Adding delta symbol just above the arrow
    delta_pos = ((arrow_start[0] + arrow_end[0]) / 2, (arrow_start[1] + arrow_end[1]) / 2 + 0.05)  # Adjusting position
    axs[1].text(delta_pos[0], delta_pos[1], '$\delta$', fontsize=16, va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(filename)

def test_trajectory(bf, x_history):
    pass



if __name__ == "__main__":

    log = torch.load('large_data/nav_training_data.pt')
    bf = BarrierFunction(log)
    # bf = generate_bf(log)
    hull = bf.cvx_hull
    x = np.array([1.25, 0.2, 2.0, 0.02, 2.0, -0.3])

    # this x was the one given to the safety filter - halfspace says its good
    x = np.array([0.47738674, 0.8017296 , 1.3384718 , 1.0735897 , 0.65211016,
       0.3316325 ], dtype=np.float32)
    bf.is_in_safe_set(x)
    # this x was the one that violated the box constraints:

    # x_history = np.load("data/xu_sf_p2p_mj_0.001.npz")['x_history']
    # point = np.array([1.25, 0.2, 2.0, 0.02, 1.0, -0.3])
    # obs_history = np.vstack([x_history[:, i] for i in [0,7,1,8,2,9]])
    print('fin')