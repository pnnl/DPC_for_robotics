import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from time import time


def generate_random_points_on_sphere(num_points, radius=1):
    # Using spherical coordinates method
    theta = np.random.uniform(0, 2*np.pi, num_points)  # Angle from 0 to 2pi
    phi = np.arccos(2*np.random.uniform(0, 1, num_points) - 1)  # Angle from 0 to pi
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.vstack((x, y, z)).T

# Generate random points
num_points = 10000  # Number of points on the sphere
points = generate_random_points_on_sphere(num_points)

# Define a new point outside the sphere
# Here as an example, 1.5 times the radius along the x-axis
outlier_point = np.array([[1.5, 0, 0]])

# Create the convex hull
hull = ConvexHull(points)

# Plotting the convex hull
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2])  # plot the points

# Plot the new point outside the sphere
ax.scatter(outlier_point[:,0], outlier_point[:,1], outlier_point[:,2], color='r', label='Outlier Point', s=100)  # s is the size of the point

# Plot the convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Convex Hull of Random Points on a Sphere')
plt.savefig('test.png')

# ============================================================

start = np.array([0])
simplex_idx = np.copy(start)
search_step_depth = 3
centroids = np.mean(hull.points[hull.simplices], axis=1)

# Get all neighbors until search_depth is satisfied
tic = time()

ordered_neighbors = []
previously_visited = set(simplex_idx)
for _ in range(search_step_depth):
    simplex_idx = hull.neighbors[simplex_idx]
    simplex_idx = np.setdiff1d(simplex_idx, list(previously_visited)) # filter out previously visited
    ordered_neighbors.append(simplex_idx)
    previously_visited.update(simplex_idx)

degree_lengths = [len(i) for i in ordered_neighbors]
neighbors = np.hstack([start, np.hstack(ordered_neighbors)])
distances = np.linalg.norm(centroids[neighbors] - outlier_point, axis=1)
min_idx = np.argmin(distances)
min_distance = distances[min_idx]

# now we must see if the minimum distance is on the interior of the search space
if min_idx - 1 > degree_lengths[-2]:
    # point on exterior or outside of search space
    min_distance # we are done
else:
    # we have found the closest point
    start = np.array([neighbors[min_idx]])

    # repeat - excluding previously visited locations
toc = time()
print(toc-tic)


tic = time()
control_distances = np.linalg.norm(centroids - outlier_point, axis=1)
control_min_idx = np.argmin(control_distances)
control_min_distance = control_distances[control_min_idx]
toc = time()
print(toc-tic)

def control():
    control_distances = np.linalg.norm(centroids - outlier_point, axis=1)
    control_min_idx = np.argmin(control_distances)
    control_min_distance = control_distances[control_min_idx]

def search_convex_hull(hull, simplex_idx, search_step_depth, outlier_point):
    ordered_neighbors = []
    previously_visited = set(simplex_idx)
    complete = False
    while complete is False:

        for _ in range(search_step_depth):
            simplex_idx = hull.neighbors[simplex_idx]
            simplex_idx = np.setdiff1d(simplex_idx, list(previously_visited))  # filter out previously visited
            ordered_neighbors.append(simplex_idx)
            previously_visited.update(simplex_idx)

        degree_lengths = [len(i) for i in ordered_neighbors]
        neighbors = np.hstack([simplex_idx, np.hstack(ordered_neighbors)])
        distances = np.linalg.norm(centroids[neighbors] - outlier_point, axis=1)  # Adjusted for points
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        if min_idx - 1 > degree_lengths[-2]:
            complete = True
            print(f'min_distance: {min_distance}')
            return min_distance  # Point on exterior or outside of search space
        else:
            simplex_idx = neighbors[min_idx]
            # return np.array([neighbors[min_idx]])  # Found the closest point

def search_convex_hull_physics_informed(hull, old_simplex, old_point, search_step_depth, point):
    pass

import cProfile
cProfile.run('search_convex_hull(hull, simplex_idx, search_step_depth, outlier_point)')
# cProfile.run('control()')

print('fin')
