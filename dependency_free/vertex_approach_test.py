import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt

from redundant.safe_set import SafeSet

log = torch.load('large_data/nav_training_data.pt')
ss = SafeSet(log)

point = np.array([4., -1.])
hull = ss.non_cvx_hull

vertices = hull.points[hull.vertices]
dim = vertices.shape[1]

"""Experiment 1: single point centroids approach vs vertices approach"""

# centroids approach
centroids_approach = {}
centroids_approach['centroids'] = np.mean(hull.points[hull.simplices], axis=1)
centroids_approach['distances'] = np.linalg.norm(centroids_approach['centroids'] - point, axis=1)
centroids_approach['closest_simplex'] = np.argmin(centroids_approach['distances'])
centroids_approach['equation'] = hull.equations[centroids_approach['closest_simplex']]
nc = centroids_approach['centroids'][centroids_approach['closest_simplex']]
a, b, c = centroids_approach['equation'][[0, 1, 2]]
centroids_approach['x'] = np.linspace(-1 + nc[0], 1 + nc[0], 2)
centroids_approach['y'] = (-a * centroids_approach['x'] - c) / b
centroids_m = -a/b
centroids_c = -c/b

# vertices approach
vertices_approach = {}
vertices_approach['distances'] = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)
vertices_approach['argmins'] = np.argsort(vertices_approach['distances'])[:dim]
vertices_approach['A_cvx'] = hull.points[hull.vertices][vertices_approach['argmins']]
vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 0), (0, 1)), constant_values=1)
vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 1), (0, 0)), constant_values=0)
vertices_approach['equation'] = sp.linalg.null_space(vertices_approach['A_cvx']).flatten()
a, b, c = vertices_approach['equation'][[0, 1, 2]]
vertices_approach['x'] = np.linspace(-1 + nc[0], 1 + nc[0], 2)
vertices_approach['y'] = (-a * vertices_approach['x'] - c) / b
vertices_m = -a/b
vertices_c = -c/b

plt.plot(np.hstack([vertices[:,0], vertices[0,0]]), np.hstack([vertices[:,1], vertices[0,1]]), marker='o')
plt.scatter(point[0], point[1], color='r')
plt.plot(centroids_approach['x'], centroids_approach['y'])
plt.axis('equal')
plt.plot(vertices_approach['x'], vertices_approach['y'])
plt.show()

# conclusion: these two methods are the same for this one point in 2D
print(f"centroids: y = {centroids_m}x + {centroids_c}")
print(f"vertices:  y = {vertices_m}x + {vertices_c}")

"""Experiment 2: grid of points centroids approach vs vertices approach"""

# lets now try a grid of points
x_min = -1
x_max = 5
y_min = -3
y_max = 5
grid_resolution = 0.1  # Adjust this to control the density of grid points

# Create a grid of x and y coordinates
x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)

# Generate a grid of 2D points by taking all combinations of x and y
point_grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

# Create an empty array to store the results for each point in the grid
results_centroids = []

for point in point_grid:
    centroids_approach['distances'] = np.linalg.norm(centroids_approach['centroids'] - point, axis=1)
    centroids_approach['closest_simplex'] = np.argmin(centroids_approach['distances'])
    centroids_approach['equation'] = hull.equations[centroids_approach['closest_simplex']]
    nc = centroids_approach['centroids'][centroids_approach['closest_simplex']]
    a, b, c = centroids_approach['equation'][[0, 1, 2]]
    centroids_approach['x'] = np.linspace(-1 + nc[0], 1 + nc[0], 2)
    centroids_approach['y'] = (-a * centroids_approach['x'] - c) / b
    centroids_m = -a/b
    centroids_c = -c/b

    # Append the results for the current point to the results array
    results_centroids.append((centroids_m, centroids_c))

# Convert the results to a NumPy array
results_centroids = np.round(np.array(results_centroids), 8)

# vertices_approach
results_vertices = []
for point in point_grid:
    vertices_approach['distances'] = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)
    vertices_approach['argmins'] = np.argsort(vertices_approach['distances'])[:dim]
    vertices_approach['A_cvx'] = hull.points[hull.vertices][vertices_approach['argmins']]
    vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 0), (0, 1)), constant_values=1)
    vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 1), (0, 0)), constant_values=0)
    vertices_approach['equation'] = sp.linalg.null_space(vertices_approach['A_cvx']).flatten()
    a, b, c = vertices_approach['equation'][[0, 1, 2]]
    vertices_approach['x'] = np.linspace(-1 + nc[0], 1 + nc[0], 2)
    vertices_approach['y'] = (-a * vertices_approach['x'] - c) / b
    vertices_m = -a/b
    vertices_c = -c/b
    results_vertices.append((vertices_m, vertices_c))

# Convert the results to a NumPy array
results_vertices = np.round(np.array(results_vertices), 8)


# Check which rows are not equal between results_vertices and results_centroids
discrepancy_indices = np.where(~np.all(results_vertices == results_centroids, axis=1))[0]

# Initialize an empty list to store the row indices with discrepancies
discrepancy_rows = []

# Initialize an empty list to store the horizontally stacked arrays
stacked_arrays = []

# Iterate through the discrepancy_indices
for idx in discrepancy_indices:
    # Stack the corresponding rows from results_vertices and results_centroids horizontally
    stacked_row = np.hstack((results_vertices[idx], results_centroids[idx]))
    
    # Append the stacked row to the stacked_arrays list
    stacked_arrays.append(stacked_row)
    
    # Append the row index to the discrepancy_rows list
    discrepancy_rows.append(idx)

# Convert the stacked arrays and discrepancy rows to numpy arrays
stacked_arrays = np.array(stacked_arrays)
discrepancy_rows = np.array(discrepancy_rows)

plt.plot(np.hstack([vertices[:,0], vertices[0,0]]), np.hstack([vertices[:,1], vertices[0,1]]), marker='o')
# Extract the grid points with discrepancies from the point_grid
grid_points_with_discrepancies = point_grid[discrepancy_rows]

# Plot the grid points with discrepancies
plt.scatter(grid_points_with_discrepancies[:, 0], grid_points_with_discrepancies[:, 1], color='red', marker='x', label='Discrepancies')



print('fin')