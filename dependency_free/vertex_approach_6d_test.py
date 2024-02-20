import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt

from redundant.safe_set import SafeSet

log = torch.load('large_data/nav_training_data.pt')
ss = SafeSet(log)

point = np.array([-0.58833, 0.78767, 0.96962, -0.056039, -0.22199, 0.2492])
hull = ss.cvx_hull

# vertices = hull.points[hull.vertices]
simplices = hull.simplices
dim = hull.points.shape[1]

"""Experiment 1: single point centroids approach vs vertices approach"""

# centroids approach - validated
centroids_approach = {}
centroids_approach['centroids'] = np.mean(hull.points[hull.simplices], axis=1)
centroids_approach['distances'] = np.linalg.norm(centroids_approach['centroids'] - point, axis=1)
centroids_approach['closest_simplex'] = np.argmin(centroids_approach['distances'])
centroids_approach['equation'] = hull.equations[centroids_approach['closest_simplex']]
nc = centroids_approach['centroids'][centroids_approach['closest_simplex']]

print(f"centroids_approach distance to closest centroid: {np.linalg.norm(hull.points[simplices[centroids_approach['closest_simplex']]] - point, axis=0)}")

centroids_approach['closest_vertices'] = hull.points[simplices[centroids_approach['closest_simplex']]]
test = np.mean(centroids_approach['closest_vertices'], axis=0) 
test3 = np.linalg.norm(centroids_approach['closest_vertices'] - point, axis=1)
dist = np.linalg.norm(test - point) # produces the same distance as we expect

# vertices approach
vertices_approach = {}
vertices_approach['distances'] = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)
vertices_approach['argmins'] = np.argsort(vertices_approach['distances']) # [:dim]
# check points are not colinear, search nearest vertices until not colinear
guess = np.arange(6)
A = hull.points[hull.vertices][vertices_approach['argmins']][guess]
SV = 1e-10      # singular value threshold
detT = 1e-10    # determinant threshold
i = 0
while np.linalg.det(A) < detT:
    print(i)
    i += 1
    U, S, VT = np.linalg.svd(A)
    colinear_points = np.where(S < SV)[0]
    if len(colinear_points) == 0:
        vertices_approach['A_cvx'] = A
        break
    # add new points from argsorted points
    new_points = np.arange(guess[-1] + 1, guess[-1] + len(colinear_points) + 1)
    # remove colinear points
    guess = np.hstack([guess, new_points])
    guess = np.delete(guess, colinear_points)
    A = hull.points[hull.vertices][vertices_approach['argmins']][guess]
vertices_approach['A_cvx'] = A
vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 0), (0, 1)), constant_values=1)
vertices_approach['A_cvx'] = np.pad(vertices_approach['A_cvx'], ((0, 1), (0, 0)), constant_values=0)
vertices_approach['equation'] = sp.linalg.null_space(vertices_approach['A_cvx']).flatten() # needs to be a 1D nullspace!
centroid = np.mean(A, axis=0)
print(f"vertices approach distance to closest centroid: {np.linalg.norm(centroid - point)}")

# lets check that the centroids approach vertices of closest simplex are not closer...
v = hull.points[simplices[centroids_approach['closest_simplex']]]
t = np.linalg.norm(v - point, axis=0)
t2 = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)

# here are the two sets of vertices, the equations comparison has fairly little meaning, other than angles compared against each other i guess
centroids_approach['closest_vertices'] = np.linalg.norm(hull.points[simplices[centroids_approach['closest_simplex']]] - point, axis=1)
vertices_approach['closest_vertices'] = np.linalg.norm(A - point, axis=1)

# lets try reconstructing the closest centroids simplex equation using the vertices approach and compare it with the
# one generated by scipy as a proof of concept
centroids_approach['A_cvx'] = hull.points[simplices[centroids_approach['closest_simplex']]]
centroids_approach['A_cvx'] = np.pad(centroids_approach['A_cvx'], ((0, 0), (0, 1)), constant_values=1)
centroids_approach['A_cvx'] = np.pad(centroids_approach['A_cvx'], ((0, 1), (0, 0)), constant_values=0)
centroids_approach['equation_test'] = sp.linalg.null_space(centroids_approach['A_cvx']).flatten() # needs to be a 1D nullspace!

# comparing centroids_approach equation_test and equation we see they are the same to 2 Significant Figures, I wonder why not more...
# then again this shows its roughly correct to use the scipy null_space in this way lol


"""
Results:

- I have seen that the vertices approach does indeed find vertices that are closer than those of the centroids approach for 
a single point. 
- I have shown that forming the plane equation using the scipy null_space method on the vertices of the centroids approach
is only equivalent to the scipy convex hull normal equation to 2 significant figures. It is unclear whether the discrepancy
is an inaccuracy of the convex hull formulation or the svd null space formulation. That being said I think they are close
enough for my purposes here.

These two results give me confidence in the accuracy of the vertices approach, that being said I can tune the SV and detT thresholds
until the matrix is well conditioned if I choose down the line. Unlike the 2D case I cannot do a grid search as a sanity check on the final result.
"""

