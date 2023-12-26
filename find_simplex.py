import casadi as ca
import torch
import numpy as np
from bf import BarrierFunction

barrier_func = BarrierFunction(torch.load('large_data/nav_training_data.pt'))

x = np.array([.5,0.0,0.0,0.0,0.0,0.0])

# Convert points to CasADi parameters or constants
casadi_points = [ca.MX(p) for p in barrier_func.cvx_delaunay.points]

# Convert simplex indices to a CasADi-friendly format
# Note: This is a simple representation and might need to be adapted based on how you intend to use the simplexes
casadi_simplices = [ca.MX(simplex) for simplex in barrier_func.cvx_delaunay.simplices]

# Calculate centroid of each simplex (very simplified conceptual example)
centroids = [ca.sum1(ca.vertcat(*[casadi_points[idx] for idx in simplex])) / len(simplex) for simplex in barrier_func.cvx_delaunay.simplices]

# Calculate distance from point x to each centroid (using Euclidean distance here for simplicity)
distances = [ca.norm_2(x - centroid) for centroid in centroids]

# Conceptual step: Identify the index of the simplex with the minimum distance
# Note: This step is more of a placeholder as finding the exact minimum index isn't straightforward in CasADi without a loop or conditional statement
min_dist_index = ca.argmin(ca.vertcat(*distances))  # This is conceptual; CasADi doesn't directly support argmin like this
