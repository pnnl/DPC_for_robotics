"""
Overview:

This whole file and utils/geometry.c are an efficient way to calculate
the closest simplex on the surface of a high dimensional convex hull
defined by halfspaces to an arbitrary point. We can simply use np.argmin
to do the same thing, but this is a brute force approach as it needs to 
search across all simplices. This is slow (800ms in my use case). This 
algorithm takes advantage of two facts:

1. This is a convex hull, and so if the point is outside of it, there is
   only one vertex/simplex on the surface of the hull that is closest to it.
   Therefore a locally optimal solution is globally optimal (convex problem).
2. The point that I am interested in finding the closest simplex to will move,
   but not by very much. Therefore the next solution will be close to the
   prior solution.

Algorithm:

Start at an initial guess (usually previous answer is good choice) (this can 
be advanced by using physics informed methods to predict where we will be next). Then
evaluate the distances to the centroids of every neighboring simplices to the point.
If none of these distances are shorter than the one we are at yet we call this a
candidate solution, if one of the distances is shorter we find all of its neighbors
and repeat the process. In this case neighbors are considered simplices that share a
common edge, this is a subset of simplices that share a vertex.

We know that the SciPy ConvexHull algorithm will ensure for an N dimensional 
hull, that one vertex will be shared by no more than N simplices. Therefore as
a final rigorous certificate of optimality I find all neighbors to the Nth degree
of the proposed final answer and validate that in this region there exists no
simplices closer to the point.

Notes:

- storing every simplex which shares a vertex with every other gets 
  out of hand in high dimensions in theory. As theoretically there
  can be almost every simplex sharing one "singularity" vertex. But 
  the scipy ConvexHull algorithm (I think its quickhull) ensures that
  vertices are equally spaced, so that no more than N simplices share
  a single vertex in an N dimensional convex hull.
- this means that in theory to evalutate every simplex surrounding
  a simplex we would need to search N * (N - 2) simplices directly 
  surrounding the simplex. Finding exactly what these simplices are 
  is the biggest problem. If we blindly search for N steps outside 
  of the simplex then we can get the same result, but with searching
  N + (N - 1)^N points, which is MUCH more expensive. This expensive 
  algorithm is what is implemented in the C code and in this code in
  sixth_degree_search (because I am operating in 6 dimensions).
- I tried to implement something that would find the optimal set of 
  simplices in the high dimensional space, but I could not think of
  the correct data structures for it/when working with what the SciPy
  ConvexHull gives me.
- Future work:
    - Complete the optimal simplices sharing vertices with the simplex
    - replace the SciPy QuickHull algorithm with a dimension agnostic 
      one: Computing the Approximate Convex Hull in High Dimensions
           https://arxiv.org/abs/1603.04422
"""

import numpy as np
import torch
import utils.pytorch as ptu
import ctypes
import subprocess
import time
from tqdm import tqdm
from copy import deepcopy
from dpc import posVel2cyl
from scipy.spatial import ConvexHull as SPConvexHull

np.random.seed(0)

# replicate the data structures in the C so that we can manipulate/create them using ctypes
# -----------------------------------------------------------------------------------------

class Point(ctypes.Structure):
    _fields_ = [("coordinates", ctypes.POINTER(ctypes.c_double))]

class Vector(ctypes.Structure):
    _fields_ = [("components", ctypes.POINTER(ctypes.c_double))]

class Simplex(ctypes.Structure):
    _fields_ = [
        ("vertexIndices", ctypes.POINTER(ctypes.c_int)),
        ("numVertices", ctypes.c_int),
        ("neighborIndices", ctypes.POINTER(ctypes.c_int)),
        ("directions", ctypes.POINTER(Vector)),
        ("centroid", ctypes.POINTER(Point))
    ]

class ConvexHull(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.POINTER(Point)),
        ("numPoints", ctypes.c_int),
        ("simplices", ctypes.POINTER(Simplex)),
        ("numSimplices", ctypes.c_int),
        ("dimension", ctypes.c_int)
    ]

# compile the C and setup the ctypes things
# -----------------------------------------

def get_geometry_shared_libary(compile=True):

    if compile is True:
        process = subprocess.run("gcc -fPIC -shared -o utils/geometry.so utils/geometry.c", shell=True)
        time.sleep(0.2)

    # Load the shared library
    lib = ctypes.CDLL('./utils/geometry.so')

    # Input the argtypes for InstantiateConvexHull function
    lib.InstantiateConvexHull.argtypes = [
        ctypes.POINTER(ConvexHull),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),  # Neighbors array
        ctypes.POINTER(ctypes.c_double),  # Centroid data
        ctypes.c_int  # Dimension
    ]

    # Assuming FindNearestSimplexBruteForce is defined in your C library
    lib.FindNearestSimplexBruteForce.argtypes = [ctypes.POINTER(ConvexHull), ctypes.POINTER(Point)]
    lib.FindNearestSimplexBruteForce.restype = None  # or appropriate return type

    return lib


# define the safe set from the scipy ConvexHulls and DPC training dataset
# -----------------------------------------------------------------------
class SafeSet:
    def __init__(self, data) -> None:
        self.cylinder_radius = 0.5
        self.cylinder_position = np.array([[1.,1.]])
        self.cylinder_margin = 0.15
        num_points_considered = 100_000

        print(f"----------------------------------------------------------------------------------")
        print(f"generating safe set from DPC training dataset using {num_points_considered} points")
        print(f"----------------------------------------------------------------------------------")

        print('preprocessing data...')
        cvx_safe_points, non_cvx_safe_points = self.preprocess_data(data, num_points_considered)
        safe_points = [cvx_safe_points, non_cvx_safe_points]

        print('generating scipy convex hulls...')
        sphulls = [SPConvexHull(p) for p in safe_points]

        print('transferring data to the C...')
        # libs = [self._init_hull(hull) for hull in sphulls]
        self.lib, self.hull = self._init_hull(sphulls[0])

        print('running test...')
        tic = time.time()
        self.find_nearest_simplex(np.array([[3.,1.,3.,1.,3.,1.]]), self.lib)
        toc = time.time()
        print(f"time taken in test to find simplex: {toc-tic}")

        print('fin')

    def find_nearest_simplex(self, point, simplex_guess, brute_force=False):
        point_struct = Point(point.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if brute_force is False:
            self.lib.FindNearestSimplexDirected(ctypes.byref(self.hull), simplex_guess, ctypes.byref(point_struct))
        else:
            self.lib.FindNearestSimplexBruteForce(ctypes.byref(self.hull), ctypes.byref(point_struct))
        # return closest simplex to warm start on

    def _init_hull(self, sphull):

        # create the library for this hull
        lib = get_geometry_shared_libary(compile=True)

        # assign the processed data in sphull to the C library lib
        points = sphull.points
        vertices = points[sphull.vertices]
        dimension = points.shape[1]
        simplices = sphull.simplices
        neighbors = sphull.neighbors
        centroids = np.mean(points[simplices], axis=1)

        hull = ConvexHull()
        points_c = points.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        simplices_c = simplices.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        neighbors_c = neighbors.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        centroids_c = centroids.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Assign these values to the ConvexHull in C with this constructor
        lib.InstantiateConvexHull(
            ctypes.byref(hull),
            points_c, len(points),
            simplices_c, len(simplices),
            neighbors_c,
            centroids_c,  # Include the centroids
            dimension)

        return lib, hull

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
    bf = SafeSet(data)
    print('fin')