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
import ctypes
import subprocess
import time
from tqdm import tqdm
from copy import deepcopy
np.random.seed(0)

process = subprocess.run("gcc -fPIC -shared -o utils/geometry.so utils/geometry.c", shell=True)
time.sleep(0.2)

# Load the shared library
lib = ctypes.CDLL('./utils/geometry.so')

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

# define the argument type and results type of the functions in the C
# -------------------------------------------------------------------

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

# Lets get REAL data to test this 
# -------------------------------

from barrier_function import BarrierFunction
import matplotlib.pyplot as plt
import torch

# load the data so we don't need to regenerate every time
generate_new_data = False
hull_type = 'cvx' # 'non_cvx', 'cvx'
if generate_new_data is False:
    points_np =         np.load(f'data/geometry/{hull_type}_points_np.npy')
    vertices_np =       np.load(f'data/geometry/{hull_type}_vertices_np.npy')
    dimension =         int(np.load(f'data/geometry/{hull_type}_dimension.npy'))
    simplices_np =      np.load(f'data/geometry/{hull_type}_simplices_np.npy')
    neighbors_np =      np.load(f'data/geometry/{hull_type}_neighbors_np.npy')
else:
    # generate data - extract data - save data
    bf = BarrierFunction(torch.load('large_data/nav_training_data.pt'))
    if hull_type == 'cvx':
        hull = bf.cvx_hull
    elif hull_type == 'non_cvx':
        hull = bf.non_cvx_hull
    points_np = hull.points
    vertices_np = points_np[hull.vertices]
    dimension = points_np.shape[1]
    simplices_np = hull.simplices
    neighbors_np = hull.neighbors
    np.save(f'data/geometry/{hull_type}_points_np.npy', points_np)
    np.save(f'data/geometry/{hull_type}_vertices_np.npy', vertices_np)
    np.save(f'data/geometry/{hull_type}_dimension.npy', dimension)
    np.save(f'data/geometry/{hull_type}_simplices_np.npy', simplices_np)
    np.save(f'data/geometry/{hull_type}_neighbors_np.npy', neighbors_np)

centroids_np = np.mean(points_np[simplices_np], axis=1)

if hull_type == 'non_cvx':
    dummy_point = np.array([[1.,1.]])
elif hull_type == 'cvx':
    dummy_point = np.array([[3.,1.,3.,1.,3.,1.]])


# plot the convex hull itself
# plt.close()
# plt.plot(np.hstack([vertices_np[:,0], vertices_np[0,0]]), np.hstack([vertices_np[:,1], vertices_np[0,1]]), linestyle='-', marker='o', color='grey')
# # plot the centroids of the simplices
# plt.scatter(centroids_np[:,0], centroids_np[:,1], marker='o', color='r')
# # plot dummy point traversal
# plt.plot(np.hstack([dummy_point[0]]), np.hstack([dummy_point[1]]), linestyle='-', marker='o', color='b')
# 
# plt.savefig('test.png')


# Create the C datatypes and instantiate the ConvexHull
# -----------------------------------------------------

# Create an instance of ConvexHull in Python to hold the pointer
hull = ConvexHull()

# Flatten points array and convert to ctypes
points_flat = points_np.flatten()
points_c = points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Flatten simplices array and convert to ctypes
simplices_flat = simplices_np.flatten()
simplices_c = simplices_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

# Flatten neighbors array and convert to ctypes
neighbors_flat = neighbors_np.flatten()
neighbors_c = neighbors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

# Same with centroids
centroids_flat = centroids_np.flatten()
centroids_c = centroids_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

index = 3079679

# print(f"Python centroid_data: {centroids_flat[index-1+100:index+2+100]}")


print("specific index pointer:")
print(centroids_c[index:index+1]._arr.data)
print("full array pointer:")
print(centroids_c._arr.data)

[print(v) for k,v in centroids_c.contents._b_base_._objects.items()]

# Assign these values to the ConvexHull in C with this constructor
lib.InstantiateConvexHull(
    ctypes.byref(hull),
    points_c, len(points_np),
    simplices_c, len(simplices_np),
    neighbors_c,
    centroids_c,  # Include the centroids
    dimension)

# Same with dummy point for testing
dummy_point_flat = dummy_point.flatten()
dummy_point_c = dummy_point_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# Create an instance of Point and assign the coordinates
dummy_point_struct = Point(dummy_point_c)



# Test the brute force find distances to all simplices function against scipy
# ---------------------------------------------------------------------------

"""
mine: 0.11s
scipy: 0.081s
"""

# Assuming FindNearestSimplexBruteForce is defined in your C library
lib.FindNearestSimplexBruteForce.argtypes = [ctypes.POINTER(ConvexHull), ctypes.POINTER(Point)]
lib.FindNearestSimplexBruteForce.restype = None  # or appropriate return type

start_time = time.time()
# Pass the ConvexHull reference and the dummy point to the C function
lib.FindNearestSimplexBruteForce(ctypes.byref(hull), ctypes.byref(dummy_point_struct))
end_time = time.time()
print(f"my time: {end_time - start_time}")

# test this against the scipy if bf exists (new data generated)
try: # typically takes 0.0794s vs my methods 0.11s in brute force
    if hull_type == 'cvx':
        start_time = time.time()
        distances = bf.cvx_delaunay.plane_distance(np.array([1.,1.,1.,1.,1.,1.]))
        out = np.argmin(np.abs(distances))
        end_time = time.time()
        print(f"scipy time: {end_time - start_time}")
    elif hull_type == 'non_cvx':
        start_time = time.time()
        distances = bf.non_cvx_delaunay.plane_distance(np.array([1.,1.]))
        out = np.argmin(np.abs(distances))
        end_time = time.time()
        print(f"scipy time: {end_time - start_time}")        
except:
    print('no new data generated, no scipy hull constructed to compare against')

# Ok now lets go with the directed search
# ---------------------------------------

"""
mine:
no scipy
"""

simplex_idx = 1000
tic = time.time()
lib.FindNearestSimplexDirected(ctypes.byref(hull), simplex_idx, ctypes.byref(dummy_point_struct))
toc = time.time()
print(f"my algs time: {toc-tic}")
print('fin')
# simplex_idx = 68144
# lib.FindNearestSimplexDirected(ctypes.byref(hull), simplex_idx, ctypes.byref(dummy_point_struct))

# distances = np.linalg.norm(centroids_np - dummy_point, axis=1)
# control_answer = np.argmin(distances)
# 
# def python_version(simplex_idx):
#     found = False
#     while not found:
#         neighbors = neighbors_np[simplex_idx]
#         distances = np.linalg.norm(centroids_np[neighbors] - dummy_point, axis=1, ord=2)
#         start_distance = np.linalg.norm(centroids_np[simplex_idx] - dummy_point, ord=2)
#         if (start_distance < distances).all():
#             print("found")
#             found = True
#             print(f"closest distance: {start_distance}")
#         else:
#             simplex_idx = neighbors[np.argmin(distances)]
#             print(f"current closest: {np.min(distances)}")
# 
# python_version(simplex_idx)
# print(f"control answer: {distances[control_answer]}")
# 
# # lets see if the neighbors of the correct distance and their centroids line up
# 
# 
# def distances_to(simplex_indices):
#     return np.linalg.norm(centroids_np[simplex_indices] - dummy_point, axis=1, ord=2)
# 
# distances = distances_to(neighbors_np[control_answer])
# 
# simplices_list = list(range(simplices_np.shape[0]))

# ----------------------------------------
# lets do this in 3D so I can visualise it
# ----------------------------------------

# from scipy.spatial import ConvexHull as SPConvexHull
# 
# def generate_random_points_on_sphere(num_points, radius=1):
#     # Using spherical coordinates method
#     theta = np.random.uniform(0, 2*np.pi, num_points)  # Angle from 0 to 2pi
#     phi = np.arccos(2*np.random.uniform(0, 1, num_points) - 1)  # Angle from 0 to pi
#     x = radius * np.sin(phi) * np.cos(theta)
#     y = radius * np.sin(phi) * np.sin(theta)
#     z = radius * np.cos(phi)
#     return np.vstack((x, y, z)).T
# 
# # Generate random points
# num_points = 100  # Number of points on the sphere
# points = generate_random_points_on_sphere(num_points)
# 
# # Define a new point outside the sphere
# # Here as an example, 1.5 times the radius along the x-axis
# dummy_point = np.array([[1.5, 0, 0]])
# 
# # Create the convex hull
# hull = SPConvexHull(points)
# points_np = hull.points
# vertices_np = points_np[hull.vertices]
# dimension = points_np.shape[1]
# simplices_np = hull.simplices
# neighbors_np = hull.neighbors
# centroids_np = np.mean(points_np[simplices_np], axis=1)
# 
# # Plotting the convex hull
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=10)  # plot the points
# 
# # Plot the new point outside the sphere
# ax.scatter(dummy_point[:,0], dummy_point[:,1], dummy_point[:,2], color='r', label='Outlier Point', s=100)  # s is the size of the point
# 
# # Plot the convex hull
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')
# 
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Convex Hull of Random Points on a Sphere')
# 
# form the control answer:
# ------------------------
# distances = np.linalg.norm(centroids_np - dummy_point, axis=1, ord=2)
# control_answer = np.argmin(distances)
# print(f"control smallest distance: {distances[control_answer]}")
# _ = centroids_np[control_answer]
# ax.scatter(_[0], _[1], _[2], s=50, color='orange', label='true closest centroid')

# do the other algorithms answer:
# -------------------------------
# def python_version(simplex_idx):
#     found = False
#     while not found:
#         neighbors = neighbors_np[simplex_idx]
#         distances = np.linalg.norm(centroids_np[neighbors] - dummy_point, axis=1, ord=2)
#         start_distance = np.linalg.norm(centroids_np[simplex_idx] - dummy_point, ord=2)
#         if (start_distance < distances).all():
#             # print("found")
#             found = True
#             # print(f"closest distance: {start_distance}")
#         else:
#             simplex_idx = neighbors[np.argmin(distances)]
#             # print(f"current closest: {np.min(distances)}")
#     return simplex_idx
# 
# alg_answer = python_version(0)
# print(f"alg smallest distance: {distances[alg_answer]}")
# _ = centroids_np[alg_answer]
# ax.scatter(_[0], _[1], _[2], s=50, color='pink', label='alg closest centroid')
# 
# for random_start in np.random.randint(0, simplices_np.shape[0], 10):
# 
#     print(f"alg smallest distance: {distances[python_version(random_start)]}")
# 
# # finish off the plot:
# # --------------------
# plt.savefig('test.png', dpi=400)

# Results were that the directed search == brute force search, lets try a synthetic 6D dataset

# create a 6D convex hull
# -----------------------
# 
# def generate_random_points_on_hypersphere(num_points, dimension=6, radius=1):
#     # Generate points with normal distribution in each dimension
#     points = np.random.normal(size=(num_points, dimension))
# 
#     # Normalize each point to lie on the hypersphere of given radius
#     norms = np.linalg.norm(points, axis=1)
#     points = points / norms[:, np.newaxis] * radius
# 
#     return points
# 
# # Generate random points
# num_points = 100  # Number of points on the sphere
# points = generate_random_points_on_hypersphere(num_points)
# 
# # Define a new point outside the sphere
# # Here as an example, 1.5 times the radius along the x-axis
# dummy_point = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
# 
# # Create the convex hull
# hull = SPConvexHull(points)
# points_np = hull.points
# vertices_np = points_np[hull.vertices]
# dimension = points_np.shape[1]
# simplices_np = hull.simplices
# neighbors_np = hull.neighbors
# centroids_np = np.mean(points_np[simplices_np], axis=1)
# 
# # form the control answer:
# # ------------------------
distances = np.linalg.norm(centroids_np - dummy_point, axis=1, ord=2)
control_answer = np.argmin(distances)
# print(f"control smallest distance: {distances[control_answer]}")
# _ = centroids_np[control_answer]

# do the other algorithms answer:
# -------------------------------

def first_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, ord=2)
    n = neighbors_np[candidate]
    d = np.linalg.norm(centroids_np[n] - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return n[i]

def second_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, axis=1, ord=2)
    nn = []
    for n in neighbors_np[candidate]:
        nn.append(neighbors_np[n])
    nn = np.unique(np.hstack(nn))
    mask = np.isin(nn, candidate)
    nnm = nn[~mask]
    c = centroids_np[nnm]
    d = np.linalg.norm(c - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return nnm[i]
    
def third_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, axis=1, ord=2)
    nnn = []
    for n in neighbors_np[candidate]:
        for nn in neighbors_np[n]:
            nnn.append(neighbors_np[nn])
    nnn = np.unique(np.hstack(nnn))
    mask = np.isin(nnn, candidate)
    nnnm = nnn[~mask]
    c = centroids_np[nnnm]
    d = np.linalg.norm(c - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return nnnm[i]
    
def fourth_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, axis=1, ord=2)
    nnnn = []
    for n in neighbors_np[candidate]:
        for nn in neighbors_np[n]:
            for nnn in neighbors_np[nn]:
                nnnn.append(neighbors_np[nnn])
    nnnn = np.unique(np.hstack(nnnn))
    mask = np.isin(nnnn, candidate)
    nnnnm = nnnn[~mask]
    c = centroids_np[nnnnm]
    d = np.linalg.norm(c - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return nnnnm[i]
    
def fifth_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, axis=1, ord=2)
    n5 = []
    for n in neighbors_np[candidate]:
        for nn in neighbors_np[n]:
            for nnn in neighbors_np[nn]:
                for nnnn in neighbors_np[nnn]:
                    n5.append(neighbors_np[nnnn])
    n5 = np.unique(np.hstack(n5))
    mask = np.isin(n5, candidate)
    n5m = n5[~mask]
    c = centroids_np[n5m]
    d = np.linalg.norm(c - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return n5m[i]
    
def sixth_degree_search(candidate):
    d_ctrl = np.linalg.norm(centroids_np[candidate] - dummy_point, axis=1, ord=2)
    n6 = []
    for n1 in neighbors_np[candidate]:
        for n2 in neighbors_np[n1]:
            for n3 in neighbors_np[n2]:
                for n4 in neighbors_np[n3]:
                    for n5 in neighbors_np[n4]:
                        n6.append(neighbors_np[n5])
    n6 = np.unique(np.hstack(n6))
    mask = np.isin(n6, candidate)
    n6m = n6[~mask]
    c = centroids_np[n6m]
    d = np.linalg.norm(c - dummy_point, axis=1, ord=2)
    i = np.argmin(d)
    if d_ctrl <= d[i]:
        return candidate
    else:
        return n6m[i]  
    
def python_version(simplex_idx, debug=False):
    found=False
    current_candidate = simplex_idx
    while not found:
        prior_candidate = current_candidate
        current_candidate = first_degree_search(current_candidate)
        # print(f"current candidate: {current_candidate}")
        # print(f"prior_candidate: {prior_candidate}")
        if current_candidate == prior_candidate:
            # current_candidate = second_degree_search(current_candidate)
            current_candidate = sixth_degree_search(current_candidate)
            if current_candidate == prior_candidate:
                return current_candidate
            # if current_candidate == prior_candidate:
            #     current_candidate = third_degree_search(current_candidate)
            #     if current_candidate == prior_candidate:
            #         current_candidate = fourth_degree_search(current_candidate)
            #         if current_candidate == prior_candidate:
            #             current_candidate = fifth_degree_search(current_candidate)
            #             if current_candidate == prior_candidate:
            #                 return current_candidate




alg_answer = python_version(0)
print(f"alg smallest distance: {distances[alg_answer]}")

tic = time.time()
for random_start in tqdm(np.random.randint(0, simplices_np.shape[0], 1000)):
    lib.FindNearestSimplexDirected(ctypes.byref(hull), simplex_idx, ctypes.byref(dummy_point_struct))
toc = time.time()
print(f"johns looped time: {toc-tic}")

tic = time.time()
for random_start in tqdm(np.random.randint(0, simplices_np.shape[0], 1000)):
    nearest_simplex = python_version(random_start, debug=False)
toc = time.time()
print(f"python looped time: {toc-tic}")

for random_start in tqdm(np.random.randint(0, simplices_np.shape[0], 1000)):
# for random_start in tqdm(range(simplices_np.shape[0])): # brute force EVERY POINT - should find a singularity in the hull if one exists
    nearest_simplex = python_version(random_start, debug=False)
    min_distance = distances[nearest_simplex]
    # print(f"alg smallest distance: {min_distance}")
    if min_distance != distances[control_answer]:

        # lets see if its adjacent to the answer - yes almost always
        shares_vertex = len(np.intersect1d(simplices_np[control_answer] , simplices_np[nearest_simplex])) > 0
        # print(f'shares vertex with correct answer: {shares_vertex}')

        # lets see if the faces that share a vertex have a vertex shared with the answer - no
        if shares_vertex is False:
            print(f'doesnt share vertex with correct answer!')

            for simplex in neighbors_np[nearest_simplex]:
                shares_vertex = len(np.intersect1d(simplices_np[control_answer] , simplices_np[simplex])) > 0
                if shares_vertex is True:
                    print(f"found a mutual face sharing vertex!")
                    
                else:
                    print("FAILURE no shared faces found.")
                    

        # lets look at the numerical precision - not the problem
        python_version(random_start, debug=False) # 1e-3 certainty of result.

        # lets look to see if neighbors neighbors (maybe further) are closer to the point
        distances_1m = []
        distances_2m = []
        for neighbor_1m in neighbors_np[nearest_simplex]:
            distances_1m.append(np.linalg.norm(centroids_np[neighbor_1m]-dummy_point, axis=1, ord=2))
            for neighbor_2m in neighbors_np[neighbor_1m]:
                distances_2m.append(np.linalg.norm(centroids_np[neighbor_2m]-dummy_point, axis=1, ord=2))


        print('fin')

print('fin')