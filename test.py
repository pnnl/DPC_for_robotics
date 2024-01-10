import numpy as np
import ctypes
import subprocess
import time
from copy import deepcopy

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
    dummy_point = np.array([1.,1.])
elif hull_type == 'cvx':
    dummy_point = np.array([1.,1.,1.,1.,1.,1.])


# plot the convex hull itself
plt.close()
plt.plot(np.hstack([vertices_np[:,0], vertices_np[0,0]]), np.hstack([vertices_np[:,1], vertices_np[0,1]]), linestyle='-', marker='o', color='grey')
# plot the centroids of the simplices
plt.scatter(centroids_np[:,0], centroids_np[:,1], marker='o', color='r')
# plot dummy point traversal
plt.plot(np.hstack([dummy_point[0]]), np.hstack([dummy_point[1]]), linestyle='-', marker='o', color='b')

plt.savefig('test.png')


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

print(f"Python centroid_data: {centroids_flat[index-1+100:index+2+100]}")


print("specific index pointer:")
# print(centroids_c[index:index+1]._arr.data)
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
lib.FindNearestSimplexDirected(ctypes.byref(hull), simplex_idx, ctypes.byref(dummy_point_struct))

simplex_idx = 6000
lib.FindNearestSimplexDirected(ctypes.byref(hull), simplex_idx, ctypes.byref(dummy_point_struct))



# 
# --------------------------

print('fin')