import numpy as np
import ctypes
import subprocess
import time

process = subprocess.run("gcc -shared -o geometry.so geometry.c", shell=True)
time.sleep(0.2)

# Load the shared library
lib = ctypes.CDLL('./geometry.so')

# replicate the data structures in the C so that we can manipulate them using ctypes
# ----------------------------------------------------------------------------------

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

# Lets get REAL data to test this 
# -------------------------------

from barrier_function import BarrierFunction
import torch
bf = BarrierFunction(torch.load('large_data/nav_training_data.pt'))

points_np = bf.cvx_hull.points
dimension = points_np.shape[1]
simplices_np = bf.cvx_hull.simplices
neighbors_np = bf.cvx_hull.neighbors

# Create the C datatypes and instantiate the ConvexHull
# -----------------------------------------------------

# Assuming you've loaded your shared library as 'lib'
lib.InstantiateConvexHull.argtypes = [
    ctypes.POINTER(ConvexHull),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),  # Optional for neighbors
    ctypes.c_int
]

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

# Call the InstantiateConvexHull function
lib.InstantiateConvexHull(
    ctypes.byref(hull),
    points_c, len(points_np),
    simplices_c, len(simplices_np) // dimension,  # Adjust based on how you count simplices
    neighbors_c,  # You can pass None if not using neighbors
    dimension
)

# 
# --------------------------

print('fin')