"""
The aim of this file is to formally create a function for the vertex approach which can then be copied
into the main code

result:

While this seems to be correct as a function, testing shows that the numpy argsort remains the 
bottleneck - indicating the directed search will still be useful in the future. That being said
I am leaving that alone for now.
"""
import cProfile
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt

from utils.time import time_function
from safe_set import SafeSet

log = torch.load('large_data/nav_training_data.pt')
ss = SafeSet(log)

point = np.array([-0.58833, 0.78767, 0.96962, -0.056039, -0.22199, 0.2492])
hull = ss.cvx_hull
dim = len(point)

@time_function
def find_closest_simplex_equation(
        hull,
        point,
        guess=np.arange(dim), # first 6 points are first guess
        SV = 1e-10,         # singular value threshold
        detT = 1e-10,       # determinant threshold
    ):
    distances = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)
    argmins = np.argsort(distances) # the slowest section still - 0.016s to 0.054s (90% of time)
    # first guess of first dim points
    A = hull.points[hull.vertices][argmins][guess]
    while np.linalg.det(A) < detT:      # check for colinearity through volume of matrix (determinant)
        U, S, VT = np.linalg.svd(A)
        colinear_points = np.where(S < SV)[0]
        if len(colinear_points) == 0:
            # found the correct A
            break
        # add new points from argsorted points
        new_points = np.arange(guess[-1] + 1, guess[-1] + len(colinear_points) + 1)
        # remove colinear points
        guess = np.hstack([guess, new_points])
        guess = np.delete(guess, colinear_points)
        A = hull.points[hull.vertices][argmins][guess]
    A = np.pad(A, ((0, 0), (0, 1)), constant_values=1) # pad with ones and zeros to account for constant offset
    A = np.pad(A, ((0, 1), (0, 0)), constant_values=0) # pad with ones and zeros to account for constant offset
    eqn = sp.linalg.null_space(A).flatten() # needs to be a 1D nullspace!
    return eqn

def test():
    eqn = find_closest_simplex_equation(hull, point)

if __name__ == "__main__":
    cProfile.run('test()')




print('fin')