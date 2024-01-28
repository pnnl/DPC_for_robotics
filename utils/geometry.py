import numpy as np
import scipy as sp

from utils.time import time_function

# @time_function
def find_closest_simplex_equation(
        hull,
        point,
        SV = 1e-10,         # singular value threshold
        detT = 1e-10,       # determinant threshold
    ):
    guess = np.arange(hull.points.shape[1])
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
