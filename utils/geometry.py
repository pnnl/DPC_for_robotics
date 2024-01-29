import numpy as np
import scipy as sp

from utils.time import time_function

def find_angle(vec1, vec2):
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) # normalised
    return np.arccos(cosine_angle)

# @time_function
def find_closest_simplex_equation(
        hull,
        point,
        SV = 1e-10,         # singular value threshold
        detT = 1e-10,       # determinant threshold
    ):
    guess=np.arange(hull.points.shape[1]) # first N points are first guess
    distances = np.linalg.norm(hull.points[hull.vertices] - point, axis=1)
    argmins = np.argsort(distances) # the slowest section still - 0.016s to 0.054s (90% of time)
    # first guess of first dim points
    nearest_vertices = hull.points[hull.vertices][argmins][guess]
    while np.linalg.det(nearest_vertices) < detT:      # check for colinearity through volume of matrix (determinant)
        U, S, VT = np.linalg.svd(nearest_vertices)
        colinear_points = np.where(S < SV)[0]
        if len(colinear_points) == 0:
            # found the correct A
            break
        # add new points from argsorted points

        new_points = np.arange(guess[-1] + 1, guess[-1] + len(colinear_points) + 1)
        # remove colinear points
        guess = np.hstack([guess, new_points])
        guess = np.delete(guess, colinear_points)
        nearest_vertices = hull.points[hull.vertices][argmins][guess]
    A = np.pad(nearest_vertices, ((0, 0), (0, 1)), constant_values=1) # pad with ones and zeros to account for constant offset
    A = np.pad(A, ((0, 1), (0, 0)), constant_values=0) # pad with ones and zeros to account for constant offset
    eqn = sp.linalg.null_space(A).flatten() # needs to be a 1D nullspace!
    # need to ensure the equation normal is facing the right way. use the angle between centroid -> random vertex and normal >90 constraint
    # and if it fails check, just flip it. eqn *= -1. We also only want to consider a vertex we havent yet looked at to avoid eroneous 
    # results, lets use the furthest vertex from our point to keep things simple argmins[-1]
    furthest_vertex = hull.points[hull.vertices][argmins][-1]
    simplex_centroid = np.mean(nearest_vertices, axis=0)
    centroid_to_furthest_vertex = furthest_vertex - simplex_centroid
    simplex_normal = eqn[:-1]
    angle = find_angle(vec1=centroid_to_furthest_vertex, vec2=simplex_normal)
    if angle < np.pi/2:
        eqn *= -1 # inclusive of the constant offset term
        print("flipping equation normal direction")

    return eqn

