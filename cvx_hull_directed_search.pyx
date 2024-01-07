import numpy as np
cimport numpy as cnp  # This syntax is used for Cython's numpy
from libc.stdlib cimport malloc, free

def search_convex_hull(hull, int simplex_idx, int search_step_depth, outlier_point):
    cdef int i, min_idx
    cdef float min_distance
    cdef list ordered_neighbors = []
    cdef set previously_visited = set([simplex_idx])
    cdef list degree_lengths
    cdef int[:] neighbors
    cdef float[:, :] centroids = hull.points[hull.simplices].mean(axis=1)  # Assuming this structure for centroids

    # Perform the search
    for i in range(search_step_depth):
        simplex_idx = hull.neighbors[simplex_idx]
        simplex_idx = np.setdiff1d(simplex_idx, list(previously_visited))  # filter out previously visited
        ordered_neighbors.append(simplex_idx)
        previously_visited.update(simplex_idx)

    degree_lengths = [len(i) for i in ordered_neighbors]
    neighbors = np.hstack([simplex_idx, np.hstack(ordered_neighbors)])
    distances = np.linalg.norm(centroids[neighbors] - outlier_point, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # Check if the minimum distance is on the interior of the search space
    if min_idx - 1 > degree_lengths[-2]:
        # Point on exterior or outside of search space
        return min_distance
    else:
        # Found the closest point
        return np.array([neighbors[min_idx]])

