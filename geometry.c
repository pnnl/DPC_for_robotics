// The function we are trying to recreate in C

// def search_convex_hull(hull, simplex_idx, search_step_depth, outlier_point):
//     ordered_neighbors = []
//     previously_visited = set(simplex_idx)
//     complete = False
//     while complete is False:
// 
//         for _ in range(search_step_depth):
//             simplex_idx = hull.neighbors[simplex_idx]
//             simplex_idx = np.setdiff1d(simplex_idx, list(previously_visited))  # filter out previously visited
//             ordered_neighbors.append(simplex_idx)
//             previously_visited.update(simplex_idx)
// 
//         degree_lengths = [len(i) for i in ordered_neighbors]
//         neighbors = np.hstack([simplex_idx, np.hstack(ordered_neighbors)])
//         distances = np.linalg.norm(centroids[neighbors] - outlier_point, axis=1)  # Adjusted for points
//         min_idx = np.argmin(distances)
//         min_distance = distances[min_idx]
// 
//         if min_idx - 1 > degree_lengths[-2]:
//             complete = True
//             print(f'min_distance: {min_distance}')
//             return min_distance  # Point on exterior or outside of search space
//         else:
//             simplex_idx = neighbors[min_idx]
//             # return np.array([neighbors[min_idx]])  # Found the closest point

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// =============== //
// Data Structures //
// =============== //

// Define a point in N-dimensional space
typedef struct {
    double* coordinates;    // An array of coordinates (size will be N)
} Point;

// Define a vector in N-dimensional space
typedef struct {
    double* components;     // An array of components (size will be N)
} Vector;

// Define a simplex structure in N-dimensional space
typedef struct {
    int* vertexIndices;     // Indices of vertices that make up the simplex (size N for N-1 dimensional simplex)
    int numVertices;        // Always N for an N-1 dimensional simplex in N-dimensional space
    int* neighborIndices;   // Indices to neighboring simplices (size will be N+1 as well)
    Vector* neighborVectors;     // Direction to each neighbor (size will be N+1)
    Point* centroid;        // the centroid of the simplex
} Simplex;

// Define the convex hull structure for N-dimensional space
typedef struct {
    Point* points;          // Array of points in the convex hull
    int numPoints;          // Number of points
    Simplex* simplices;     // Array of simplices
    int numSimplices;       // Number of simplices
    int dimension;          // Dimensionality of the space
} ConvexHull;

// ======================= //
// Convex Hull Constructor //
// ======================= //

void InstantiateConvexHull(ConvexHull* hull, 
                           double* points_array, int num_points, 
                           int* simplices_array, int num_simplices,
                           int* neighbors_array, double* centroid_data,
                           int dimension) {
                            
    // Step 1: Initialize the points in the convex hull
    hull->points = (Point*)malloc(num_points * sizeof(Point));
    for (int i = 0; i < num_points; i++) {
        hull->points[i].coordinates = (double*)malloc(dimension * sizeof(double));
        for (int j = 0; j < dimension; j++) {
            hull->points[i].coordinates[j] = points_array[i * dimension + j];
        }
    }
    hull->numPoints = num_points;
    
    // Step 2: Initialize the simplices (faces) of the convex hull
    hull->simplices = (Simplex*)malloc(num_simplices * sizeof(Simplex));
    for (int i = 0; i < num_simplices; i++) {
        hull->simplices[i].vertexIndices = (int*)malloc((dimension) * sizeof(int));
        for (int j = 0; j < dimension; j++) {
            hull->simplices[i].vertexIndices[j] = simplices_array[i * dimension + j];
        }

        // If neighbors information is provided
        if (neighbors_array != NULL) {
            hull->simplices[i].neighborIndices = (int*)malloc((dimension) * sizeof(int));
            for (int j = 0; j < dimension; j++) {
                hull->simplices[i].neighborIndices[j] = neighbors_array[i * dimension + j];
            }
        }

        // Initialize the centroid of the simplex
        hull->simplices[i].centroid = (Point*)malloc(sizeof(Point));  // Allocate memory for the centroid structure
        hull->simplices[i].centroid->coordinates = (double*)malloc(dimension * sizeof(double));  // Allocate memory for coordinates

        // Copy the centroid coordinates from the precomputed centroid data
        for (int j = 0; j < dimension; j++) {
            hull->simplices[i].centroid->coordinates[j] = centroid_data[i * dimension + j];
        }
        
        // Initialize directions

    }
    hull->numSimplices = num_simplices;

    // Step 3: Set other properties of the convex hull
    hull->numSimplices = num_simplices;
    hull->dimension = dimension;

    // Step 4: Handle errors and edge cases
}




