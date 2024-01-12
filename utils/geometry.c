// The function we are trying to recreate in C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> // For DBL_MAX
#include <stdbool.h> // for the boolean checks

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
    //Vector* neighborVectors;     // Direction to each neighbor (size will be N+1)
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

    printf("pointer to centroid_data[0]: %p\n", &centroid_data);

    printf("C code centroid_data: ");
    for (int i = 3079679-1 + 100; i < 3079679+2 + 100; i++){
        printf("%f\n", centroid_data[i]);
    }

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

        hull->simplices[i].vertexIndices = (int*)malloc(dimension * sizeof(int));
        if (hull->simplices[i].vertexIndices == NULL) {
            printf("Memory allocation failed for vertexIndices at simplex %d\n", i);
            return; // or handle the error as needed
        }

        hull->simplices[i].numVertices = dimension;

        for (int j = 0; j < dimension; j++) {
            int index = i * dimension + j;
            if (index >= num_simplices * dimension) {
                printf("Index out of bounds for simplices_array at simplex %d, dimension %d\n", i, j);
                return; // or handle the error as needed
            }
            hull->simplices[i].vertexIndices[j] = simplices_array[index];
        }

        //printf("Processed vertexIndices for simplex index: %d\n", i);

        // If neighbors information is provided
        if (neighbors_array != NULL) {
            hull->simplices[i].neighborIndices = (int*)malloc(dimension * sizeof(int));
            if (hull->simplices[i].neighborIndices == NULL) {
                printf("Memory allocation failed for neighborIndices at simplex %d\n", i);
                return; // or handle the error as needed
            }
            if (i == 3079679) {
                printf("here 4 \n");
            }

            for (int j = 0; j < dimension; j++) {
                int index = i * dimension + j;
                if (index >= num_simplices * dimension) {
                    printf("Index out of bounds for neighbors_array at simplex %d, dimension %d\n", i, j);
                    return; // or handle the error as needed
                }
                hull->simplices[i].neighborIndices[j] = neighbors_array[index];
            }
            if (i == 3079679) {
                printf("here 5 \n");
            }
        }

        // Initialize the centroid of the simplex
        hull->simplices[i].centroid = (Point*)malloc(sizeof(Point));
        if (hull->simplices[i].centroid == NULL) {
            printf("Memory allocation failed for centroid at simplex %d\n", i);
            return; // or handle the error as needed
        }
        if (i == 3079679) {
            printf("here 6 \n");
        }

        hull->simplices[i].centroid->coordinates = (double*)malloc(dimension * sizeof(double));
        if (hull->simplices[i].centroid->coordinates == NULL) {
            printf("Memory allocation failed for centroid coordinates at simplex %d\n", i);
            return; // or handle the error as needed
        }

        if (i == 3079679) {
            printf("here 7 \n");
        }

        for (int j = 0; j < dimension; j++) {
            int index = i * dimension + j;
            if (i == 3079679) {
                printf("REAL index: %d\n", index);
                printf("&centroid_data[0]: %p\n", &centroid_data[0]);
                printf("&centroid_data[index]: %p\n", &centroid_data[index]);
                printf("centroid_data[index]: %f\n", centroid_data[index]);
            }

            if (index >= num_simplices * dimension) {
                printf("Index out of bounds for centroid_data at simplex %d, dimension %d\n", i, j);
                return; // or handle the error as needed
            }
            hull->simplices[i].centroid->coordinates[j] = centroid_data[index];
        }

        if (i == 3079679) {
            printf("here 8 \n");
        }
        
        // printf("Processed centroid for simplex index: %d\n", i);
    }    

    hull->numSimplices = num_simplices;

    // Step 3: Set other properties of the convex hull
    hull->dimension = dimension;

    // Step 4: Handle errors and edge cases
}

// ================== //
// Utility Functions  //
// ================== //

// Function to check if an integer is in an array
bool is_in_array(int value, int* array, int size) {
    for (int i = 0; i < size; ++i) {
        if (array[i] == value) {
            return true;
        }
    }
    return false;
}

// Function to add unique elements to an array
int add_unique(int* array, int value, int size) {
    if (!is_in_array(value, array, size)) {
        array[size] = value;
        return 1; // Added new element
    }
    return 0; // Element was already in the array
}

// Function to calculate Euclidean distance between two points in N-dimensional space
double EuclideanDistance(Point* p1, Point* p2, int dimension) {
    double distance = 0.0;
    for (int i = 0; i < dimension; i++) {
        double diff = p1->coordinates[i] - p2->coordinates[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

// ====================== //
// The Meat and Potatoes //
// ====================== //

// The sixth-degree search function
int SixthDegreeSearch(ConvexHull* hull, int candidate, Point* dummy_point) {
    double d_ctrl = EuclideanDistance(hull->simplices[candidate].centroid, dummy_point, hull->dimension);
    int MAX_NEIGHBORS = 93750; // 6 * 5 ^ 6
    int n6[MAX_NEIGHBORS]; // Assuming a maximum number of neighbors
    int n6_size = 0;

    // Iterating through neighbors to the sixth degree
    for (int i = 0; i < hull->simplices[candidate].numVertices; ++i) {
        int n1 = hull->simplices[candidate].neighborIndices[i];
        for (int j = 0; j < hull->simplices[n1].numVertices; ++j) {
            int n2 = hull->simplices[n1].neighborIndices[j];
            for (int k = 0; k < hull->simplices[n2].numVertices; ++k) {
                int n3 = hull->simplices[n2].neighborIndices[k];
                for (int l = 0; l < hull->simplices[n3].numVertices; ++l) {
                    int n4 = hull->simplices[n3].neighborIndices[l];
                    for (int m = 0; m < hull->simplices[n4].numVertices; ++m) {
                        int n5 = hull->simplices[n4].neighborIndices[m];
                        n6_size += add_unique(n6, hull->simplices[n5].neighborIndices[k], n6_size);
                    }
                }
            }
        }
    }

    // Filter out the candidate
    int n6m[MAX_NEIGHBORS]; // Filtered list
    int n6m_size = 0;
    for (int i = 0; i < n6_size; ++i) {
        if (n6[i] != candidate) {
            n6m[n6m_size++] = n6[i];
        }
    }

    // Find the closest centroid among neighbors
    double min_distance = DBL_MAX;
    int closest_centroid = -1;
    for (int i = 0; i < n6m_size; ++i) {
        double distance = EuclideanDistance(hull->simplices[n6m[i]].centroid, dummy_point, hull->dimension);
        if (distance < min_distance) {
            min_distance = distance;
            closest_centroid = n6m[i];
        }
    }

    // Compare and return the result
    if (d_ctrl <= min_distance) {
        return candidate;
    } else {
        return closest_centroid;
    }
}


// Function to find the index of the nearest simplex to a given point
void FindNearestSimplexBruteForce(ConvexHull* hull, Point* p) {
    if (hull == NULL || p == NULL) {
        printf("ConvexHull or Point is NULL\n");
        return;
    }

    double minDistance = DBL_MAX;
    int nearestSimplexIndex = -1;

    for (int i = 0; i < hull->numSimplices; i++) {
        double distance = EuclideanDistance(p, hull->simplices[i].centroid, hull->dimension);
        if (distance < minDistance) {
            minDistance = distance;
            nearestSimplexIndex = i;
        }
    }

    // Output the index of the nearest simplex
    if (nearestSimplexIndex != -1) {
        printf("Nearest Simplex Index: %d\n", nearestSimplexIndex);
    } else {
        printf("No nearest simplex found\n");
    }
}

// FindNearestSimplexDirected:
// guess is the initial guess for the simplex idx that is closest to the point
// 1. calculate distance from point to centroid of guess and all of its neighbors centroids
// 2. find the minimum distance of those calculated
// 3. if minimum distance is the same as guess, return guess idx
// 4. if minimum distance is a neighbor of guess then start while loop:
//      5. we replace guess with the neighbor that is the minimum
//      6. calculate distance to the new guess's centroid and all of its neighbors bar the original guess
//      7. if minimum distance is the same as the new guess, return guess idx
//      8. repeat loop until a guess is a minimum amongst its neighbors

void FindNearestSimplexDirected(ConvexHull* hull, int guess, Point* p) {

    int current_guess = guess;
    double min_distance;
    int nearest_simplex = -1;
    int dimension = hull->dimension;
    int found = 0;

    while (!found) {
        min_distance = EuclideanDistance(p, hull->simplices[current_guess].centroid, dimension);
        nearest_simplex = current_guess;

        // Check neighbors
        // printf("%d \n",hull->simplices[current_guess].numVertices);
        for (int i = 0; i < hull->simplices[current_guess].numVertices; i++) {

            // printf("current_guess: %d\n", current_guess);
            int neighbor_idx = hull->simplices[current_guess].neighborIndices[i];
             //printf("neighbor_idx: %d\n", neighbor_idx);

            double distance = EuclideanDistance(p, hull->simplices[neighbor_idx].centroid, dimension);
            
            // printf("min_distance: %f\n", min_distance);
            // printf("distance: %f\n", distance);

            if (distance < min_distance) {

                
                min_distance = distance;
                nearest_simplex = neighbor_idx;
            }

            // printf("new_min_distance: %f\n", min_distance);
            // printf("current nearest simplex: %d\n", nearest_simplex);
        }

        // Check if the current guess is the nearest
        if (nearest_simplex == current_guess) {
            // printf("checking 6th degree!\n");
            int checked_guess = SixthDegreeSearch(hull, current_guess, p);
            if (nearest_simplex == checked_guess) {
                found = 1;
            }
            else {
                current_guess = checked_guess;
            }            
        } else {
            // Update the guess for the next iteration
            current_guess = nearest_simplex;
            // printf("current guess: %d\n", nearest_simplex);
            
        }
    }

    printf("Nearest Simplex Index: %d\n", nearest_simplex);
    printf("new_min_distance: %f\n", min_distance);

}
