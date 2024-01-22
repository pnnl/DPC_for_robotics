"""
Testing to see if simply doing a cvx opt is tractable here... it was not
even the directly interfaced OSQP takes a min of 0.6s per search even with
warm starting
"""

import casadi as ca
import numpy as np
import torch
from scipy.spatial import ConvexHull
import time

# for the direct interface
import osqp
import scipy.sparse as sp


def preprocess_data(data, num_points_considered, cylinder_radius):
    # extract all states, inputs, losses from data
    x = torch.stack(data['x'])
    u = torch.stack(data['u'])
    loss = torch.stack(data['loss'])
    c = torch.vstack([torch.tensor([1.,1.])] * x.shape[2])

    log = {'success': [], 'failure': [], 'success_inputs': [], 'failure_inputs': [], 'minibatch_loss': [], 'minibatch_number': []}
    for i, (minibatch, minibatch_inputs) in enumerate(zip(x, u)):
        for trajectory, inputs in zip(minibatch, minibatch_inputs):
            # state = {x, xdot, y, ydot, z, zdot}
            xy = trajectory[:,0:4:2]

            # we must first rule out all trajectories that under DPC control intersected
            # the cylinder constraint
            distances = torch.norm(xy - c, dim=1)
            if (distances < cylinder_radius).any() == True:
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

    return cvx_safe_points

def find_closest_point_and_gradient(hull_vertices, external_point):
    # Number of vertices and dimension
    num_vertices = hull_vertices.shape[0]
    dim = hull_vertices.shape[1]

    # Define the optimization variables (coefficients of the convex combination)
    coeffs = ca.MX.sym('coeffs', num_vertices)

    # The point x inside the hull as a convex combination of the vertices
    x = hull_vertices.T @ coeffs

    # Objective: Minimize the squared Euclidean distance between x and the external point
    objective = ca.sumsqr(x - external_point)

    # Constraints: Coefficients must be non-negative and sum to 1
    constraints = [coeffs >= 0, ca.sum1(coeffs) == 1]

    # Formulate the optimization problem
    nlp = {'x': coeffs, 'f': objective, 'g': ca.vertcat(*constraints)}
    print('creating the solver obj')
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Solve the problem
    solution = solver(lbg=[0, 1], ubg=[np.inf, 1])
    optimal_coeffs = solution['x']

    # The closest point in the hull
    closest_point = np.dot(optimal_coeffs, hull_vertices)

    # Compute the gradient at the closest point
    gradient = ca.gradient(objective, coeffs)
    gradient_at_closest = ca.Function('grad', [coeffs], [gradient])(optimal_coeffs)

    return closest_point.full().flatten(), gradient_at_closest.full().flatten()

# based on solving for the optimal convex combination of the vertices that satisfies the criteria
# we can also do this by using every simplex face as a halfspace constraint - but this scales better
class ClosestSimplexSolver:
    def __init__(self, hull_vertices) -> None:

        # Number of vertices and dimension
        num_vertices = hull_vertices.shape[0]
        dim = hull_vertices.shape[1]

        # Define the optimization variables (coefficients of the convex combination)
        self.opti = ca.Opti()
        self.C = self.opti.variable(num_vertices)
        self.P = self.opti.parameter(dim)

        # The point x inside the hull as a convex combination of the vertices
        x = hull_vertices.T @ self.C

        # Objective: Minimize the squared Euclidean distance between x and the external point
        objective = ca.sumsqr(x - self.P)
        self.opti.minimize(objective)

        # Constraints: Coefficients must be non-negative and sum to 1 for convex combination
        self.opti.subject_to(self.C >= 0)
        self.opti.subject_to(ca.sum1(self.C) == 1)

        # Solver: this is a QP, ipopt is just easy lol
        # Set the QP solver to use the 'conic' plugin
        opts = {
            "qpsol": "osqp",
            "qpsol_options": {
                "verbose": True,
                "osqp": {
                    "max_iter": 10000  # Specify the maximum number of iterations here
                }
            }
        }
        # opts = {"qpsol": "osqp", "qpsol_options": {"verbose": True}}
        self.opti.solver("sqpmethod", opts)

    def __call__(self, external_point):

        # assert external point not in hull

        # assign static parameters
        self.opti.set_value(self.P, external_point)

        print('starting sol')
        sol = self.opti.solve()

        print('fin')


class DirectInterfaceApproach:
    def __init__(self, V) -> None:
        """
        V: convex hull vertices
        """
        num_vertices, dim = V.shape[0], V.shape[1]
        self.V = V
        
        P = self.V @ self.V.T
        # Convert P to a sparse CSC matrix
        P = sp.csc_matrix(sp.triu(P))
 
        dummy_p = np.zeros([dim,1])
        q = -2 * self.V @ dummy_p
        
        A1 = np.eye(num_vertices)
        A2 = np.ones(num_vertices)
        A = sp.csc_matrix(np.vstack([A1,A2]))
        
        l1 = np.zeros([num_vertices,1])
        l2 = np.array([1])
        l = np.vstack([l1,l2])
        
        u1 = np.ones([num_vertices, 1]) * np.inf
        u2 = np.array([1])
        u = np.vstack([u1,u2])

        self.m = osqp.OSQP()
        self.m.setup(P=P, q=q, A=A, l=l, u=u)

        print("performing test solve...")
        results = self.m.solve()
        print("initialisation complete.")


    def __call__(self, p):
        print("starting first call")
        start = time.time()
        q = -2 * self.V @ p
        self.m.update(q=q)
        sol = self.m.solve()
        end = time.time()
        print(f"time taken: {end - start}")
        print('fin')

 



cvx_safe_points = preprocess_data(
    data=torch.load('large_data/nav_training_data.pt'), 
    num_points_considered=100_000, 
    cylinder_radius=0.5
)
# 
# # Example usage
hull = ConvexHull(cvx_safe_points)  
hull_vertices = hull.points[hull.vertices]  # The vertices of the convex hull
external_point = np.array([-1.,0,1,0,0,0])  # The external point

# this is not working in 6D, lets try a toy 2D dataset
# this 2D example works.
# hull_vertices = np.array([
#     [0.,0],
#     [1,0],
#     [1,1],
#     [0,1]
# ])

# Number of vertices
# num_vertices = 7000

# Generate points around a circle
# theta = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
# hull_vertices = np.column_stack([np.cos(theta), np.sin(theta)])
# external_point = np.array([2.0,2.0])

solver = DirectInterfaceApproach(hull_vertices)
test = solver(external_point)
test = solver(external_point)
test = solver(external_point)
test = solver(external_point)
test = solver(external_point)


# solver = ClosestSimplexSolver(hull_vertices)
# test = solver(external_point)

# closest_point, gradient = find_closest_point_and_gradient(hull_vertices, external_point)

print('fin')

