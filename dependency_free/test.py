import casadi as ca
import numpy as np
import torch
from scipy.spatial import ConvexHull

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
    x = ca.mtimes(coeffs, hull_vertices)

    # Objective: Minimize the squared Euclidean distance between x and the external point
    objective = ca.sumsqr(x - external_point)

    # Constraints: Coefficients must be non-negative and sum to 1
    constraints = [coeffs >= 0, ca.sum1(coeffs) == 1]

    # Formulate the optimization problem
    nlp = {'x': coeffs, 'f': objective, 'g': ca.vertcat(*constraints)}
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

cvx_safe_points = preprocess_data(
    data=torch.load('large_data/nav_training_data.pt'), 
    num_points_considered=100_000, 
    cylinder_radius=0.5
)

# Example usage
hull = ConvexHull(cvx_safe_points)  
hull_vertices = hull.points[hull.vertices]  # The vertices of the convex hull
external_point = np.array([-1.,0,1,0,0,0])  # The external point
closest_point, gradient = find_closest_point_and_gradient(hull_vertices, external_point)


