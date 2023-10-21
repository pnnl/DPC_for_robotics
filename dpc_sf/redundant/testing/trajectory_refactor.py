import numpy as np

# the 5 waypoints from the original code
waypoints = np.array([
    [ 0.,  0.,  0.],
    [ 2.,  2.,  1.],
    [-2.,  3., -3.],
    [-2., -1., -3.],
    [ 3., -2.,  1.]
])

# time between each point
times = np.array([3., 3., 3., 3.])

# need the distances in order to get velocities
waypoint_deltas = np.diff(waypoints, axis=0)

# velocities
velocities = np.zeros(len(times))
for i in range(len(velocities)):
    velocities[i] = np.linalg.norm(np.diff(waypoints, axis=0)[i,:])/times[i]

# create interpolation function between waypoints
start = waypoints[0,:]
end = waypoints[1,:]
velocity = velocities[0]
time = 1.5 # directly in the middle
def interp(start, end, velocity, time):
    t = velocity * time / (end-start)
    assert 0 <= t <= 1, "Interpolation parameter should be in [0, 1] range."
    return (1 - t) * start + t * end


interp(start, end, velocity, time)
