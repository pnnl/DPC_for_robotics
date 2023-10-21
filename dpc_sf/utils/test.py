import numpy as np

states, times, references = np.load('test.npz')

from trajectory_rework import waypoint_reference, equation_reference

reference = waypoint_reference(type=reference_type, average_vel=1)


