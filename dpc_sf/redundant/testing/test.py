import numpy as np
from utils.animation_rework import Animator


file = np.load('test.npz')
states, times, references = file['state_history'], file['time_history'], file['reference_history']

from trajectory_rework import waypoint_reference, equation_reference

reference = waypoint_reference(type='wp_traj', average_vel=1)

animator = Animator(
    states=states, 
    times=times, 
    reference_history=references, 
    reference=reference, 
    reference_type='wp_traj', 
    drawCylinder=False
)
animator.animate() # contains the plt.show()
