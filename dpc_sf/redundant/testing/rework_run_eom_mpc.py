from quad_refactor import Quadcopter
import utils.pytorch_utils as ptu
import torch
import numpy as np
from trajectory_rework import waypoint_reference, equation_reference
from mpc_refactor import MPC_Point_Ref_Obstacle, MPC_Point_Ref, MPC_Traj_Ref
from utils.animation_rework import Animator

dt = 0.1
Ti = 0
Tf = 15
reference_type = 'fig8' # 'fig8', 'wp_traj', 'wp_p2p'
backend = 'eom'

quad = Quadcopter()

state = ptu.from_numpy(np.array([
    0,                  # x
    0,                  # y
    0,                  # z
    1,                  # q0
    0,                  # q1
    0,                  # q2
    0,                  # q3
    0,                  # xdot
    0,                  # ydot
    0,                  # zdot
    0,                  # p
    0,                  # q
    0,                  # r
    522.9847140714692,  # wM1
    522.9847140714692,  # wM2
    522.9847140714692,  # wM3
    522.9847140714692   # wM4
]))

# setup trajectory
if reference_type == 'wp_p2p' or reference_type == 'wp_traj':
    reference = waypoint_reference(type=reference_type, average_vel=1)
elif reference_type == 'fig8':
    reference = equation_reference(type=reference_type, average_vel=1)

# setup mpc
ctrlHzn = 30
interaction_interval=1

if reference_type == 'wp_traj' or reference_type == 'fig8':
    ctrl = MPC_Traj_Ref(
        N=ctrlHzn,
        dt=dt,
        interaction_interval=interaction_interval, 
        n=17, 
        m=4, 
        dynamics=quad.casadi_state_dot,
        state_ub=quad.params["ca_state_ub"],
        state_lb=quad.params["ca_state_lb"],
        reference_traj=reference
    )
elif reference_type == 'wp_p2p':
    ctrl = MPC_Point_Ref_Obstacle(
        N=ctrlHzn,
        dt=dt,
        interaction_interval=interaction_interval, 
        n=17, 
        m=4, 
        dynamics=quad.casadi_state_dot,
        state_ub=quad.params["ca_state_ub"],
        state_lb=quad.params["ca_state_lb"],
    )

state_history = []
time_history = []
reference_history = []

for idx, t in enumerate(np.arange(Ti, Tf, dt)):
    print(f'time is: {t}')
    # generate command based on current state
    if reference_type == 'wp_traj' or reference_type == 'fig8':
        # trajectory mpc contains the reference already, so it only needs state and time
        cmd = ctrl(state, t)
    elif reference_type == 'wp_p2p':
        cmd = ctrl(state, reference(t))

    # save the state and command
    state_history.append(state)
    time_history.append(t)
    reference_history.append(reference(t))

    # step the state
    state += quad.state_dot(state, cmd) * dt

    print(state - reference(t+dt))

# animator expects numpy arrays
state_history = ptu.to_numpy(torch.vstack(state_history))
time_history = np.array(time_history)
reference_history = np.vstack(reference_history)

np.savez('test.npz', state_history=state_history, time_history=time_history, reference_history=reference_history)

if reference_type == 'wp_p2p':
    drawCylinder = True
else:
    drawCylinder = False

animator = Animator(
    states=state_history, 
    times=time_history, 
    reference_history=reference_history, 
    reference=reference, 
    reference_type=reference_type, 
    drawCylinder=drawCylinder
)
animator.animate() # contains the plt.show()
