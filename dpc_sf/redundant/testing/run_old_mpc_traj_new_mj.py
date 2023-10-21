from env import Sim
from mpc import MPC_Point_Ref_Obstacle
from trajectory import waypoint_reference
from quad import Quadcopter
import utils.pytorch_utils as ptu
import matplotlib.pyplot as plt
import torch
import numpy as np

state = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,522.9847140714692,522.9847140714692,522.9847140714692,522.9847140714692])
# state[13:] *= 0 # zero thrust to fall down

reference = waypoint_reference(type='wp_p2p', average_vel=0.5)
quad = Quadcopter()
ctrl2 = MPC_Point_Ref_Obstacle(N=300,dt=0.01,interaction_interval=1, n=17, m=4, dynamics=quad.casadi_state_dot,state_ub=quad.params["ca_state_ub"],state_lb=quad.params["ca_state_lb"],return_type='numpy')

mj_env = Sim(dt=0.01,Ti=0,Tf=4,params=quad.params,backend='mj',init_state=state,reference=reference,state_dot=quad.state_dot,xml_path="mujoco/quadrotor_x.xml",write_path="media/mujoco/",)

from redundant.testing.quad import sys_params, Quadcopter
from redundant.testing.mpc import MPC_Point_Ref_Obstacle
from redundant.testing.trajectory import Trajectory
import utils

quad = Quadcopter()
traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
# ctrl = MPC_Point_Ref_Obstacle(N=30, sim_Ts=0.1, interaction_interval=1, n=17, m=4, quad=quad)

# need to call this at time = 0 to instantiate some attributes
sDes = traj.desiredState(0, 0.1, quad)

while mj_env.t < mj_env.Tf:

    state = mj_env.get_state()

    # define reference
    sDes = traj.desiredState(mj_env.t+5, 0.1, quad)     
    # reference = utils.sDes2state(sDes) 

    # cmd = ctrl(state.tolist(), reference(mj_env.t).tolist()).value(ctrl.U)[:,0]
    cmd = ctrl2(state, reference(mj_env.t))
    print(state - reference(mj_env.t))

    mj_env.step(cmd)

mj_env.animate()


mj_history = np.stack(mj_env.state_history) # shape [40, 17]

