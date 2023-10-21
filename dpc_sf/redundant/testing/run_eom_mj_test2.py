# this script will directly compare the mj and eom environments in the new code

from env import Sim
from mpc import MPC_Point_Ref
from control.trajectory import waypoint_reference
from quad import Quadcopter
import utils.pytorch_utils as ptu
import matplotlib.pyplot as plt
import torch
import numpy as np

Tf = 1
integrator = 'RK4'

state = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,522.9847140714692,522.9847140714692,522.9847140714692,522.9847140714692])
state[13:] *= 0 # zero thrust to fall down

reference = waypoint_reference(type='wp_p2p', average_vel=0.5)
quad = Quadcopter()
# ctrl = MPC_Point_Ref(N=30,dt=0.1,interaction_interval=1, n=17, m=4, dynamics=quad.casadi_state_dot,state_ub=quad.params["ca_state_ub"],state_lb=quad.params["ca_state_lb"],return_type='torch',integrator_type=integrator)

eom_env = Sim(dt=0.1,Ti=0,Tf=Tf,params=quad.params,backend='eom',init_state=ptu.from_numpy(state),reference=reference,state_dot=quad.state_dot,xml_path="mujoco/quadrotor_x.xml",write_path="media/mujoco/",integrator_type=integrator)
mj_env = Sim(dt=0.1,Ti=0,Tf=Tf,params=quad.params,backend='mj',init_state=ptu.to_numpy(ptu.from_numpy(state)),reference=reference,state_dot=quad.state_dot,xml_path="mujoco/quadrotor_x.xml",write_path="media/mujoco/",integrator_type=integrator)

eom_cmd = torch.zeros(4)
mj_cmd = np.zeros(4)

def cmd(t):
    cmd = np.array([0.0001,0,0,0.0001])*np.sin(t)
    cmd = np.array([0,0,0,0])
    return cmd

def state_delta():
    return ptu.to_numpy(eom_env.state) - mj_env.state

while eom_env.t < eom_env.Tf:
    eom_env.step(ptu.from_numpy(cmd(eom_env.t)))
    mj_env.step(cmd(mj_env.t))
    assert np.abs(eom_env.t - mj_env.t).max() < 1e-08
    

eom_history = torch.stack(eom_env.state_history) # shape [40,17]
mj_history = np.stack(mj_env.state_history) # shape [40, 17]

# Create a figure and subplots
fig, axs = plt.subplots(7, 3, figsize=(10, 8))

# Flatten the axs array to iterate over subplots
axs_flat = axs.flatten()

# Iterate through each state history
# for i in range(17):
#     axs_flat[i].plot(eom_history[:, i], label='EOM History')
#     axs_flat[i].plot(mj_history[:, i], label='MJ History')
#     axs_flat[i].set_ylabel(f'State {i+1}')
#     axs_flat[i].legend()

delta_array = ptu.to_numpy(eom_history)-mj_history
for i in range(17):
    axs_flat[i].plot(delta_array[:, i], label='delta')
    axs_flat[i].set_ylabel(f'State {i+1}')
    axs_flat[i].legend()

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

eom_env.animate()
mj_env.animate()

# ---------------------------------------------------------
# eom_history = torch.stack(eom_env.state_history)[:,13:17]
# mj_history = np.stack(mj_env.state_history)[:,13:17]
# 
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# 
# axs[0, 0].plot(eom_history[:,0], label='eom')
# axs[0, 0].plot(mj_history[:,0], label='mj')
# axs[0, 0].set_title('omega 1')
# axs[0, 0].legend()
# 
# axs[0, 1].plot(eom_history[:,1], label='eom')
# axs[0, 1].plot(mj_history[:,1], label='mj')
# axs[0, 1].set_title('omega 1')
# axs[0, 1].legend()
# 
# axs[1, 0].plot(eom_history[:,2], label='eom')
# axs[1, 0].plot(mj_history[:,2], label='mj')
# axs[1, 0].set_title('omega 1')
# axs[1, 0].legend()
# 
# axs[1, 1].plot(eom_history[:,3], label='eom')
# axs[1, 1].plot(mj_history[:,3], label='mj')
# axs[1, 1].set_title('omega 1')
# axs[1, 1].legend()
# 
# # Adjust spacing between subplots
# plt.tight_layout()
# 
# # Display the plot
# plt.show()
# ---------------------------------------------------------


print('fin')