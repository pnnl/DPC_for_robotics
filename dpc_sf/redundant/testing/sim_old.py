from quad import Quadcopter
from utils.windModel import Wind
from utils.stateConversions import sDes2state
from trajectory import Trajectory
# from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
import torch
import time
import numpy as np
import utils
import matplotlib.pyplot as plt

class Sim:
    def __init__(
            self, 
            Ts,
            Ti,
            Tf,
            quad, 
            reference = 'obstacle', # obstacle, trajectory, p2p
            backend = 'eom', # mj, eom
        ) -> None:        

        # Simulation Setup
        # --------------------------- 
        self.Ti = Ti
        self.Ts = Ts
        self.Tf = Tf # 50 seconds end of point2point

        # Initialize Quadcopter, Controller, Wind, Result Matrixes
        # ---------------------------
        self.quad = quad
        self.traj = Trajectory(self.quad, 'xyz_pos', np.array([13,3,0]))
        self.wind = Wind('None', 2.0, 90, -15)

        # Trajectory for First Desired States
        # ---------------------------
        if reference == 'obstacle' or 'p2p':
            self.traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
            # instantiate some attributes:
            self.traj.desiredState(0, Ts, quad)
        elif reference == 'trajectory':
            self.traj = Trajectory(quad, 'xyz_pos', np.array([2,3,1]))
            # instantiate some attributes:
            self.traj.desiredState(0, Ts, quad)
        else:
            raise Exception(f'invalid reference choice: {reference}')
        # sDes = self.traj.desiredState(0, self.Ts, self.quad)
        # self.reference = sDes2state(sDes)
        #self.cmd = self.ctrl(self.quad.state.tolist(), reference.tolist()).value(self.ctrl.U)[:,0]


        # Initialize Result Matrixes
        # ---------------------------
        numTimeStep = int(self.Tf/self.Ts+1)

        self.t_all          = np.zeros(numTimeStep)
        self.s_all          = np.zeros([numTimeStep, len(self.quad.state)])
        self.pos_all        = np.zeros([numTimeStep, len(self.quad.pos)])
        self.vel_all        = np.zeros([numTimeStep, len(self.quad.vel)])
        self.quat_all       = np.zeros([numTimeStep, len(self.quad.quat)])
        self.omega_all      = np.zeros([numTimeStep, len(self.quad.omega)])
        self.euler_all      = np.zeros([numTimeStep, len(self.quad.euler)])
        self.sDes_traj_all  = np.zeros([numTimeStep, len(self.traj.sDes)])
        self.wMotor_all     = np.zeros([numTimeStep, len(self.quad.wMotor)])
        self.thr_all        = np.zeros([numTimeStep, len(self.quad.thr)])
        self.tor_all        = np.zeros([numTimeStep, len(self.quad.tor)])

        self.t_all[0]            = self.Ti
        self.s_all[0,:]          = self.quad.state
        self.pos_all[0,:]        = self.quad.pos
        self.vel_all[0,:]        = self.quad.vel
        self.quat_all[0,:]       = self.quad.quat
        self.omega_all[0,:]      = self.quad.omega
        self.euler_all[0,:]      = self.quad.euler
        self.sDes_traj_all[0,:]  = self.traj.sDes
        self.wMotor_all[0,:]     = self.quad.wMotor
        self.thr_all[0,:]        = self.quad.thr
        self.tor_all[0,:]        = self.quad.tor

    def get_reference(self, t):
        sDes = self.traj.desiredState(t+5, self.Ts, self.quad)     
        return utils.stateConversions.sDes2state(sDes) 
    
    def step(self, cmd, t, i):

        state = self.quad.update(cmd, self.wind, t, self.Ts)

        self.t_all[i]             = t
        self.s_all[i,:]           = self.quad.state
        self.pos_all[i,:]         = self.quad.pos
        self.vel_all[i,:]         = self.quad.vel
        self.quat_all[i,:]        = self.quad.quat
        self.omega_all[i,:]       = self.quad.omega
        self.euler_all[i,:]       = self.quad.euler
        self.sDes_traj_all[i,:]   = self.traj.sDes
        self.wMotor_all[i,:]      = self.quad.wMotor
        self.thr_all[i,:]         = self.quad.thr
        self.tor_all[i,:]         = self.quad.tor

        return state
            

    def animate(self):

        # View Results
        # ---------------------------
        animator = utils.Animator(
            self.t_all, 
            self.traj.wps, 
            self.pos_all, 
            self.quat_all, 
            self.sDes_traj_all, 
            self.Ts, 
            self.quad.params, 
            self.traj.xyzType, 
            self.traj.yawType, 
            ifsave=True, 
            drawCylinder=True
        )

        ani = animator.animate()

        plt.show()

quad = Quadcopter()

# simulation setup
Ti=0
Tf=4
Ts=0.1
reference='obstacle'
backend='eom'
sim = Sim(
    Ts=Ts,
    Ti=Ti,
    Tf=Tf,
    quad=quad,
    reference=reference,
    backend=backend
)

# control setup and first command
ctrlHzn = 30
interaction_interval = 1
ctrl = MPC_Point_Ref_Obstacle(
    N=ctrlHzn, sim_Ts=0.1, 
    interaction_interval=interaction_interval, 
    n=17, 
    m=4, 
    quad=quad
)
cmd = ctrl(sim.quad.state.tolist(), sim.reference.tolist()).value(ctrl.U)[:,0]

# Run Simulation
# ---------------------------
for i, t in enumerate(np.arange(sim.Ti,sim.Tf,sim.Ts)):

    print(i)
    # Dynamics (using last timestep's commands)
    # ---------------------------
    # state = self.quad.update(self.cmd, self.wind, t, self.Ts)
    state = sim.step(cmd, t, i)

    # Trajectory for Desired States 
    # ---------------------------
    # this is for point2point

    sDes = sim.traj.desiredState(t+5, sim.Ts, sim.quad)     
    reference = sDes2state(sDes) 
    reference = sim.get_reference(t)
    print(f'state error: {sim.quad.state - reference}')
    print(f'input: {cmd}')

    # Generate Commands (for next iteration)
    # ---------------------------
    if i % interaction_interval == 0:
        print(f'{i} is a multiple of {interaction_interval}')
        cmd = ctrl(state.tolist(), reference.tolist()).value(ctrl.U)[:,0]

sim.animate()