from trajectory import Trajectory
import utils
from quad import Quadcopter

import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import os
import mujoco as mj
import copy
import gymnasium as gym
from gymnasium import spaces
from utils.mujoco_utils import mj_get_state, state2qpv

class Sim(gym.Env):

    def __init__(
            self, 
            Ts,
            Ti,
            Tf,
            quad, 
            reference = 'obstacle', # obstacle, trajectory, p2p
            backend = 'eom', # mj, eom
            xml_path = "mujoco/quadrotor_x.xml",
            write_path = "media/mujoco/",
        ) -> None:

        self.t = copy.deepcopy(Ti)
        self.i = 0
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.quad = quad
        self.backend = backend
        self.reference_type = reference

        if reference == 'obstacle' or reference == 'p2p':
            self.traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
            # instantiate some attributes:
            sDes = self.traj.desiredState(0, Ts, quad)
            self.reference0 = utils.stateConversions.sDes2state(sDes)
        elif reference == 'trajectory':
            self.traj = Trajectory(quad, 'xyz_pos', np.array([2,3,1]))
            # instantiate some attributes:
            sDes = self.traj.desiredState(0, Ts, quad)
            self.reference0 = utils.stateConversions.sDes2state(sDes)
        else:
            raise Exception(f'invalid reference choice: {reference}')
        
        if backend == 'eom':
            # zero wind option
            self.wind = utils.Wind('None', 2.0, 90, -15)

            # timehistory saving options
            numTimeStep = int(Tf/Ts+1)

            self.t_all          = np.zeros(numTimeStep)
            self.s_all          = np.zeros([numTimeStep, len(quad.state)])
            self.pos_all        = np.zeros([numTimeStep, len(quad.pos)])
            self.vel_all        = np.zeros([numTimeStep, len(quad.vel)])
            self.quat_all       = np.zeros([numTimeStep, len(quad.quat)])
            self.omega_all      = np.zeros([numTimeStep, len(quad.omega)])
            self.euler_all      = np.zeros([numTimeStep, len(quad.euler)])
            self.sDes_traj_all  = np.zeros([numTimeStep, len(self.traj.sDes)])
            self.wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
            self.thr_all        = np.zeros([numTimeStep, len(quad.thr)])
            self.tor_all        = np.zeros([numTimeStep, len(quad.tor)])

            self.t_all[0]            = Ti
            self.s_all[0,:]          = quad.state
            self.pos_all[0,:]        = quad.pos
            self.vel_all[0,:]        = quad.vel
            self.quat_all[0,:]       = quad.quat
            self.omega_all[0,:]      = quad.omega
            self.euler_all[0,:]      = quad.euler
            self.sDes_traj_all[0,:]  = self.traj.sDes
            self.wMotor_all[0,:]     = quad.wMotor
            self.thr_all[0,:]        = quad.thr
            self.tor_all[0,:]        = quad.tor

        elif backend == 'mj':
            # instantiate the states that mj does not simulate
            self.omegas = np.array([self.quad.params["w_hover"]]*4)

            self.write_path = write_path
            dirname = os.path.dirname(__file__)
            abspath = os.path.join(dirname + "/" + xml_path)
            xml_path = abspath

            # MuJoCo data structures
            self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
            dt = self.model.opt.timestep # should be 0.01
            assert self.Ts == dt
            assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers

            # mjdata constains the state and quantities that depend on it.
            self.data = mj.MjData(self.model)

            # Make renderer, render and show the pixels
            # model.vis.scale.framelength *= 10
            # model.vis.scale.framewidth *= 10
            self.renderer = mj.Renderer(model=self.model, height=720, width=1280)

            # Simulate and display video.
            self.frames = []
            mj.mj_resetData(self.model, self.data)  # Reset state and time.
            self.data.ctrl = [self.quad.params["kTh"] * self.quad.params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943

            # animation
            self.duration = 3.8  # (seconds)
            self.framerate = 60  # (Hz)
        
        # setup the gym environment variables
        self.state_lower_bound = np.array(quad.raw_constraints[0])
        self.state_upper_bound = np.array(quad.raw_constraints[1])

        self.input_lower_bound = np.array([-100]*4)
        self.input_upper_bound = np.array([100]*4)
        
        # observation space size is implied by the shape of the bounds
        self.observation_space = spaces.Dict({
                "agent": spaces.Box(self.state_lower_bound, self.state_upper_bound, dtype=float),
                "target": spaces.Box(self.state_lower_bound, self.state_upper_bound, dtype=float)
            })
        
        self.action_space = spaces.Box(self.input_lower_bound, self.input_upper_bound, dtype=float)

        # cost matrices
        self.Q = np.zeros([17,17])
        self.Q[0,0] =   1 # x
        self.Q[1,1] =   1 # y
        self.Q[2,2] =   1 # z
        self.Q[3,3] =   1 # q0
        self.Q[4,4] =   1 # q1
        self.Q[5,5] =   1 # q2
        self.Q[6,6] =   1 # q3
        self.Q[7,7] =   1 # xdot
        self.Q[8,8] =   1 # ydot
        self.Q[9,9] =   1 # zdot
        self.Q[10,10] = 1 # p
        self.Q[11,11] = 1 # q
        self.Q[12,12] = 1 # r
        self.Q[13,13] = 0 # wM1
        self.Q[14,14] = 0 # wM2
        self.Q[15,15] = 0 # wM3
        self.Q[16,16] = 0 # wM4

        self.R = np.eye(4) * 1

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def get_reference(self, t):
        sDes = self.traj.desiredState(t+5, Ts, quad)     
        return utils.stateConversions.sDes2state(sDes) 
        
    def step_eom(self, cmd):

        # the updated state is saved internally, but is also returned for control here
        self.quad.update(cmd, self.wind, 0, self.Ts)

        return self.quad.state
    
    def save_eom_state(self, t, i):
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

    
    def animate_eom(self):

        if self.reference_type == 'obstacle':
            drawCylinder = True
        else:
            drawCylinder = False

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
            drawCylinder=drawCylinder
        )

        ani = animator.animate()
        plt.show()
    
    # Mujoco section

    def get_primary_state_from_mj(self, data):
        # generalised positions/velocities not in right coordinates
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        qpos[1] *= -1
        qpos[2] *= -1
        qpos[5] *= -1

        qvel[1] *= -1
        qvel[2] *= -1 # 
        qvel[4] *= -1 # qdot

        return np.concatenate([qpos, qvel]).flatten()
    
    # define the actuator dynamics alone
    def actuator_dot(self, cmd):
        return cmd/self.quad.params["IRzz"]

    def step_mj(self, cmd):

        # translate omegas to thrust (mj input)
        thr = self.quad.params["kTh"] * self.omegas ** 2
        self.data.ctrl = thr.tolist()
        print(self.data.time)
        print(self.data.ctrl)

        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data)
        self.omegas += self.actuator_dot(cmd) * self.Ts

        # draw
        if len(self.frames) < self.data.time * self.framerate:
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)

        # get new state
        # primary_state = self.get_primary_state_from_mj(self.data)
        # new_state = np.concatenate([primary_state, self.omegas])

        new_state = mj_get_state(self.data, self.omegas)

        return new_state
    
    def animate_mj(self):
        media.write_video(self.write_path + "video.mp4", self.frames, fps=self.framerate)

    def mpc_cost(self, state, cmd):
        return state @ self.Q @ state + cmd @ self.R @ cmd

    def step(self, cmd):
        if self.backend == 'eom':
            new_state = self.step_eom(cmd)
            self.save_eom_state(self.t, self.i)
            self.t += self.Ts
            self.i += 1
        elif self.backend == 'mj':
            new_state = self.step_mj(cmd)
            self.t += self.Ts
            self.i += 1
        
        # we observe the whole state for now
        observation = new_state
        # quadratic MPC cost
        cost = self.mpc_cost(new_state, cmd)
        reward = - cost
        # have we breached the constraints
        terminated = (new_state > self.state_upper_bound).any() or (new_state < self.state_lower_bound).any()
        # no info
        info = {}

        return observation, reward, terminated, False, info
        
    def randomise_state(self):

        pos_rand = self.np_random.uniform(-10, 10, 3)
        quat_rand = self.np_random.uniform(-np.pi, np.pi, 4)
        vel_rand = self.np_random.uniform(-5, 5, 3)
        angv_rand = self.np_random.uniform(-0.3, 0.3, 3)
        omegas = self.np_random.uniform(80,800,4)

        if self.backend == 'mj':
            self.data.qpos = np.concatenate([pos_rand, quat_rand])
            self.data.qvel = np.concatenate([vel_rand, angv_rand])
            self.omegas = omegas
            return np.concatenate([self.get_primary_state_from_mj(self.data), omegas])
        
        elif self.backend == 'eom':
            self.quad.state = np.concatenate([pos_rand, quat_rand, vel_rand, angv_rand, omegas])
            return self.quad.state

        
    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.randomise_state()

        # ...
        observation = self._get_obs()

        # reset classes time and index counters
        self.t = copy.deepcopy(self.Ti)
        self.i = 0

        # don't have anything to return for now
        info = {}

        return observation, info
        
    def animate(self):
        if self.backend == 'eom':
            self.animate_eom()
        elif self.backend == 'mj':
            self.animate_mj()



if __name__ == '__main__':

    from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to run closed loop simulations on a quadcopter')

    # Add arguments
    parser.add_argument('-hz', '--ctrl_hzn', default=30, type=float, help='mpc control horizon')
    parser.add_argument('-ts', '--timestep', default=0.1, type=float, help='timestep')
    parser.add_argument('-ti', '--initial_time', default=0, type=float, help='initial time of the simulation')
    parser.add_argument('-tf', '--final_time', default=4, type=float, help='final time of the simulation')
    parser.add_argument('-ii', '--interaction_interval', default=1, type=int, help='interaction interval for the mpc')
    parser.add_argument('-rt', '--reference_type', default='trajectory', help='reference type of the simulation, ["obstacle", "p2p", "trajectory"]')
    parser.add_argument('-bk', '--backend', default='eom', help='the backend of the simulation used, ["mj, eom"]')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    ctrlHzn = args.ctrl_hzn
    Ts = args.timestep
    Ti = args.initial_time
    Tf = args.final_time
    interaction_interval = args.interaction_interval
    reference_type = args.reference_type
    backend = args.backend

    # initial conditions
    quad = Quadcopter()
    state = quad.state

    # setup sim
    sim = Sim(
        Ts=Ts,
        Ti=Ti,
        Tf=Tf,
        quad=quad,
        reference=reference_type,
        backend=backend
    )

    # setup mpc
    if reference_type == "p2p" or reference_type == "obstacle":
        ctrl = MPC_Point_Ref_Obstacle(
            N=ctrlHzn,
            sim_Ts=Ts,
            interaction_interval=interaction_interval, 
            n=17,
            m=4,
            quad=quad
        )
    elif reference_type == 'trajectory':
            ctrl = MPC_Traj_Ref(
            N=ctrlHzn,
            sim_Ts=Ts,
            interaction_interval=interaction_interval, 
            n=17, 
            m=4, 
            quad=quad,
            reference_traj=sim.traj
        )
            
    # testing code:
    import utils.pytorch_utils as ptu
    load = np.load('failed_state.npz')
    state = ptu.from_numpy(load['state'])
    t = load['t'].tolist()

    print(t)

    cmd = ctrl

    # initial command
    if reference_type == "p2p" or reference_type == "obstacle":
        reference = sim.get_reference(0)
        cmd = ctrl(state.tolist(), sim.reference0.tolist()).value(ctrl.U)[:,0]
    elif reference_type == "trajectory":
        cmd = np.zeros([4])

    for idx, t in enumerate(np.arange(Ti, Tf, Ts)):

        print(f'time is: {t}')
        state,reward,_,_,_ = sim.step(cmd)

        print(f'state is: {state}')

        print(reward)

        # if we are changing the command at this timestep
        if idx % interaction_interval == 0 and reference_type == "p2p" or reference_type == "obstacle":
            # sim wraps the reference for a bit of convenience - I should rly rewrite the reference code...
            reference = sim.get_reference(t)
            print(f'state error: {state - reference}')        
            cmd = ctrl(state.tolist(), reference.tolist()).value(ctrl.U)[:,0]

        elif idx % interaction_interval == 0 and reference_type == 'trajectory':
            
            print(f'state error: {state - utils.stateConversions.sDes2state(sim.traj.desiredState(t, Ts, quad))}')
            cmd = ctrl(state.tolist(), t).value(ctrl.U)[:,0]

    sim.animate()

        






















