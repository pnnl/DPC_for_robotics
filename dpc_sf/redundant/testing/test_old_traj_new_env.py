import utils
from quad_refactor import Quadcopter
import utils.pytorch_utils as ptu
import torch
import numpy as np
from trajectory_rework import waypoint_reference, equation_reference
from mpc_refactor import MPC_Point_Ref_Obstacle, MPC_Point_Ref, MPC_Traj_Ref
from utils.animation_rework import Animator
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
            dt, 
            Ti,
            Tf,
            params,
            init_state,
            backend='mj', 
            *args, 
            **kwargs) -> None:
        
        self.dt = dt
        self.Ti = Ti
        self.Tf = Tf
        self.params = params # used to do a lot of the instantiating
        self.t = copy.deepcopy(self.Ti)

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(params["state_lb"], params["state_ub"], dtype=float),
            "target": spaces.Box(params["state_lb"], params["state_ub"], dtype=float)
        })

        self.action_space = spaces.Box(np.array([params["minCmd"]]*4), np.array([params["maxCmd"]]*4), dtype=float)

        self.args = args
        self.kwargs = kwargs

        if backend == 'mj':
            self.mj_init(
                init_state = init_state,
                xml_path = kwargs['xml_path'],
                write_path = kwargs['write_path']
            )
        elif backend == 'eom':
            self.eom_init(
                init_state = init_state,
                state_dot = kwargs['state_dot'],
                reference = kwargs['reference']
            )

    # Backend Specific Code
    # ---------------------

    def eom_init(self, init_state, state_dot, reference):

        # initial conditions
        self.state = copy.deepcopy(init_state)

        # the dynamics function from quad
        self.state_dot = state_dot 

        # we need to create a saving attribute to preserve state
        self.state_history = []
        self.time_history = []
        self.reference_history = []
        
        self.reference = reference

        # dictate functions are related to eom
        self.step = self.eom_step
        self.get_state = self.eom_get_state
        self.set_state = self.eom_set_state
        self.animate = self.eom_animate
        self.reset_sim = self.eom_reset

    def mj_init(self, init_state, xml_path="mujoco/quadrotor_x.xml", write_path="media/mujoco/"):

        # IO
        self.xml_path = xml_path
        self.write_path = write_path

        # instantiate the states that mj does not simulate
        self.omegas = np.array([self.params["w_hover"]]*4)

        self.write_path = write_path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        dt = self.model.opt.timestep # should be 0.01
        assert self.dt == dt
        assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers

        # mjdata constains the state and quantities that depend on it.
        self.data = mj.MjData(self.model)

        # Make renderer, render and show the pixels
        self.renderer = mj.Renderer(model=self.model, height=720, width=1280)

        # Simulate and display video.
        self.frames = []
        mj.mj_resetData(self.model, self.data)  # Reset state and time.
        self.data.ctrl = [self.params["kTh"] * self.params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943

        # animation
        self.duration = 3.8  # (seconds)
        self.framerate = 60  # (Hz)

        # dictate functions are related to eom
        self.step = self.mj_step
        self.get_state = self.mj_get_state
        self.set_state = self.mj_set_state
        self.animate = self.mj_animate
        self.reset_sim = self.mj_reset

        # set initial conditions
        self.set_state(copy.deepcopy(init_state))

    def eom_step(self, cmd):
        
        # save current state to history
        self.state_history.append(copy.deepcopy(self.state)) # deepcopy required, tensor stored by reference
        self.time_history.append(self.t) # no copy required as it is a float, not stored by reference
        self.reference_history.append(np.copy(self.reference(self.t))) # np.copy great as reference is a np.array

        # propogate
        self.state += self.state_dot(self.state, cmd) * self.dt
        self.t += self.dt

        return self.state

    def mj_step(self, cmd):

        # translate omegas to thrust (mj input)
        thr = self.params["kTh"] * self.omegas ** 2
        self.data.ctrl = thr.tolist()

        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data)
        self.omegas += cmd/self.params["IRzz"] * self.dt

        # retrieve time for the environment
        self.t = self.data.time

        # draw
        if len(self.frames) < self.data.time * self.framerate:
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)

        return mj_get_state(self.data, self.omegas)
    
    def eom_get_state(self):

        return self.state
    
    def mj_get_state(self):

        return mj_get_state(self.data, self.omegas)
    
    def eom_set_state(self, state):

        # set the eom state
        self.state = state

    def mj_set_state(self, state):

        qpos, qvel = state2qpv(state)

        self.data.qpos = qpos
        self.data.qvel = qvel

    def eom_animate(self):

        if reference_type == 'wp_p2p':
            drawCylinder = True
        else:
            drawCylinder = False

        animator = Animator(
            states=ptu.to_numpy(torch.vstack(self.state_history)), 
            times=np.array(self.time_history), 
            reference_history=np.vstack(self.reference_history), 
            reference=self.reference, 
            reference_type=self.reference.type, 
            drawCylinder=drawCylinder
        )
        animator.animate() # contains the plt.show()

    def mj_animate(self):
        media.write_video(self.write_path + "video.mp4", self.frames, fps=self.framerate)

    def eom_reset(self, init_state):

        self.state = copy.deepcopy(init_state)
        self.state_history = []
        self.time_history = []
        self.reference_history = []
        self.t = copy.deepcopy(self.Ti)

    def mj_reset(self, init_state):

        self.omegas = np.array([self.params["w_hover"]]*4)
        self.set_state(copy.deepcopy(init_state))

    # General Gym environment from here on out
    # ----------------------------------------

    def random_state(self):
        # we know what the max and min are of the state space from self.params
        # generate a random state using gym env np_random
        pos_rand = self.np_random.uniform(-10, 10, 3)
        quat_rand = self.np_random.uniform(-np.pi, np.pi, 4)
        vel_rand = self.np_random.uniform(-5, 5, 3)
        angv_rand = self.np_random.uniform(-0.3, 0.3, 3)
        omegas = self.np_random.uniform(80,800,4)

        state = np.concatenate([pos_rand, quat_rand, vel_rand, angv_rand, omegas])
        return state
    
    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # set a random state
        self.reset_sim(self.random_state())

        # 
        observation = self.get_state()

        # 
        info = {}

        return observation, info

if __name__ == '__main__':

    from mpc_refactor import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
    import argparse
    from trajectory import Trajectory

    dt = 0.1
    Ti = 0
    Tf = 4
    reference_type = 'wp_p2p' # 'fig8', 'wp_traj', 'wp_p2p'
    backend = 'eom' # 'eom', 'mj'
    
    if backend == 'eom':
        mpc_return_type = 'torch'
    elif backend == 'mj':
        mpc_return_type = 'numpy'
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
    if reference_type == 'wp_traj':
        reference = waypoint_reference(type=reference_type, average_vel=1.6)
    elif reference_type == 'wp_p2p':
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
            reference_traj=reference,
            return_type=mpc_return_type
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
            return_type=mpc_return_type
        )

    # if later I include the reference trajectory in the mujoco then mj will
    # also require the references to be passed to it.
    env = Sim(
        ### parameters for both backends
        dt=dt,
        Ti=Ti,
        Tf=Tf,
        params=quad.params,
        backend=backend,
        init_state=state,
        ### eom specific arguments
        state_dot=quad.state_dot,
        reference=reference,
        ### mj specific arguments
        xml_path="mujoco/quadrotor_x.xml",
        write_path="media/mujoco/",
    )

    # TESTING
    traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
    # instantiate some attributes:
    sDes = traj.desiredState(0, dt, quad)
    reference0 = utils.stateConversions.sDes2state(sDes)
    cmd = ctrl(state, reference0)

    while env.t < env.Tf:

        print(f'time is: {env.t}')
        # generate command based on current state
        if reference_type == 'wp_traj' or reference_type == 'fig8':
            # trajectory mpc contains the reference already, so it only needs state and time
            print(f'state: {env.get_state()}')
            try:
                cmd = ctrl(state, env.t)
            except:
                env.animate()
                raise Exception('mpc infeasible, run animated')
            
        elif reference_type == 'wp_p2p':
            try:
                # TESTING

                sDes = traj.desiredState(env.t, dt, quad)
                reference = utils.stateConversions.sDes2state(sDes)
                cmd = ctrl(state, reference)
                # cmd = ctrl(state, reference(env.t))  
            except:
                env.animate()
                raise Exception('mpc infeasible, run animated')

        # step the state
        state = env.step(cmd)

        # TESTING
        # print(f'state error: {state - reference(env.t)}')
        print(f'state error: {state - reference}')

    env.animate()


