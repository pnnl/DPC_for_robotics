import utils
from quad import Quadcopter
import utils.pytorch_utils as ptu
import torch
import numpy as np
from trajectory import waypoint_reference, equation_reference
from utils.animation import Animator
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import os
import mujoco as mj
import copy
from gymnasium import spaces
import gymnasium as gym
from utils.mujoco_utils import mj_get_state, state2qpv
from datetime import datetime

# for the gym env
from typing import Optional
from gymnasium.envs.mujoco import MujocoEnv


class Sim(MujocoEnv):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
            self, 
            Ts, 
            Ti,
            Tf,
            params,
            init_state,
            reference,
            backend='mj',
            integrator_type='RK4', # 'euler', 'RK4',
            Q=ptu.from_numpy(np.eye(17))*0.1, # state cost matrix
            R=ptu.from_numpy(np.eye(4)), # input cost matrix
            *args, 
            **kwargs) -> None:
        """
        
        """
        self.integrator_type = integrator_type
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.params = params # used to do a lot of the instantiating
        self.t = copy.deepcopy(self.Ti)

        # I can only say this is the obs space if the actual returned observation is a dict much like this
        f32_state_ub = copy.deepcopy(params['state_ub']).astype(np.float32)
        f32_state_lb = copy.deepcopy(params['state_lb']).astype(np.float32)

        # illegal when working with gymnasium as opposed to gym it seems
        # self.observation_space = spaces.Dict({
        #     "agent": spaces.Box(f32_state_lb, f32_state_ub, dtype=np.float32),
        #     "target": spaces.Box(f32_state_lb, f32_state_ub, dtype=np.float32)
        # })

        # must instead stack the reference and the observation
        self.observation_space = spaces.Box(np.hstack([f32_state_lb]*2), np.hstack([f32_state_ub]*2))

        self.action_space = spaces.Box(np.array([params["minCmd"]]*4, dtype=np.float32), np.array([params["maxCmd"]]*4, dtype=np.float32), dtype=np.float32)

        Q[13,13],Q[14,14],Q[15,15],Q[16,16]=0,0,0,0 # we don't care about the rotor rotational rates
        self.Q = Q
        self.R = R

        self.args = args
        self.kwargs = kwargs

        if backend == 'mj':
            assert isinstance(init_state, np.ndarray), "mujoco requires a numpy state"
            self.mj_init(
                init_state = init_state,
                xml_path = kwargs['xml_path'],
                write_path = kwargs['write_path']
            )
        elif backend == 'eom':
            assert isinstance(init_state, torch.Tensor), "equations of motion require torch tensor state"
            self.eom_init(
                init_state = init_state,
                state_dot = kwargs['state_dot'],
            )

        # we need to create a saving attribute to preserve state
        self.state_history = []
        self.time_history = []
        self.reference_history = []

        # initialise
        self.reference = reference

    # Backend Specific Code
    # ---------------------

    def eom_init(self, init_state, state_dot):

        self.state = copy.deepcopy(init_state)

        # dictate functions are related to eom
        self.state_dot = state_dot 
        self.step = self.eom_step
        self.get_state = self.eom_get_state
        self.set_state = self.eom_set_state
        self.animate = self.eom_animate
        self.reset_sim = self.eom_reset
        self._get_obs = self.eom_get_obs


    def mj_init(self, init_state, xml_path="mujoco/quadrotor_x.xml", write_path="media/mujoco/"):

        DEFAULT_CAMERA_CONFIG = {
            "trackbodyid": 2,
            "distance": 3.0,
            "lookat": np.array((0.0, 0.0, 1.15)),
            "elevation": -20.0,
        }

        # IO
        self.xml_path = xml_path
        self.write_path = write_path

        # instantiate the states that mj does not simulate
        self.omegas = copy.deepcopy(init_state[13:17])

        self.write_path = write_path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.model.opt.timestep = self.Ts
        assert self.Ts == self.model.opt.timestep # should be 0.01
        # assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers
        # mjdata constains the state and quantities that depend on it.
        if self.integrator_type == 'euler':
            # this is NOT explicit euler this is semi-implicit euler
            self.model.opt.integrator = 0
        elif self.integrator_type == 'RK4':
            self.model.opt.integrator = 1
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

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=1,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )

        # dictate functions are related to eom
        self.step = self.mj_step
        self.get_state = self.mj_get_state
        self.set_state = self.mj_set_state
        self.animate = self.mj_animate
        self.reset_sim = self.mj_reset
        self._get_obs = self.mj_get_obs

        # set initial conditions, track with state attribute for convenience
        self.set_state(copy.deepcopy(init_state))

        # mujoco operates on numpy arrays not tensors
        self.state = copy.deepcopy(init_state)

    def eom_step(
            self, 
            cmd: torch.Tensor, 
        ):
        if type(cmd) == np.ndarray:
            cmd = ptu.from_numpy(cmd)
        assert isinstance(cmd, torch.Tensor), f"cmd should be a torch.Tensor for equations of motion sim, recieved: {type(cmd)}"
        
        # keep track of states in this class
        self.save_state()

        # propogate simulation
        # --------------------
        if self.integrator_type == 'euler':
            # this is explicit euler, unlike  mujocos semi-implicit euler
            self.state += self.state_dot(self.state, cmd) * self.Ts
        elif self.integrator_type == 'RK4':
            k1 = self.state_dot(self.state, cmd)
            k2 = self.state_dot(self.state + self.Ts/2 * k1, cmd)
            k3 = self.state_dot(self.state + self.Ts/2 * k2, cmd)
            k4 = self.state_dot(self.state + self.Ts * k3, cmd)
            self.state += self.Ts/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        self.t += self.Ts

        # create the observation to return
        # --------------------------------
        obs = self._get_obs()

        # reward at timestep for command given
        # ------------------------------------
        reward = self.reward(next_state=self.state, cmd=cmd)

        # check if the simulation has breached the constraints or final time reached
        # --------------------------------------------------------------------------
        terminated = self.is_terminate(next_state=self.state)

        # return useful information in compliance with gym env spec
        # ---------------------------------------------------------
        info = {}

        return obs, reward, terminated, False, info

    def mj_step(
            self, 
            cmd: np.ndarray,
        ):
        assert isinstance(cmd, np.ndarray), "cmd should be a np.ndarray for mujoco sim"

        # keep track of states in this class
        self.save_state()

        # translate omegas to thrust (mj input)
        thr = self.params["kTh"] * self.omegas ** 2 # original
        self.data.ctrl = thr.tolist() # original

        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data) # ORIGINAL
        self.omegas += cmd/self.params["IRzz"] * self.Ts
        # thr = self.params["kTh"] * self.omegas ** 2 # DEBUG
        # self.data.ctrl = thr.tolist() # DEBUG
        # mj.mj_step(self.model, self.data) # DEBUG

        # retrieve time for the environment
        self.t = self.data.time

        # draw
        if len(self.frames) < self.data.time * self.framerate:
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)

        self.state = mj_get_state(self.data, self.omegas)

        # create the observation to return
        # --------------------------------
        obs = self._get_obs()

        # reward at timestep for command given
        # ------------------------------------
        # reward = self.reward(next_state=self.state, cmd=cmd)
        reward = 0

        # check if the simulation has breached the constraints or final time reached
        # --------------------------------------------------------------------------
        # terminated = self.is_terminate(next_state=self.state)
        terminated = False

        # return useful information in compliance with gym env spec
        # ---------------------------------------------------------

        info = {}

        return obs, reward, terminated, False, info
        
    def eom_get_state(self):

        return self.state
    
    def mj_get_state(self):

        return mj_get_state(self.data, self.omegas)
    
    def eom_set_state(self, state):

        # set the eom state
        self.state = state

    def mj_set_state(self, state):

        # convert state to mujoco compatible 
        qpos, qvel = state2qpv(state)

        # apply
        self.data.qpos = qpos
        self.data.qvel = qvel

        # handle the rotors (omegas) and state save separately
        self.omegas = copy.deepcopy(state.squeeze()[13:17])
        self.state = copy.deepcopy(state)

    def eom_animate(self, state_prediction=None):

        if self.reference.type == 'wp_p2p':
            drawCylinder = True
        else:
            drawCylinder = False

        animator = Animator(
            states=ptu.to_numpy(torch.vstack(self.state_history)), 
            times=np.array(self.time_history), 
            reference_history=np.vstack(self.reference_history), 
            reference=self.reference, 
            reference_type=self.reference.type, 
            drawCylinder=drawCylinder,
            state_prediction=state_prediction
        )
        animator.animate() # contains the plt.show()

    def mj_animate(self, type='matplotlib', state_prediction=None):

        if type == 'matplotlib':
            if self.reference.type == 'wp_p2p':
                drawCylinder = True
            else:
                drawCylinder = False

            animator = Animator(
                states=np.vstack(self.state_history), 
                times=np.array(self.time_history), 
                reference_history=np.vstack(self.reference_history), 
                reference=self.reference, 
                reference_type=self.reference.type, 
                drawCylinder=drawCylinder,
                state_prediction=state_prediction
            )
            animator.animate() # contains the plt.show()
        elif type == 'mujoco':
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

    def eom_get_obs(self):
        state = self.get_state()
        reference = ptu.from_numpy(self.reference(self.t))
        # return {"agent": state, "target": reference}
        return torch.hstack([state, reference])

    def mj_get_obs(self):
        state = self.get_state()
        reference = self.reference(self.t)
        # return {"agent": state, "target": reference}
        return np.hstack([state, reference])        

    # General Gym environment from here on out
    # ----------------------------------------


    def is_terminate(self, next_state):
        # see if we have left the safe flight envelope (crucial)
        # ------------------------------------------------------
        if (ptu.to_numpy(next_state.squeeze()) < self.observation_space['agent'].low).any():
            return True
        elif (ptu.to_numpy(next_state.squeeze()) > self.observation_space['agent'].high).any():
            return True

        # calculate the distance from the f o r b i d d e n cylinder
        # ----------------------------------------------------------
        next_state = next_state.squeeze()
        x_pos = 1
        y_pos = 1
        radius = 0.5
        dist = torch.sqrt((next_state[0] - x_pos)**2 + (next_state[1] - y_pos)**2) - radius
        if dist < 0 or self.t > self.Tf:
            return True
        else:
            return False

    def reward(self, next_state, cmd):
        # we recieve the next state that was produced by applying the current command to the current state

        # standard quadratic cost - same as MPC
        # -------------------------------------
        next_state = next_state.squeeze()
        cost = next_state @ self.Q @ next_state + cmd @ self.R @ cmd

        # add cost to constraint violation
        # --------------------------------
        x_pos = 1
        y_pos = 1
        radius = 0.5
        dist = torch.sqrt((next_state[0] - x_pos)**2 + (next_state[1] - y_pos)**2) - radius
        zero = ptu.from_numpy(np.array(0))
        cost += torch.heaviside( - dist, zero) * 1

        # switch cost to reward to return
        # -------------------------------
        print(next_state)
        return - cost

    def save_state(self):
        # consistent saving scheme for mj and eom
        self.state_history.append(copy.deepcopy(self.state)) # deepcopy required, tensor stored by reference
        self.time_history.append(self.t) # no copy required as it is a float, not stored by reference
        self.reference_history.append(np.copy(self.reference(self.t))) # np.copy great as reference is a np.array

    def uniform_random_state(self, num_samples=1):
        # we know what the max and min are of the state space from self.params
        # generate a random state using gym env np_random
        pos_rand = self.np_random.uniform(-10, 10, [num_samples, 3])
        quat_rand = self.np_random.uniform(-np.pi, np.pi, [num_samples, 4])
        vel_rand = self.np_random.uniform(-5, 5, [num_samples, 3])
        angv_rand = self.np_random.uniform(-0.3, 0.3, [num_samples, 3])
        omegas = self.np_random.uniform(80,800,[num_samples, 4])
        self.omegas = omegas

        state = np.concatenate([pos_rand, quat_rand, vel_rand, angv_rand, omegas], axis=1)
        return ptu.from_numpy(state)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        # We need the following line to seed self.np_random only when its a gym.Env
        # super().reset(seed=seed)

        # set a random state
        self.reset_sim(self.uniform_random_state())

        # 
        observation = self._get_obs()

        # 
        info = {}

        return observation, info

if __name__ == '__main__':

    from mpc import MPC_Point_Ref, MPC_Traj_Ref
    import argparse

    Ts = 0.1
    Ti = 0
    Tf = 4
    reference_type = 'wp_p2p' # 'fig8', 'wp_traj', 'wp_p2p'
    backend = 'mj' # 'eom', 'mj'
    integrator_type = 'euler' # 'euler', 'RK4'

    if backend == 'eom':
        mpc_return_type = 'torch'
    elif backend == 'mj':
        mpc_return_type = 'numpy'
    quad = Quadcopter()

    state = np.array([
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
    ])

    if backend == 'mj':
        state = ptu.to_numpy(ptu.from_numpy(state))
    elif backend == 'eom':
        state = ptu.from_numpy(state)

    # setup trajectory
    if reference_type == 'wp_traj':
        reference = waypoint_reference(type=reference_type, average_vel=1.6)
    elif reference_type == 'wp_p2p':
        reference = waypoint_reference(type=reference_type, average_vel=0.5)
    elif reference_type == 'fig8':
        reference = equation_reference(type=reference_type, average_vel=0.6)

    # setup mpc
    ctrlHzn = 30
    interaction_interval=1

    if reference_type == 'wp_traj' or reference_type == 'fig8':
        ctrl = MPC_Traj_Ref(
            N=ctrlHzn,
            dt=Ts,
            interaction_interval=interaction_interval, 
            n=17, 
            m=4, 
            dynamics=quad.casadi_state_dot,
            state_ub=quad.params["ca_state_ub"],
            state_lb=quad.params["ca_state_lb"],
            reference_traj=reference,
            return_type=mpc_return_type,
            integrator_type=integrator_type
        )
    elif reference_type == 'wp_p2p':
        ctrl = MPC_Point_Ref(
            N=ctrlHzn,
            dt=Ts,
            interaction_interval=interaction_interval, 
            n=17, 
            m=4, 
            dynamics=quad.casadi_state_dot,
            state_ub=quad.params["ca_state_ub"],
            state_lb=quad.params["ca_state_lb"],
            return_type=mpc_return_type,
            obstacle=True,
            integrator_type=integrator_type
        )


    # if later I include the reference trajectory in the mujoco then mj will
    # also require the references to be passed to it.
    env = Sim(
        ### parameters for both backends
        Ts=Ts,
        Ti=Ti,
        Tf=Tf,
        params=quad.params,
        backend=backend,
        init_state=state,
        reference=reference,
        integrator_type=integrator_type,

        ### eom specific arguments
        state_dot=quad.state_dot,

        ### mj specific arguments
        xml_path="mujoco/quadrotor_x.xml",
        write_path="media/mujoco/",
    )

    # save control predictions for plotting and debugging
    ctrl_pred_x = []
    ctrl_pred_u = []

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
                dist = np.sqrt((state[0] - 1)**2 + (state[1] - 1)**2)
                print(f'dist: {dist}')
                cmd = ctrl(state, reference(env.t))

            except:
                env.animate(state_prediction=np.stack(ctrl_pred_x))
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                np.savez(f'states/{current_datetime}.npz', state=state, reference=reference(env.t))
                # raise Exception('mpc infeasible, run animated, state saved')

        # for debugging
        # -------------
        ctrl_predictions = ctrl.get_predictions() 
        ctrl_pred_x.append(ctrl_predictions[0])
        ctrl_pred_u.append(ctrl_predictions[1])

        # step the state
        obs,_,_,_,_ = env.step(cmd)
        state = obs[0:17]

        # TESTING
        print(f'state error: {state - reference(env.t)}')

    ctrl_pred_x = np.stack(ctrl_pred_x)
    ctrl_pred_u = np.stack(ctrl_pred_u)

    env.animate(state_prediction=ctrl_pred_x, type='mujoco')


