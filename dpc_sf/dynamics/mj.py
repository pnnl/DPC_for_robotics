import dpc_sf.utils as utils
import dpc_sf.utils.pytorch_utils as ptu
import torch
import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference, equation_reference
from dpc_sf.utils.animation import Animator
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import os
import mujoco as mj
import copy
from gymnasium import spaces, Env
import gymnasium as gym
from dpc_sf.utils.mujoco_utils import mj_get_state, state2qpv
from datetime import datetime

# for the gym env example
from typing import Optional

import mujoco_viewer

from dpc_sf.dynamics.params import params

class QuadcopterMJ():

    def __init__(
            self, 
            state: torch.Tensor, # initial condition
            reference,
            params=params,
            Ts: float = 0.1,
            Ti: float = 0.0,
            Tf: float = 4.0,
            integrator='euler', # 'euler', 'rk4'
            xml_path="quadrotor_x.xml", 
            write_path="media/mujoco/",
            render='matplotlib', # 'matplotlib', 'mujoco'
            save_first_reset=False, # in the event of
        ) -> None:
        """
        
        """
        self.integrator = integrator
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.params = params # used to do a lot of the instantiating
        self.t = Ti
        self.render = render

        # I can only say this is the obs space if the actual returned observation is a dict much like this
        f32_state_ub = copy.deepcopy(params['state_ub']).astype(np.float32)
        f32_state_lb = copy.deepcopy(params['state_lb']).astype(np.float32)

        # must instead stack the reference and the observation
        self.observation_space = spaces.Box(np.hstack([f32_state_lb]*2), np.hstack([f32_state_ub]*2))
        self.action_space = spaces.Box(np.array([params["minCmd"]]*4, dtype=np.float32), np.array([params["maxCmd"]]*4, dtype=np.float32), dtype=np.float32)

        # we need to create a saving attribute to preserve state
        self.state_history = []
        self.input_history = []
        self.time_history = []
        self.reference_history = []

        # initialise
        self.reference = reference

        print('initialising mujoco environment')

        # DEFAULT_CAMERA_CONFIG = {
        #     "trackbodyid": 2,
        #     "distance": 3.0,
        #     "lookat": np.array((0.0, 0.0, 1.15)),
        #     "elevation": -20.0,
        # }

        # IO
        self.xml_path = xml_path
        self.write_path = write_path

        # instantiate the states that mj does not simulate
        self.omegas = copy.deepcopy(state[13:17])

        self.write_path = write_path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path, )  # MuJoCo model
        self.model.opt.timestep = self.Ts
        assert self.Ts == self.model.opt.timestep # should be 0.01
        # assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers
        # mjdata constains the state and quantities that depend on it.
        if self.integrator == 'euler':
            # this is NOT explicit euler this is semi-implicit euler
            self.model.opt.integrator = 0
        elif self.integrator == 'rk4':
            self.model.opt.integrator = 1
        self.data = mj.MjData(self.model)

        # Make renderer, render and show the pixels
        if render == 'mujoco':
            self.renderer = mj.Renderer(model=self.model, height=720, width=1280)
            # self.data.cam_xpos = np.array([[1,2,3]])
            self.model.cam_pos0 = np.array([[1,2,3]])
            self.model.cam_pos = np.array([[1,2,3]])

        # Simulate and display video.
        self.frames = []
        mj.mj_resetData(self.model, self.data)  # Reset state and time.
        self.data.ctrl = [self.params["kTh"] * self.params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943

        # animation
        self.duration = 3.8  # (seconds)
        self.framerate = 60  # (Hz)

        self.current_cmd = np.zeros(4)

        # set initial conditions, track with state attribute for convenience
        self.set_state(copy.deepcopy(state))

        # mujoco operates on numpy arrays not tensors
        self.state = copy.deepcopy(state)


    def start_online_render(self):
        self.render = 'online_mujoco'
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def step(
            self, 
            cmd: np.ndarray,
        ):
        assert isinstance(cmd, np.ndarray), "cmd should be a np.ndarray for mujoco sim"
        self.current_cmd = cmd
        # keep track of states in this class
        self.save_state()

        # translate omegas to thrust (mj input)
        thr = self.params["kTh"] * self.omegas ** 2
        self.data.ctrl = thr.tolist()
        # self.model.cam_pos = np.array([[1,2,3]])
        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data)
        self.omegas += cmd/self.params["IRzz"] * self.Ts

        # retrieve time for the environment
        self.t = self.data.time

        # draw
        if len(self.frames) < self.data.time * self.framerate and self.render == 'mujoco':
            # self.data.cam_xpos = np.array([[0,0,0]])
            
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)
        elif len(self.frames) < self.data.time * self.framerate and self.render == 'online_mujoco':
            self.viewer.render()
            

        self.state = self.get_state()
    
    def get_state(self):

        return mj_get_state(self.data, self.omegas)

    def set_state(self, state):

        # convert state to mujoco compatible 
        qpos, qvel = state2qpv(state)

        # apply
        self.data.qpos = qpos
        self.data.qvel = qvel

        # handle the rotors (omegas) and state save separately
        self.omegas = copy.deepcopy(state.squeeze()[13:17])
        self.state = copy.deepcopy(state)

        # save the set state
        self.save_state()

    def save_state(self):
        # consistent saving scheme for mj and eom
        self.state_history.append(copy.deepcopy(self.state)) # deepcopy required, tensor stored by reference
        self.time_history.append(self.t) # no copy required as it is a float, not stored by reference
        self.reference_history.append(np.copy(self.reference(self.t))) # np.copy great as reference is a np.array
        self.input_history.append(np.copy(self.current_cmd)) # np.co py great as reference is a np.array


    def animate(self, state_prediction=None, render_interval=1):

        if self.render == 'matplotlib':

            if self.reference.type == 'wp_p2p':
                drawCylinder = True
            else:
                drawCylinder = False

            if state_prediction is not None:
                animator = Animator(
                    states=np.vstack(self.state_history)[::render_interval,:], 
                    times=np.array(self.time_history)[::render_interval], 
                    reference_history=np.vstack(self.reference_history)[::render_interval,:], 
                    reference=self.reference, 
                    reference_type=self.reference.type, 
                    drawCylinder=drawCylinder,
                    state_prediction=state_prediction[::render_interval,...]
                )
            else:
                animator = Animator(
                    states=np.vstack(self.state_history)[::render_interval,:], 
                    times=np.array(self.time_history)[::render_interval], 
                    reference_history=np.vstack(self.reference_history)[::render_interval,:], 
                    reference=self.reference, 
                    reference_type=self.reference.type, 
                    drawCylinder=drawCylinder,
                    state_prediction=state_prediction
                )
            animator.animate() # contains the plt.show()
        elif self.render == 'mujoco':
            media.write_video(self.write_path + "video.mp4", self.frames, fps=self.framerate)

    def reset(self, state):

        print('performing mujoco reset')

        self.omegas = np.array([self.params["w_hover"]]*4)
        self.set_state(copy.deepcopy(state))

        # added during sysID phase
        self.t = self.Ti

        self.state_history = []
        self.time_history = []
        self.reference_history = []

        # self.save_state()

    def mj_get_obs(self):
        state = self.get_state()
        reference = self.reference(self.t)
        return np.hstack([state, reference])        

if __name__ == '__main__':

    # end to end test
    def e2e_test(test='wp_traj'):

        from control.mpc import MPC_Point_Ref, MPC_Traj_Ref
        from control.trajectory import waypoint_reference
        from control.trajectory import equation_reference
        from quad_dynamics.eom_ca import QuadcopterCA

        Ts = 0.1
        Ti = 0.0
        Tf = 15
        integrator = 'euler' # 'euler', 'RK4'

        if test == 'wp_p2p' or test == 'wp_traj':
            reference = waypoint_reference(test, average_vel=1.0)
        else:
            reference = equation_reference(test, average_vel=0.6)

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

        quadCA = QuadcopterCA(params=params)

        quad = QuadcopterMJ(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator='euler',
            xml_path="mujoco/quadrotor_x.xml",
            write_path="media/mujoco/",
            render='matplotlib'
        )

        if test == 'wp_p2p':
            ctrl = MPC_Point_Ref(
                N=30,
                dt=Ts,
                interaction_interval=1,
                n=17,
                m=4,
                dynamics=quadCA.state_dot,
                state_ub=params['ca_state_ub'],
                state_lb=params['ca_state_lb'],
                return_type='numpy',
                obstacle=True,
                integrator_type=integrator
            )
        elif test == 'wp_traj' or test == 'fig8':
            ctrl = MPC_Traj_Ref(
                N=30,
                dt=Ts,
                interaction_interval=1, 
                n=17, 
                m=4, 
                dynamics=quadCA.state_dot,
                state_ub=params["ca_state_ub"],
                state_lb=params["ca_state_lb"],
                reference_traj=reference,
                return_type='numpy',
                integrator_type=integrator
            )


        ctrl_pred_x = []
        while quad.t < quad.Tf:
            print(quad.t)
            if test == 'wp_p2p':
                cmd = ctrl(quad.state, reference(quad.t))
            else:
                cmd = ctrl(quad.state, quad.t)
            quad.step(cmd)

            ctrl_predictions = ctrl.get_predictions() 
            ctrl_pred_x.append(ctrl_predictions[0])


        ctrl_pred_x = np.stack(ctrl_pred_x)
        quad.animate(state_prediction=ctrl_pred_x)

    e2e_test('wp_traj')

