import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
from typing import Optional
import copy

import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.utils.mixer import mixerFM_np
from dpc_sf.dynamics.params import params
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ

# gym for training point to point quadcopter

class QuadcopterGymP2P(Env):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }


    def __init__(
            self,
            state,
            reference,
            Q,  # state penalty 
            R,  # input penalty
            Ts=0.1,
            Ti=0.0,
            Tf=4.0,
            params=params,
            backend='mj', # 'mj', 'eom'
            integrator='euler', # 'euler', 'rk4'
            sa_dtype=np.float64,
        ):

        self.Q = Q
        self.R = R

        self.sa_dtype = sa_dtype

        if backend == 'eom':
            self.quad = QuadcopterPT(
                state=ptu.from_numpy(state),
                reference=reference,
                params=params,
                Ts=Ts,
                Ti=Ti,
                Tf=Tf,
                integrator=integrator
            )

        elif backend == 'mj':
            self.quad = QuadcopterMJ(
                state=state,
                reference=reference,
                params=params,
                Ts=Ts,
                Ti=Ti,
                Tf=Tf,
                integrator=integrator,
                xml_path="quadrotor_x.xml",
                write_path="media/mujoco/",
                render='matplotlib'
            )

        # I can only say this is the obs space if the actual returned observation is a dict much like this
        f32_state_ub = copy.deepcopy(params['state_ub']).astype(sa_dtype)
        f32_state_lb = copy.deepcopy(params['state_lb']).astype(sa_dtype)
        f32_cyl_ub = np.array([1.5, 1.5, 0.5], dtype=sa_dtype)
        f32_cyl_lb = np.array([0.5, 0.5, 0.2], dtype=sa_dtype)

        # must instead stack the reference and the observation
        self.observation_space = spaces.Box(np.hstack([f32_state_lb, f32_state_lb, f32_cyl_lb]), np.hstack([f32_state_ub, f32_state_ub, f32_cyl_ub]), dtype=sa_dtype)

        # actions space will be thrust (1,) (1 -> 20), and moment (3,) (-5 -> 5)
        action_lb = np.array([1,-5,-5,-5], dtype=sa_dtype)
        action_ub = np.array([20,5,5,5], dtype=sa_dtype)
        self.action_space = spaces.Box(action_lb, action_ub, dtype=sa_dtype)

        # obstacle parameters - should be changed on every rollout
        cylinder_xpos = 1
        cylinder_ypos = 1
        cylinder_radius = 0.5
        self.cylinder_params = np.array([cylinder_xpos, cylinder_ypos, cylinder_radius])


    def _get_obs(self):
        state = self.quad.get_state()
        ref = self.quad.reference(self.quad.t)
        return np.hstack([state, ref, self.cylinder_params])
    
    # is state in flight envelope
    def _is_state_in_bound(self, state):
        # ignoring rotor rotational rates as those are clipped anyway
        return (state[:13] > self.quad.params['state_lb'][:13]).all() and (state[:13] < self.quad.params['state_ub'][:13]).all()
    
    def _is_terminal(self):
        state = self.quad.get_state()

        # check if state is within bounds defined in params
        is_state_out_of_bound = not self._is_state_in_bound(state)

        # debugging
        if is_state_out_of_bound:
            ub_breach = state < self.quad.params['state_lb']
            lb_breach = state > self.quad.params['state_ub']
            if ub_breach[13:].any() or lb_breach[13:].any():
                print('rotor angular velocity limit exceeded')
            if ub_breach[:13].any() or lb_breach[:13].any():
                print('primary state limit exceeded')

        # check if simulation time exceeded (reference a function of time)
        is_finish_time = not (self.quad.t < self.quad.Tf)

        # accumulate constraints:
        is_terminal = is_state_out_of_bound or is_finish_time

        # debugging
        if is_terminal:
            pass

        return is_terminal

    def _get_info(self):
        return {}
    
    def _random_state(self, num_samples=1):
        # we know what the max and min are of the state space from self.params
        # generate a random state using gym env np_random
        pos_rand = self.np_random.uniform(-0.1, 0.1, [num_samples, 3])
        # quat_rand = self.np_random.uniform(-np.pi*0.01, np.pi*0.01, [num_samples, 4])
        quat_rand = np.vstack([np.array([1,0,0,0])]*num_samples)
        vel_rand = self.np_random.uniform(-0.05, 0.05, [num_samples, 3])
        angv_rand = self.np_random.uniform(-0.03, 0.03, [num_samples, 3])
        omegas = self.np_random.uniform(522,523,[num_samples, 4])
        random_state = np.concatenate([pos_rand, quat_rand, vel_rand, angv_rand, omegas], axis=1)
        return random_state
    
    def _random_cylinder(self):
        # range of randomness is 0.5 <-> 1.5 for x,y and 0.2 to 0.5 for radius
        xpos = self.np_random.uniform(0.5, 1.5)
        ypos = self.np_random.uniform(0.5, 1.5)
        rad = self.np_random.uniform(0.2, 0.5)
        return np.array([xpos, ypos, rad])

    def _cost(self, cmd: np.ndarray):
        state = self.quad.get_state()

        # quadratic cost
        cost = state @ self.Q @ state + cmd @ self.R @ cmd

        # cylinder constraint violation cost
        xpos, ypos, rad = self.cylinder_params
        dist = np.sqrt((state[0] - xpos)**2 + (state[1] - ypos)**2) - rad
        cost += np.heaviside(-dist, 0) * 1

        # flight envelope constraint violation cost
        is_state_out_of_bound = not self._is_state_in_bound(state)
        cost += is_state_out_of_bound * 1000

        return cost

    def animate(self, state_prediction=None):
        self.quad.animate(state_prediction=state_prediction)

    def step(self, cmd: np.ndarray):

        # cmd is a desired set thrusts and torques
        cmd_w = mixerFM_np(self.quad.params, cmd[0], cmd[1:])
        w_error = cmd_w - self.quad.state.squeeze()[13:]

        # gives the exact desired omegas for the next iteration when using EoM
        p_gain = self.quad.params["IRzz"] / self.quad.Ts

        ctrl = w_error * p_gain

        self.quad.step(ctrl)

        obs = self._get_obs()

        reward = -self._cost(ctrl)

        done = self._is_terminal()

        info = self._get_info()

        return obs, reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        random_state = self._random_state()

        self.quad.set_state(random_state)

        self.quad.t = self.quad.Ti

        self.cylinder_params = self._random_cylinder()

        obs = self._get_obs()

        info = self._get_info()

        return obs, info



if __name__ == '__main__':

    from control.trajectory import waypoint_reference

    # initial conditions
    state=np.array([
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

    reference = waypoint_reference('wp_p2p', average_vel=1.0)

    Q = np.eye(17)
    Q[13,13], Q[14,14], Q[15,15], Q[16,16] = 0, 0, 0, 0
    R = np.eye(4)

    env = QuadcopterGymP2P(
        state=state,
        reference=reference,
        Q = Q,
        R = R,
        Ts = 0.1,
        Ti = 0.0,
        Tf = 4.0,
        params = params,
        backend = 'mj',
        integrator = 'euler'
    )

    env._get_obs()



        

        

        
