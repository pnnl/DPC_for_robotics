# -*- coding: utf-8 -*-
"""
original author, link to original: https://github.com/bobzwik/Quadcopter_SimCon
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

"""
edits:
    - Dynamics ported to PyTorch for differentiability
    - Rotor actuators 2nd order -> 1st order
        - 21 states -> 17 states
    - Jacobian formed and validated
        - SymPy derived jacobian, validated through generating the functorch.jacrev
          jacobian using the nonlinear dynamics and validating consistency with
          equations.
    - Minor changes to reduce number of files/complexity
"""

import torch
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.normalisation import denormalize_pt, normalize_pt
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.mj import QuadcopterMJ

import torch
from dpc_sf.dynamics.eom_pt import state_dot_nm

class QuadcopterDPC(torch.nn.Module):
    """
    nn.Module version of dynamics
    """
    def __init__(
            self, 
            params=quad_params,
            ts = 0.1,
            mean = None,
            var = None,
            normalize = True,
            include_actuators = False,
            nx = 17,
            nu=4,
            backend='eom',
            reference=None
        ) -> None:

        super().__init__()

        self.params = params
        self.ts = ts
        self.t = 0.0
        self.nx = nx
        self.nu = nu
        self.normalize = normalize
        self.mean = mean
        self.var = var
        self.include_actuators = include_actuators
        self.backend = backend
        self.reference = reference

        if backend == 'eom':
            self.forward = self.eom_forward
        elif backend == 'mj':
            assert reference is not None, "reference is required for mj backend"
            self.mj_quad = QuadcopterMJ(
                state=params["default_init_state_np"],
                reference=reference,
                params=params,
                Ts=ts,
                Ti=0.0,
                Tf=100.0,
                integrator='euler',
                xml_path="quadrotor_x.xml",
                write_path="media/mujoco/",
                render='matplotlib' # render='online_mujoco'
            )
            self.forward = self.mj_forward
        else:
            print("invalid backend selected")

    def mj_reset(self, x):
        # required at the end of a rollout
        self.mj_quad.set_state(x)

    def mj_forward(self, x, u):
        u = u.squeeze(0)
        self.mj_quad.step(ptu.to_numpy(u))
        xnext = ptu.from_numpy(self.mj_quad.get_state()).unsqueeze(0)
        return xnext
        
    def eom_forward(self, x, u):

        if self.normalize:
            x = denormalize_pt(x, means=ptu.from_numpy(self.mean), variances=ptu.from_numpy(self.var))

        xdot = state_dot_nm(state=x, cmd=u, params=self.params, include_actuators=self.include_actuators)
        xnext = x + xdot * self.ts

        if self.normalize:
            xnext = normalize_pt(xnext, means=ptu.from_numpy(self.mean), variances=ptu.from_numpy(self.var))

        if xnext.isnan().any():
            print('xnext has NaNs within it')

        return xnext

# This quadcopter class contains no state, only simulation parameters and function
# class QuadcopterDPC(torch.nn.Module):
#     def __init__(self, params=params) -> None:
#         super().__init__()
#         self.params = params
#         self.in_features, self.out_features = 21, 17
# 
#     def forward(self, state_cmd: torch.tensor):
# 
#         state = state_cmd[:,:17]
#         cmd = state_cmd[:,17:]
# 
#         # the baseline known dynamics
#         state_dot = state_dot_pt(state, cmd, self.params)#  * 1.5 * self.beta
# 
#         def ensure_2d(tensor):
#             if len(tensor.shape) == 1:
#                 tensor = torch.unsqueeze(tensor, 0)
#             return tensor
#         
#         state_dot = ensure_2d(state_dot)
# 
#         return state_dot
    

# class QuadcopterODE(ODE):
#     """
#     PSL port of quadcopter simulation
#     """
#     @property
#     def params(self):
#         variables = {'x0': quad_params["default_init_state_list"]}
#         constants = {'ts': 0.1}
#         parameters = {}
#         meta = {}
#         return variables, constants, parameters, meta
#     
#     @cast_backend
#     def get_x0(self):
#         return random_state()
#     
#     @cast_backend
#     def get_U(self, nsim, signal=None, **signal_kwargs):
#         if signal is not None:
#             return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
#         u = step(nsim=nsim, d=4, min=-1, max=1, randsteps=int(np.ceil(self.ts*nsim/30)), rng=self.rng)
#         return u
#     
#     @cast_backend
#     def equations(self, t, x, u):
#         return state_dot_np(state=x, cmd=u, params=quad_params)
