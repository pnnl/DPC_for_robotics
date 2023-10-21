from dpc_sf.utils.random_state import random_state
from dpc_sf.utils.normalisation import normalize_nm, denormalize_nm, normalise_dict
from dpc_sf.dynamics.params import params
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.dynamics.eom_pt import QuadcopterPT

from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.dynamics.integrators import Euler, RK4

import dpc_sf.utils.pytorch_utils as ptu

import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
# ODESystem wrapper for quad.state_dot
class EulerIntegrator(torch.nn.Module):
    def __init__(self, state_dot, ts=0.1):
        super().__init__()
        self.state_dot = state_dot

        interp_u = lambda tq, t, u: u
        self.integrator = Euler(self.state_dot, h=torch.tensor(ts), interp_u=interp_u)
    
    def forward(self, xn, u):
        return self.integrator(xn, u=u)
    
def get_node(state_dot, ts=0.1):
    integrator = EulerIntegrator(state_dot=state_dot, ts=ts)
    nodes = [Node(integrator, ['xn', 'U'], ['xn'])]
    system = System(nodes)
    return system

def get_dpc_node(state_dot, ts=0.1):
    interp_u = lambda tq, t, u: u
    integrator = EulerIntegrator(state_dot=state_dot, ts=ts, )
    nodes = [Node(integrator, ['xn', 'R'], ['U'])]
    system = System(nodes)
    return system

def simulate(quad, nsim, x0=None):
    # u = (np.random.uniform([0,0,0,0]) - 0.5)*1e-5 # between -0.25, 0.25
    if x0 is None:
        x = random_state()
    else:
        x = x0
    init_state = copy.deepcopy(x)
    quad.reset(x)
    u_history = []
    u = np.zeros(4) # (np.random.uniform([0,0,0,0]) - 0.5)*1e-3
    for i in range(nsim):
         # between -0.25, 0.25
        quad.step(u)
        u_history.append(u)
    u_history = np.vstack(u_history)
    history = np.vstack(quad.state_history)
    state_dot_history = np.diff(history, axis=0)
    return {'X': history, 'U': u_history, 'xn': init_state}   

def simulate_pt(quad, nsim, x0=None):
    if x0 is None:
        x = random_state()
    else:
        x = x0
    init_state = ptu.from_numpy(copy.deepcopy(x))
    init_state.requires_grad = False
    quad.reset(x)
    Ts = quad.Ts
    x = ptu.from_numpy(x)
    x.requires_grad = False
    u_history = []
    x_history = []
    u = torch.zeros(4)
    for i in range(nsim):
        xd = quad.state_dot(x, u)
        x += xd * Ts
        u_history.append(u)
        x_history.append(x)
    u_history = torch.vstack(u_history)
    x_history = torch.vstack(x_history)
    return {'X': x_history, 'U': u_history, 'xn': init_state}

def get_dpc_data(nsteps, quad, nsim, bs, normalize=True):
    train_data, dev_data, test_data = [simulate_pt(quad, nsim=nsim) for i in range(3)]
    if normalize:
        train_data, dev_data, test_data = [normalise_dict(d) for d in [train_data, dev_data, test_data]]
    
    train_dataset = DictDataset({'R': train_data['X'], 'X': train_data['X'], 'xn': train_data['xn']}, name='train')
    dev_dataset = DictDataset({'R': dev_data['X'], 'X': train_data['X'], 'xn': dev_data['xn']}, name='dev')
    train_loader, dev_loader = [DataLoader(d, batch_size=bs, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset]]
    return train_loader, dev_loader

def get_data(nsteps, quad, nsim, bs, normalize=False):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    X, T, U = [], [], []
    # quad_mj.start_online_render()
    for _ in range(nsim):
        sim = simulate(quad, nsim=nsteps, x0=random_state())
        X.append(sim['X'])
        U.append(sim['U'])

    X, U = np.stack(X), np.stack(U)
    sim = simulate(quad, nsim=nsim * nsteps, x0=random_state())

    nx, nu = X.shape[-1], U.shape[-1]
    x, u = sim['X'].reshape(nsim, nsteps, nx), sim['U'].reshape(nsim, nsteps, nu)
    X, U = np.concatenate([X, x], axis=0), np.concatenate([U, u], axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    U = torch.tensor(U, dtype=torch.float32)
    if normalize:
        X = normalize_nm(X, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))
        U = normalize_nm(U, means=ptu.from_numpy(params["input_mean"]), variances=ptu.from_numpy(params["input_var"]))
    train_data = DictDataset({'X': X, 'U': U, 'xn': X[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    dev_data = DictDataset({'X': X[0:1], 'U': U[0:1], 'xn': X[0:1, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return nx, nu, train_loader, dev_loader, test_loader

def generate_data(
        Ts=0.1,
        nsteps=3,
        nsim=10000,
        bs=32,
        quad_type='eom', # 'mj', 'eom'
        normalize=True
    ):

    state = params['default_init_state_np']
    reference = waypoint_reference('wp_p2p', average_vel=1.0)
    if quad_type == 'mj':
        quad = QuadcopterMJ(state=state, reference=reference, integrator='euler', Ts=Ts)
    elif quad_type == 'eom':
        quad = QuadcopterPT(state=state, reference=reference, integrator='euler', Ts=Ts)
    nx, nu, train_loader, dev_loader, test_loader = get_data(nsteps=nsteps, quad=quad, nsim=nsim, bs=bs, normalize=normalize)
    return nx, nu, train_loader, dev_loader, test_loader, quad 

def generate_dpc_data(
        Ts=0.1,
        nsteps=3,
        nsim=10000,
        bs=32,
        quad_type='eom', # 'mj', 'eom'
        normalize=True
    ):

    state = params['default_init_state_np']
    reference = waypoint_reference('wp_p2p', average_vel=1.0)
    if quad_type == 'mj':
        quad = QuadcopterMJ(state=state, reference=reference, integrator='euler', Ts=Ts)
    elif quad_type == 'eom':
        quad = QuadcopterPT(state=state, reference=reference, integrator='euler', Ts=Ts)
    train_loader, dev_loader = get_dpc_data(nsteps=nsteps, quad=quad, nsim=nsim, bs=bs, normalize=normalize)
    return train_loader, dev_loader, quad 
