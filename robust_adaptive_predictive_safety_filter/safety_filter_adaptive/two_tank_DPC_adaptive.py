"""
Neural Ordinary Differentiable predictive control (NO-DPC)

Reference tracking of nonlinear ODE system with explicit neural control policy via DPC algorithm

system: Two Tank model
example inspired by: https://apmonitor.com/do/index.php/Main/LevelControl
"""

import torch
import torch.nn as nn
import numpy as np
import math
import copy

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase
from neuromancer.dynamics.ode import ODESystem

class TwoTankParam(ODESystem):
    def __init__(self, insize=6, outsize=2):
        """

        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)

    def ode_equations(self, x, u, a):
        # heights in tanks
        h1 = torch.clip(x[:, [0]], min=0, max=1.0)
        h2 = torch.clip(x[:, [1]], min=0, max=1.0)
        # Inputs (2): pump and valve
        pump = torch.clip(u[:, [0]], min=0, max=1.0)
        valve = torch.clip(u[:, [1]], min=0, max=1.0)
        c1 = a[:, [0]]
        c2 = a[:, [1]]
        # equations
        dhdt1 = c1 * (1.0 - valve) * pump - c2 * torch.sqrt(h1)
        dhdt2 = c1 * valve * pump + c2 * torch.sqrt(h1) - c2 * torch.sqrt(h2)
        return torch.cat([dhdt1, dhdt2], dim=-1)

class TwoTankDynamics():
    def __init__(self, model,dbar=0):
        self.model = model
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1, 1]
        self.xmin = [0, 0]
        self.x0 = [0.1, 0.1]
        self.in_features = 6
        self.out_features = 2

    def __call__(self, x, u, params):


        c1 = params[0]
        c2 = params[1]
        dhdt0 = c1 * (1.0 - u[1]) * u[0] - c2 * torch.sqrt(x[0])
        dhdt1 = c1 * u[1] * u[0] + c2 * torch.sqrt(x[0]) - c2 * torch.sqrt(x[1])

        return torch.cat([dhdt0, dhdt1], dim=-1)

# class Integrator():
#     def __init__(self, f, dT, integrator_type='Euler'):
#         self.f = f
#         self.dT = dT
#         self.integrator_type = integrator_type
#     def __call__(self, x, u, a):
#         '''Integrate system f at time k by one time step with x and u and time k using integrator_type method
#         x: state
#         u: control input
#         '''
#
#         if self.integrator_type == 'RK4':
#             # Runge-Kutta 4 integration
#             k1 = self.f(x, u, a)
#             k2 = self.f(x + self.dT / 2 * k1, u, a)
#             k3 = self.f(x + self.dT / 2 * k2, u, a)
#             k4 = self.f(x + self.dT * k3, u, a)
#             x_next = x + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
#
#         if self.integrator_type == 'Euler':
#             # Standard Euler integration
#             x_next = x + self.dT * self.f(x, u, a)
#
#         if self.integrator_type == 'cont':
#             # Treat system as 'continuous'
#             x_next = self.f(x, u, a)
#
#         return x_next

if __name__ == "__main__":

    #=============
    # Parameters
    params = {}
    params['policy_number'] = 4
    params['nsteps'] = 50
    params['n_samples'] = 4000
    params['integrator_type'] = 'Euler'
    params['hsizes'] = 32
    params['nh'] = 2
    params['activation'] = 'gelu'
    params['bound_method'] = 'sigmoid_scale'
    params['Qr'] = 5.
    params['Qc'] = 10.
    params['terminal_eps'] = 0.01
    params['lr'] = 0.002
    params['batchsize'] =  200

    """
    # # #  Ground truth system model
    """
    torch.manual_seed(0)
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model)
    # sampling rate
    ts = 10*gt_model.params[1]['ts']
    params['ts'] = ts
    # problem dimensions
    nx = gt_model.nx    # number of states
    nu = gt_model.nu    # number of control inputs
    nref = nx           # number of references
    # constraints bounds
    umin = 0
    umax = 1.
    xmin = 0.2
    xmax = 1.

    # Model bounds
    c1_max = 0.12
    c1_min = 0.07
    c2_max = 0.08
    c2_min = 0.03
    model_max = torch.tensor([[c1_max, c2_max]])
    model_min = torch.tensor([[c1_min, c2_min]])
    nmodel = 2


    """
    # # #  Dataset 
    """


    #  sampled references for training the policy
    model_list = [ torch.mm(torch.ones(params['nsteps'] + 1, 1), (model_min + torch.mm((model_max - model_min), torch.diag(torch.rand(nmodel))))) for k in
                  range(params['n_samples'])]
    model_data = torch.cat(model_list)
    list_refs = [torch.rand(1, 1)*torch.ones(params['nsteps']+1, nref) for k in range(params['n_samples'])]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([params['n_samples'], params['nsteps']+1, nref])
    # Training dataset
    train_data = DictDataset({'x': torch.rand(params['n_samples'], 1, nx),
                              'r': batched_ref,
                              'a': model_data.reshape([params['n_samples'], params['nsteps']+1, nmodel])}, name='train')

    # references for dev set
    model_list = [torch.mm(torch.ones(params['nsteps'] + 1, 1), (model_min + torch.mm((model_max - model_min), torch.diag(torch.rand(nmodel))))) for k in
                  range(params['n_samples'])]
    model_data = torch.cat(model_list)
    list_refs = [torch.rand(1, 1)*torch.ones(params['nsteps']+1, nref) for k in range(params['n_samples'])]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([params['n_samples'], params['nsteps']+1, nref])
    # Development dataset
    dev_data = DictDataset({'x': torch.rand(params['n_samples'], 1, nx),
                            'r': batched_ref,
                            'a': model_data.reshape([params['n_samples'], params['nsteps']+1, nmodel])}, name='dev')

    # torch dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batchsize'],
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=params['batchsize'],
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    """
    # # #  System model and Control policy in Neuromancer
    """
    # white-box ODE model with no-plant model mismatch
    two_tank_ode = TwoTankParam()
    #integrator = Integrator( f=model, dT=ts, integrator_type=params['integrator_type'] )
    # integrate continuous time ODE
    interp_u = lambda tq, t, u: u
    integrator = integrators.Euler(two_tank_ode, h=torch.tensor(ts), interp_u=interp_u)
    # symbolic system model
    model = Node(integrator, ['x', 'u', 'a'], ['x'], name='model')

    # concatenate control parameters x and r into a vector xi
    cat_fun = lambda x, r, a: torch.cat([x, r, a], dim=-1)
    params_node = Node(cat_fun, ['x', 'r', 'a'], ['xi'], name='params')

    # neural net control policy
    params['mlp_in_size'] = nx + nref + nmodel
    #net = blocks.MLP_bounds(insize=params['mlp_in_size'], outsize=nu, hsizes=[params['hsizes'] for ii in range(params['nh'])],
                        #nonlin=activations[params['activation']], min=umin, max=umax)
    net = blocks.MLP_bounds(insize=params['mlp_in_size'], outsize=nu,
                            hsizes=[params['hsizes'] for ii in range(params['nh'])],
                            nonlin=activations[params['activation']],
                            min=umin,
                            max=umax,
                            )

    policy = Node(net, ['xi'], ['u'], name='policy')
    # closed-loop system model
    cl_system = System([params_node, policy, model], nsteps=params['nsteps'],
                       name='cl_system')
    cl_system.show()

    """
    # # #  Differentiable Predictive Control objectives and constraints
    """
    # variables
    x = variable('x')
    ref = variable("r")
    # objectives
    regulation_loss = params['Qr'] * ((x == ref) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = params['Qc']*(x > xmin)
    state_upper_bound_penalty = params['Qc']*(x < xmax)
    terminal_lower_bound_penalty = params['Qc']*(x[:, [-1], :] > ref-params['terminal_eps'])
    terminal_upper_bound_penalty = params['Qc']*(x[:, [-1], :] < ref+params['terminal_eps'])
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'ref_tracking'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  Differentiable optimal control problem 
    """
    # data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
    nodes = [cl_system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(nodes, loss)
    # plot computational graph
    problem.show()

    """
    # # #  Solving the problem 
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=params['lr'])
    #  Neuromancer trainer
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=100,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=50,
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)

    """
    Test Closed Loop System
    """
    print('\nTest Closed Loop System \n')
    nsteps = 750 #params['nsteps'] #
    step_length = 150
    # generate reference
    np_refs = psl.signals.step(nsteps+1, 1, min=xmin, max=xmax, randsteps=5)
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
    torch_ref = torch.cat([R, R], dim=-1)

    a = torch.tensor( torch.mm(torch.ones(nsteps + 1, 1), (model_min + torch.mm((model_max - model_min), torch.diag(torch.rand(nmodel))))), dtype=torch.float32).reshape(1, nsteps+1, nmodel)
    print('a = ' + str(a[0, 0, :]))

    # generate initial data for closed loop simulation
    data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
            'r': torch_ref,
            'a': a}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])
    Xmin = xmin * np.ones([nsteps+1, nx])
    Xmax = xmax * np.ones([nsteps+1, nx])
    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
          R=trajectories['r'].detach().reshape(nsteps + 1, nref),
          U=trajectories['u'].detach().reshape(nsteps, nu),
          Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx))

    torch.save(best_model, 'two_tank_adaptive_policy_'+str(params['policy_number'])+'.pth')
    torch.save(params, 'two_tank_adaptive_params_'+str(params['policy_number'])+'.pth')