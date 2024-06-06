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


if __name__ == "__main__":

    #=============
    # Parameters
    params = {}
    params['policy_number'] = 2
    params['nsteps'] = 50
    params['n_samples'] = 2000
    params['integrator_type'] = 'Euler'
    params['hsizes'] = 32
    params['nh'] = 2
    params['activation'] = 'gelu'
    params['bound_method'] = 'sigmoid_scale'
    params['Qr'] = 5.
    params['Qu'] = 0.1
    params['Qc'] = 10.
    params['terminal_eps'] = 0.01
    params['lr'] = 0.002
    params['batchsize'] =  200

    """
    # # #  Ground truth system model
    """

    nx = 1
    nu = 1
    dt = 0.01  # 0.001
    wbar = 0.01
    f = lambda x, u, dt=dt: x + dt*( u)
    torch.manual_seed(0)

    # sampling rate
    ts = dt
    params['ts'] = ts
    # problem dimensions
    nref = nx           # number of references
    # constraints bounds
    umax = 10
    umin = -umax



    """
    # # #  Dataset 
    """
    rbar = 0.5
    freq = 0.05
    xr = lambda t, rbar=rbar: rbar * math.sin(freq * t)

    #  sampled references for training the policy
    list_refs = [torch.rand(1, 1)*torch.ones(params['nsteps']+1, nref) for k in range(params['n_samples'])]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([params['n_samples'], params['nsteps']+1, nref])
    # Training dataset
    train_data = DictDataset({'x': torch.rand(params['n_samples'], 1, nx),
                              'r': batched_ref}, name='train')

    # references for dev set
    list_refs = [torch.rand(1, 1)*torch.ones(params['nsteps']+1, nref) for k in range(params['n_samples'])]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([params['n_samples'], params['nsteps']+1, nref])
    # Development dataset
    dev_data = DictDataset({'x': torch.rand(params['n_samples'], 1, nx),
                            'r': batched_ref}, name='dev')

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
    # integrate continuous time ODE
    # symbolic system model
    model = Node(f, ['x', 'u'], ['x'], name='model')

    # concatenate control parameters x and r into a vector xi
    cat_fun = lambda x, r: torch.cat([x, r], dim=-1)
    params_node = Node(cat_fun, ['x', 'r'], ['xi'], name='params')

    # neural net control policy
    params['mlp_in_size'] = nx + nref
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
    u = variable("u")
    # objectives
    regulation_loss = params['Qr'] * ((x == ref) ^ 2)  # target posistion
    control_loss = params['Qu'] * ((u == 0.) ^ 2)  # minimize control
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'ref_tracking'

    # list of constraints and objectives
    objectives = [regulation_loss, control_loss]
    constraints = []

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
    R = torch.tensor([xr(k) for k in range(nsteps+1)], dtype=torch.float32).reshape(1, nsteps+1, 1)
    # generate initial data for closed loop simulation
    print('shape = '+str(torch.rand(1, 1, nx, dtype=torch.float32).shape))
    #raise(SystemExit)
    data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
            'r': R}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])

    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
          R=trajectories['r'].detach().reshape(nsteps + 1, nref),
          U=trajectories['u'].detach().reshape(nsteps, nu),
          Umin=Umin, Umax=Umax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx))

    torch.save(best_model, 'single_integrator_policy_'+str(params['policy_number'])+'.pth')
    torch.save(params, 'single_integrator_params_'+str(params['policy_number'])+'.pth')