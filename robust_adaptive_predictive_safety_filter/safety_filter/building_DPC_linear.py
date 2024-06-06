import torch
import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.modules.activations import activations
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase, pltOL
from neuromancer.psl import systems
from building_safety_filter import SquareWave
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #====================================
    # Define parameters
    params = {}
    params['batchsize'] = 1000
    params['mini_batch'] = 100
    params['model_name'] = 'SimpleSingleZone'  #'Reno_full'#  #'SimpleSingleZone' #'Reno_full' # 'Old_full' #'RenoLight_full' #
    params['model_type'] = 'Linear'
    params['policy_number'] = 5
    #params['bound_method'] = 'relu_clamp' #'sigmoid_scale'
    params['hsizes'] = 32
    params['nh'] = 2
    params['nstep'] = 100
    params['Nsim'] = 1000
    params['Qu'] = 0.01 #0.0 # 0.000001 #0#.
    params['Qdu'] = 0.1
    params['Qc'] = 50.
    params['lr'] = 0.001# 0.01 #
    params['time_var'] = False
    params['activation'] = 'gelu'
    params['loss_type'] = 'PenaltyLoss' #'AugmentedLagrangeLoss' # 'BarrierLoss' #
    my_loss = {'AugmentedLagrangeLoss': AugmentedLagrangeLoss, 'BarrierLoss': BarrierLoss, 'PenaltyLoss': PenaltyLoss}

    # Define comfort constraints
    params['Cup'] = 22.0
    params['Clow'] = 18.0
    if params['time_var']:
        scu_k = SquareWave(period=300, amp=0, offset=params['Cup'])
        scl_k = SquareWave(period=300, amp=0, offset=params['Clow'])
        #scu_k = SquareWave(period=300, amp=2.5, offset=27.5)
        #scl_k = SquareWave(period=300, amp=-2.5, offset=20)


    #====================================

    #torch.manual_seed(0)

    # Define dynamics and parameters
    sys = psl.systems['LinearSimpleSingleZone']()

    # simulate the building model over 1000 timesteps
    nsim = 8000
    sim = sys.simulate(nsim=nsim, x0=sys.get_x0(), U=sys.get_U(nsim + 1))
    # plot open-loop response
    pltOL(Y=sim['Y'], X=sim['X'], U=sim['U'], D=sim['D'])

    # extract exact state space model matrices:
    A = torch.tensor(sys.params[2]['A'])
    B = torch.tensor(sys.params[2]['Beta'])
    C = torch.tensor(sys.params[2]['C'])
    E = torch.tensor(sys.params[2]['E'])

    # get control action bounds
    umin = torch.tensor(sys.umin)
    umax = torch.tensor(sys.umax)

    # problem dimensions
    nx = sys.nx  # number of states
    nu = sys.nu  # number of control inputs
    nd = E.shape[1]  # number of disturbances
    nd_obsv = sys.nd  # number of observable disturbances
    ny = sys.ny  # number of controlled outputs
    nref = sys.ny  # number of references
    y_idx = 3

    """
        # # #  Dataset 
        """
    nsteps = params['nstep']  # prediction horizon
    n_samples = params['batchsize']  # number of sampled scenarios

    # range for lower comfort bound
    x_min = params['Clow']
    x_max = params['Cup']
    #  sampled references for training the policy
    list_xmin = [x_min + (x_max - x_min) * torch.rand(1, 1) * torch.ones(nsteps, nref)
                 for k in range(n_samples)]
    xmin = torch.cat(list_xmin)
    batched_ymin = xmin.reshape([n_samples, nsteps, nref])
    batched_ymax = batched_ymin + 2.0
    # get sampled disturbance trajectories from the simulation model
    list_dist = [torch.tensor(sys.get_D(nsteps))
                 for k in range(n_samples)]
    batched_dist = torch.stack(list_dist, dim=0)
    # get sampled initial conditions
    list_x0 = [torch.tensor(sys.get_x0().reshape(1, nx))
               for k in range(n_samples)]
    batched_x0 = torch.stack(list_x0, dim=0)
    # Training dataset
    train_data = DictDataset({'x': batched_x0,
                              'ymin': batched_ymin,
                              'ymax': batched_ymax,
                              'd': batched_dist},
                             name='train')

    # references for dev set
    list_xmin = [x_min + (x_max - x_min) * torch.rand(1, 1) * torch.ones(nsteps, nref)
                 for k in range(n_samples)]
    xmin = torch.cat(list_xmin)
    batched_ymin = xmin.reshape([n_samples, nsteps, nref])
    batched_ymax = batched_ymin + 2.0
    # get sampled disturbance trajectories from the simulation model
    list_dist = [torch.tensor(sys.get_D(nsteps))
                 for k in range(n_samples)]
    batched_dist = torch.stack(list_dist, dim=0)
    # get sampled initial conditions
    list_x0 = [torch.tensor(sys.get_x0().reshape(1, nx))
               for k in range(n_samples)]
    batched_x0 = torch.stack(list_x0, dim=0)
    # Development dataset
    dev_data = DictDataset({'x': batched_x0,
                            'ymin': batched_ymin,
                            'ymax': batched_ymax,
                            'd': batched_dist},
                           name='dev')

    # torch dataloaders
    batch_size = params['mini_batch']
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    # normalization statistics
    x_mean = batched_x0.mean(dim=[0, 1])
    x_std = batched_x0.std(dim=[0, 1])
    ymin_mean = batched_ymin.mean(dim=[0, 1])
    ymin_std = batched_ymin.std(dim=[0, 1])
    ymax_mean = batched_ymax.mean(dim=[0, 1])
    ymax_std = batched_ymax.std(dim=[0, 1])
    d_mean = batched_dist.mean(dim=[0, 1])
    d_std = batched_dist.std(dim=[0, 1])
    # concatenate means and variances
    means = torch.cat([x_mean, ymin_mean, ymax_mean, d_mean])
    stds = torch.cat([x_std, ymin_std, ymax_std, d_std])

    params['means'] = means
    params['stds'] = stds

    """
        # # #  System model and Control policy in Neuromancer
        """
    # state-space model of the building dynamics:
    #   x_k+1 =  A x_k + B u_k + E d_k
    xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
    state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')
    #   y_k = C x_k
    ynext = lambda x: x @ C.T
    output_model = Node(ynext, ['x'], ['y'], name='y=Cx')


    # get normalization layer to generate policy features
    def normalize_features(*inputs):
        x = torch.cat(inputs, dim=-1)
        return (x - means) / stds


    # features node
    features = Node(normalize_features, ['x', 'ymin', 'ymax', 'd'],
                    ['xi'], name='features')

    # neural net control policy
    params['mlp_in_size'] = nx + 2 * nref + nd
    net = blocks.MLP_bounds(insize=params['mlp_in_size'],
                            outsize=nu, hsizes=[params['hsizes'] for ii in range(params['nh'])],
                            nonlin=activations[params['activation']],
                            min=umin, max=umax)
    # symbolic policy
    policy = Node(net, ['xi'], ['u'], name='policy')

    # closed-loop system model
    cl_system = System([features, policy, state_model, output_model],
                       nsteps=nsteps,
                       name='cl_system')
    cl_system.show()

    """
        # # #  Differentiable Predictive Control objectives and constraints
        """
    # variables
    y = variable('y')
    u = variable('u')
    ymin = variable('ymin')
    ymax = variable('ymax')

    # objectives
    action_loss = params['Qu'] * (u == 0.0)  # energy minimization
    du_loss = params['Qdu'] * (u[:, :-1, :] - u[:, 1:, :] == 0.0)  # delta u minimization
    # # constraints
    state_lower_bound_penalty = params['Qc'] * (y > ymin)
    state_upper_bound_penalty = params['Qc'] * (y < ymax)
    # # objectives and constraints names for nicer plot
    action_loss.name = 'action_loss'
    du_loss.name = 'du_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    # list of constraints and objectives
    objectives = [action_loss, du_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty
    ]

    """
    # # #  Differentiable optimal control problem 
    """
    # data -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
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
    epochs = 200
    optimizer = torch.optim.AdamW(problem.parameters(), lr=params['lr'])
    #  Neuromancer trainer
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=epochs,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=epochs,
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)

    """
    Test Closed Loop System
    """
    print('\nTest Closed Loop System \n')
    nsteps_test = 2000
    # generate reference
    np_refs = psl.signals.step(nsteps_test + 1, 1, min=x_min, max=x_max, randsteps=5)
    ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test + 1, 1)
    ymax_val = ymin_val + 2.0
    # generate disturbance signal
    torch_dist = torch.tensor(sys.get_D(nsteps_test + 1)).unsqueeze(0)
    # initial data for closed loop simulation
    x0 = torch.tensor(sys.get_x0()).reshape(1, 1, nx)
    data = {'x': x0,
            'ymin': ymin_val,
            'ymax': ymax_val,
            'd': torch_dist}
    cl_system.nsteps = nsteps_test
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps_test, nu])
    Umax = umax * np.ones([nsteps_test, nu])
    Ymin = trajectories['ymin'].detach().reshape(nsteps_test + 1, nref)
    Ymax = trajectories['ymax'].detach().reshape(nsteps_test + 1, nref)
    # plot closed loop trajectories
    pltCL(Y=trajectories['y'].detach().reshape(nsteps_test, ny),
          R=Ymax,
          X=trajectories['x'].detach().reshape(nsteps_test + 1, nx),
          D=trajectories['d'].detach().reshape(nsteps_test + 1, nd),
          U=trajectories['u'].detach().reshape(nsteps_test, nu),
          Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)

    # Save file
    torch.save(best_model, params['model_name']+'_'+params['model_type']+'_'+str(params['policy_number'])+'_policy.pth')
    torch.save(params, params['model_name'] + '_' + params['model_type'] + '_' + str(params['policy_number']) + '_params.pth')





