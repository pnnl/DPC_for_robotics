"""
This code now performs the DPC on top of the low level control. Still failure to converge

"""
import numpy as np
import torch 
from tqdm import tqdm
import copy

from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils.normalisation import normalize_nm, denormalize_nm, denormalize_np
from dpc_sf.utils.random_state import random_state
from dpc_sf.utils.animation import Animator
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.utils.random_state import random_state
from dpc_sf.control.dpc.redundant2.policy import Policy
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss


import matplotlib.pyplot as plt
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
    
class Vis():
    def __init__(self) -> None:
        self.x = []
        self.u = []
        self.r = []
        self.t = []

    def save(self, x, u, r, t):
        self.x.append(x)
        self.u.append(u)
        self.r.append(r)
        self.t.append(t)

    def animate(self, x_p=None, drawCylinder=False):
        animator = Animator(
            states=np.vstack(self.x), 
            times=np.array(self.t), 
            reference_history=np.vstack(self.r), 
            reference=self.reference, 
            reference_type=self.reference.type, 
            drawCylinder=drawCylinder,
            state_prediction=x_p
        )
        animator.animate() # does not contain plt.show()        

if __name__ == '__main__':

    Ts = 0.01 # simulation timestep
    # nsim = 1001 # number of STEPS taken in the statistical sampling stage of the system init
    save_dir = 'logs/data'
    test_closed_loop = False
    test_sampled_data = False
    test_trained_closed_loop = True
    normalize = True
    lr = 0.001 # 0.001
    epochs = 50
    patience = 20
    train_nsteps = 50 # number of steps to take in each training rollout
    nx = 13
    nu = 3
    num_train_samples = 1_000
    num_dev_samples = 1_000
    iterations = 2
    hsize = 64
    n_layers = 3
    include_actuators = False

    """
    Dataset
    -------

    The DPC algorithm requires a dataset of starting points and references. 

    We must generate a dictionary with the keys: ['X', 'R']
        - X is the random initial conditions, which I choose to be about a trajectory reference I have created
            shape: (batch size, 1, state size)
        - R is a time varying reference
            shape: (batch size, train rollout length (train_nsteps), state size)
    """
    def get_data(num_points=100000, nsteps=3, reference=waypoint_reference('wp_traj', average_vel=1.0), include_actuators=False):
        reference.include_actuators = include_actuators
        reference_end_time = 19.23 # really 19.23 for average_vel at 1.0
        R = []
        X = []
        print(f"collecting {len(np.linspace(start=0.0, stop=reference_end_time, num=num_points))} samples from the reference trajectory...")
        for t in tqdm(np.linspace(start=0.0, stop=reference_end_time, num=num_points)):
            # get the reference state at the given time
            r = reference(t)
            R.append(r)
            # get the random initial state centered around the reference at that time
            # we set the initial quaternion to always be pointed forwards for now
            x = random_state(include_actuators=include_actuators) + r
            x[3:7] = random_state(include_actuators=include_actuators)[3:7]
            X.append(x)

        R = np.stack(R).astype(np.float32)
        R = R.reshape(R.shape[0], 1, R.shape[1])

        # we repeat R nsteps times for waypoint tracking training
        R = np.concatenate([R]*nsteps, axis=1)
        
        X = np.stack(X).astype(np.float32)
        X = X.reshape(X.shape[0], 1, X.shape[1])

        return {'R': R, 'X': X}

    # the reference about which we generate data
    R = waypoint_reference('wp_traj', average_vel=1.0, include_actuators=include_actuators)

    # Normalisation
    # -------------

    def get_stats(train_data):
        # gather statistics on the training data
        train_data_stats = {}
        for key, value in train_data.items():
            train_data_stats[key] = {
                'mean': [],
                'var': []
            }
            for state_idx in range(nx):
                mean = value[:,:,state_idx].mean()
                var = value[:,:,state_idx].var()
                train_data_stats[key]['mean'].append(mean)
                train_data_stats[key]['var'].append(var)
            train_data_stats[key]['mean'] = np.stack(train_data_stats[key]['mean'])
            train_data_stats[key]['var'] = np.stack(train_data_stats[key]['var'])

        # want to normalise both on the starting point dataset
        mean = train_data_stats['X']['mean']
        var = train_data_stats['X']['var']

        # further we don't want to normalise the quaternion for fear of breaching limits
        mean[3:7] = np.array([0,0,0,0])
        var[3:7] = np.array([1,1,1,1])

        return mean, var

    def get_dictdataset(train_data, dev_data, mean=None, var=None, normalize=True):
        # normalise the train, dev data
        if normalize:
            train_data = {key: normalize_nm(value, means=mean, variances=var) for key, value in train_data.items()}
            dev_data = {key: normalize_nm(value, means=mean, variances=var) for key, value in dev_data.items()}

        # put datasets into torch
        train_data = {key: ptu.from_numpy(value) for key, value in train_data.items()}
        dev_data = {key: ptu.from_numpy(value) for key, value in dev_data.items()}

        # Put data dictionaries into NM DictDatasets
        train_data = DictDataset(train_data, name='train')
        dev_data = DictDataset(dev_data, name='dev')

        return train_data, dev_data
    
    def get_loader(train_data, dev_data):

        # Put NM Dictionaries into Torch DataLoaders
        # Shuffle false as we have already randomised start positions and want them lining up with their end positions
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_data.datadict['R'].shape[0], collate_fn=train_data.collate_fn, shuffle=False)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=dev_data.datadict['R'].shape[0], collate_fn=dev_data.collate_fn, shuffle=False)

        return train_loader, dev_loader

    train_data = get_data(num_points=num_train_samples, nsteps=train_nsteps, reference=R, include_actuators=include_actuators)
    dev_data = get_data(num_points=num_dev_samples, nsteps=train_nsteps, reference=R, include_actuators=include_actuators)

    mean, var = get_stats(train_data=train_data)

    train_data, dev_data = get_dictdataset(train_data, dev_data, mean=mean, var=var, normalize=True)
    train_loader, dev_loader = get_loader(train_data, dev_data)

    print("done")

    """
    Model
    -----

    Now that I am not using the PSL system, normalisation will have to be done manually.
    This is achieved through the RL normalisation statistics that I have previously saved.

    These are contained in the following functions:
        - normalise_nm
        - normalise_dict
        - denormalize_nm
    """
    sys = QuadcopterDPC(
        params=quad_params,
        nx=nx,
        nu=nu,
        ts=Ts,
        normalize=normalize,
        mean=mean,
        var=var,
        include_actuators=include_actuators
    )
    sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'])

    """
    Policy
    ------

    The policy needs to be clipped in a normalisation bound for effective learning. This
    was done using a new MLP class built during Soumyas submersible project.

    Further this MLP outputs desired rotor rotational rates, and a proportional controller
    decides on the true input to adjust the rates to be the desired ones.
    """
    if not normalize:
        w_mean=None
        w_var=None

    policy = Policy(
        insize = sys.nx * 2, # state and reference
        outsize = sys.nu,        # input size
        hsize=hsize,
        n_layers=n_layers,
        mlp_umin=-1,                # minimum output of MLP within policy
        mlp_umax=0.90,                 # maximum output of MLP within policy
        Ts=Ts,
        normalize=normalize,        # whether or not policy expects a normalized state
        mean=mean,
        var=var,
        bs = num_train_samples
    )
    policy_node = Node(policy, input_keys=['X', 'R'], output_keys=['U'])

    """
    Define NeuroMANCER System
    -------------------------
    """
    cl_system = System([policy_node, sys_node], nsteps=train_nsteps)#, init_func=lambda x: x)

    if test_closed_loop:
        print(f"testing closed loop control...")
        # test call of closed loop system
        nstep = 3
        if normalize:
            if include_actuators:
                data = {
                    'X': ptu.from_numpy(normalize_nm(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32), means=mean, variances=var)),
                    'R': torch.concatenate([ptu.from_numpy(normalize_nm(R(5)[None,:][None,:].astype(np.float32), means=mean, variances=var))]*nstep, axis=1)
                }
            else:
                data = {
                    'X': ptu.from_numpy(normalize_nm(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32), means=mean, variances=var)),
                    'R': torch.concatenate([ptu.from_numpy(normalize_nm(R(5)[None,:][None,:].astype(np.float32), means=mean, variances=var))]*nstep, axis=1)
                }                
        else:
            data = {
                'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
                'R': torch.concatenate([ptu.from_numpy(R(5)[None,:][None,:].astype(np.float32))]*nstep, axis=1)
            }
        cl_system.nodes[0].callable.vel_sp_2_w_cmd.bs = data['X'].shape[1]
        cl_system.nodes[0].callable.vel_sp_2_w_cmd.reset() 
        cl_system.nsteps = nstep
        cl_system(data)
        print("done")

        # reset batch expectations of low level control
        cl_system.nodes[0].callable.vel_sp_2_w_cmd.bs = num_train_samples
        cl_system.nodes[0].callable.vel_sp_2_w_cmd.reset() 

    if test_sampled_data:
        print(f"testing sampled data on closed loop system in inference...")
        cl_system.nsteps = train_nsteps
        test_data = copy.deepcopy(train_data.datadict)
        cl_system(test_data)
        print("animating...")
        # lets animate this test data on the untrained system
        animator = Animator(
            states=ptu.to_numpy(test_data['X'][0,...]), 
            times=np.linspace(0, train_nsteps * Ts, train_nsteps), 
            reference_history=np.zeros([train_nsteps, nx]), 
            reference=R, 
            reference_type=R.type, 
            drawCylinder=False,
            state_prediction=None
        )
        animator.animate() # does not contain plt.show()      
        print("done")

    """
    Define Optimisation Problem
    ---------------------------

    Now we must define the NM variables which we want to optimise and setup the 
    cost function, learning rate, trainer object etc...
    """
    u = variable('U')
    x = variable('X')
    r = variable('R')

    # action_loss = 0.001 * (u == 0.0) ^ 2

    # we only care about matching references after the first one
    # we also only care about first 13 states (not rotor rotational rates)
    tracking_loss = 5.0 * (x[:,1:,10:13] == r[:,:,10:13]) ^ 2
    # tracking_loss = 5.0 * (x[:,1:,7:10] == r[:,:,7:10]) ^ 2


    loss = PenaltyLoss(
        objectives=[tracking_loss], 
        constraints=[]
    )

    problem = Problem(
        nodes = [cl_system],
        loss = loss
    )
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)



    # Train model with prediction horizon of train_nsteps
    for i in range(iterations):
        print(f'training with prediction horizon: {train_nsteps}, lr: {lr}')

        trainer = Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            epochs=epochs,
            patience=patience,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
        )

        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.0
        # train_nsteps *= 2

        # update the prediction horizon
        cl_system.nsteps = train_nsteps

        # get another set of data
        train_data = get_data(num_points=num_train_samples, nsteps=train_nsteps, reference=R)
        dev_data = get_data(num_points=num_dev_samples, nsteps=train_nsteps, reference=R)
        train_data, dev_data = get_dictdataset(train_data, dev_data, mean=mean, var=var, normalize=True)
        train_loader, dev_loader = get_loader(train_data, dev_data)

        # apply new training data and learning rate to trainer
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data
        optimizer.param_groups[0]['lr'] = lr

    if test_trained_closed_loop:

        # Test best model with prediction horizon of 50
        problem.load_state_dict(best_model)
        nstep = 50
        if normalize:
            data = {
                'X': ptu.from_numpy(normalize_nm(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32), means=mean, variances=var)),
                'R': torch.concatenate([ptu.from_numpy(normalize_nm(R(5)[None,:][None,:].astype(np.float32), means=mean, variances=var))]*nstep, axis=1)
            }
        else:
            data = {
                'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
                'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1)
            }

        cl_system.nsteps = nstep
        trajectories = cl_system(data)
        trajectories = {key: denormalize_nm(ptu.to_numpy(value)) for key, value in trajectories.items()}

        x_p = None
        drawCylinder = False

        t = np.linspace(0, nstep*Ts, nstep)

        animator = Animator(
            states=denormalize_np(ptu.to_numpy(trajectories['X'].squeeze())), 
            times=t, 
            reference_history=denormalize_np(ptu.to_numpy(trajectories['R'].squeeze())), 
            reference=R, 
            reference_type='wp_p2p', 
            drawCylinder=drawCylinder,
            state_prediction=x_p
        )
        animator.animate() # does not contain plt.show()    

    print('fin')