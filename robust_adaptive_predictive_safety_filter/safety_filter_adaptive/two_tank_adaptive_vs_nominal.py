from casadi import *
import matplotlib.pyplot as plt
from neuromancer import psl
import torch
from closed_loop import DPC_control
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
import copy


class TwoTankDynamics():
    def __init__(self, model,dbar=0):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1., 1.]
        self.xmin = [0., 0.]
        self.x0 = [0.1, 0.1]

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = self.c1 * (1.0 - u[1]) * u[0] - self.c2 * sqrt(x[0])
        dhdt[1] = self.c1 * u[1] * u[0] + self.c2 * sqrt(x[0]) - self.c2 * sqrt(x[1])

        return dhdt


if __name__ == "__main__":




    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 750

    # ground truth system model
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model)
    nx = gt_model.nx
    ny = gt_model.nx
    nu = gt_model.nu
    nmodel = 2

    # Model bounds
    c1_max = 0.12
    c1_min = 0.07
    c2_max = 0.08
    c2_min = 0.03
    model_max = torch.tensor([[c1_max, c2_max]])
    model_min = torch.tensor([[c1_min, c2_min]])

    seed = 4

    #----------------------------
    # Reference trajectory
    r_nsteps = Nsim
    np_refs = psl.signals.step(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], randsteps=5, rng=np.random.default_rng(seed=seed))
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(  1, r_nsteps + 1)
    torch_ref = torch.cat([R, R], dim=0)

    #-------------------------------
    # Load DPC policies

    # Nominal
    policy_name = 'two_tank'
    version = 4
    policy = torch.load(policy_name+'_policy_'+str(version)+'.pth')
    policy_params = torch.load(policy_name + '_params_'+str(version)+'.pth')
    u_DPC = DPC_control(nu, policy, policy_params, {'r': torch_ref}, umin=model.umin, umax=model.umax)
    dt = policy_params['ts']

    # Adaptive
    policy_name = 'two_tank_adaptive'
    version = 4
    policy = torch.load(policy_name + '_policy_' + str(version) + '.pth')
    policy_params = torch.load(policy_name + '_params_' + str(version) + '.pth')
    a = torch.tensor(torch.mm(torch.tensor([[model.c1], [model.c2]]), torch.ones(1, Nsim + 1), ), dtype=torch.float32).reshape(nmodel, Nsim + 1)
    u_ADPC = DPC_control(nu, policy, policy_params, {'r': torch_ref, 'a': a}, umin=model.umin, umax=model.umax)
    dt = policy_params['ts']

    #---------------------------------
    # Setup and run simulation

    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    dummy_control = DummyControl()

    CL_nom = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_DPC, dt=dt, int_type=policy_params['integrator_type'])
    CL_adapt = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_ADPC, dt=dt, int_type=policy_params['integrator_type'])

    n_iter = 100
    c1 = model.c1
    c2 = model.c2
    e_nom = []
    e_adapt = []
    for kk in range(n_iter):

        if kk > 0:

            # Update model parameters
            c1 = c1_min + (c1_max - c1_min) * torch.rand(1)
            c2 = c2_min + (c2_max - c2_min) * torch.rand(1)
            a = torch.tensor(torch.mm(torch.tensor([[c1], [c2]]), torch.ones(1, Nsim + 1), ), dtype=torch.float32).reshape(nmodel, Nsim + 1)
            CL_nom.f.c1 = c1
            CL_nom.f.c2 = c2
            CL_adapt.f.c1 = c1
            CL_adapt.f.c2 = c2
            CL_adapt.u_nom.exogenous_variables = {'r': torch_ref, 'a': a}

        print('Model parameters: c1 = '+str(c1)+', c2 = '+str(c2))

        # Run simulations
        [X_nom, U_nom] = CL_nom.simulate(x0=x0, N=Nsim)
        [X_adapt, U_adapt] = CL_adapt.simulate(x0=x0, N=Nsim)

        # Compute error
        e_nom.append( np.sum([np.linalg.norm( X_nom[kk,:] - torch_ref.reshape(Nsim+1, nmodel).numpy()[kk,:] ) for kk in range(Nsim+1) ]))
        e_adapt.append( np.sum([np.linalg.norm( X_adapt[kk,:] - torch_ref.reshape(Nsim+1, nmodel).numpy()[kk,:] ) for kk in range(Nsim+1) ]))

        if kk < 5:

            t = [kk for kk in range(Nsim+1)]
            plt.figure()
            plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
            plt.plot(X_nom[:, 0], color='crimson', label='x1 (DPC nominal)', linewidth=2.0)
            plt.plot(X_nom[:, 1],  color='chartreuse', label='x2 (DPC nominal)', linewidth=2.0)
            plt.plot(X_adapt[:, 0], color = 'orange', linestyle='--', label='x1 (DPC adaptive)', linewidth=2.0)
            plt.plot(X_adapt[:, 1], color='magenta', linestyle='--', label='x2 (DPC adaptive)', linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Tank Level')
            plt.title('c1 = '+str(c1)+' , c2 = '+str(c2))
            plt.legend(loc='lower left')



            plt.figure()
            for ii in range(nu):
                plt.step(range(U_nom.shape[0]), 1.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
                plt.step(range(U_nom.shape[0]), 0.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
            plt.plot(U_nom[:,0], color='crimson', label="u1 (DPC nominal)", linewidth=2.0)
            plt.plot(U_nom[:, 1], color='chartreuse', label="u2 (DPC nominal)", linewidth=2.0)
            plt.plot(U_adapt[:, 0], color='orange', linestyle='--', label="u1 (DPC adaptive)", linewidth=2.0)
            plt.plot(U_adapt[:, 1], color='magenta', linestyle='--', label="u2 (DPC adaptive)", linewidth=2.0)
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.xlabel('Time step')
            plt.ylabel('Pump and Valve (Inputs)')
            plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))

    print(e_nom)
    plt.figure()
    plt.plot(e_nom, label = "nominal")
    plt.plot(e_adapt, label = "adaptive")

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Error')
    plt.legend()

    plt.show()