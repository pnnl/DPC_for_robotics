
import numpy as np
import traceback
from neuromancer.modules import blocks
import torch
from tqdm import tqdm
from neuromancer.modules.activations import activations
from predictive_safety_filter.safety_filter import integrate


class ClosedLoopSystem():
    '''Closed-loop simulation of given dynamics f with control u'''

    def __init__(self, f, u, u_nom, dt, int_type='Euler'):
        '''
        f: Function defining system dynamics f(x,k,u)
        u: Control that closes the loop in the system, must have member compute_control_step(unom, x,k)
        unom: Nominal control, function of x and k, unom(x,k)
        dt: Sampling time for Euler integration
        int_type: Integration type, e.g., Euler, RK4, cont
        '''
        self.f = f
        self.u = u
        self.u_nom = u_nom
        self.dt = dt
        self.int_type = int_type

    def simulate(self, x0, N):
        '''Simulates the closed-loop system starting from state x0 for N steps'''

        # Initialize states
        x = x0
        nx = len(x)
        X = x.flatten()


        # Iterate through simulation horizon
        for kk in tqdm(range(N)):
            # Compute control and integrate to compute the next state
            try:
                uk = self.u.compute_control_step(self.u_nom, x, kk)
            except:

                print(traceback.format_exc())
                print('WARNING: Control computation failed, exiting closed-loop simulation')
                return X,U
            #print('uk = '+str(uk))
            x = integrate(self.f, x, kk, uk, self.dt, self.int_type)

            # For the first time step, get the size of u and setup U to collect all control inputs
            if kk == 0:
                try:
                    nu = len(uk)
                except:
                    nu = 1
                    uk = np.array([uk])
                U = uk.reshape((1, nu))
            if nu == 1:
                uk = np.array([uk])

            # Store all states and variables
            X = np.vstack((X, np.array(x).flatten()))
            if kk > 0:
                U = np.vstack((U, uk.reshape((1, nu))))

        # Compute control for last state
        try:
            uk = self.u.compute_control_step(self.u_nom, x, kk)
        except:

            print(traceback.format_exc())
            print('WARNING: Control computation failed, exiting closed-loop simulation')
            return X, U

        U = np.vstack((U, np.array([uk]).flatten().reshape((1, nu))))

        if self.u.infeasibilities:
            print('Infeasibilities of safety filter occured at the following time steps: '+str(self.u.infeasibilities))
        else:
            print('No infeasibilities occurred.')

        return X, U

class ClosedLoopSystemAdaptive():
    '''Closed-loop simulation of given dynamics f with control u'''

    def __init__(self, f, u, u_nom, adaptive_estimator, dt, int_type='Euler'):
        '''
        f: Function defining system dynamics f(x,k,u)
        u: Control that closes the loop in the system, must have member compute_control_step(unom, x,k)
        unom: Nominal control, function of x and k, unom(x,k)
        adaptive_estimator: Adaptive Estimator that takes current state and control action and returns current estimate of model parameters (theta)
        dt: Sampling time for Euler integration
        int_type: Integration type, e.g., Euler, RK4, cont
        '''
        self.f = f
        self.u = u
        self.u_nom = u_nom
        self.adaptive_estimator = adaptive_estimator
        self.dt = dt
        self.int_type = int_type

    def simulate(self, x0, N):
        '''Simulates the closed-loop system starting from state x0 for N steps'''

        # Initialize states
        self.adaptive_estimator.cl_reset()
        x = x0
        nx = len(x)
        X = x.flatten()
        theta = self.adaptive_estimator.theta_val
        Theta = theta.flatten()


        # Iterate through simulation horizon
        for kk in tqdm(range(N)):
            # Compute control and integrate to compute the next state
            try:
                uk = self.u.compute_control_step(self.u_nom, x, kk)
            except:

                print(traceback.format_exc())
                print('WARNING: Control computation failed, exiting closed-loop simulation')
                return X,U

            # Compute estimator of model parameters and update nominal control
            theta, Ld = self.adaptive_estimator(x, kk, uk)
            len_theta = len(theta)
            theta_vec = np.array(theta.flatten().reshape((len_theta, 1)))
            a = torch.tensor(torch.mm(torch.tensor(theta_vec.tolist()), torch.ones(1, N + 1), ),
                             dtype=torch.float32).reshape(len_theta, N + 1)
            self.u_nom.update_exogenous_variables({'a':a})

            x = integrate(self.f, x, kk, uk, self.dt, self.int_type)

            # For the first time step, get the size of u and setup U to collect all control inputs
            if kk == 0:
                try:
                    nu = len(uk)
                except:
                    nu = 1
                    uk = np.array([uk])
                U = uk.reshape((1, nu))
            if nu == 1:
                uk = np.array([uk])

            # Store all states and variables
            X = np.vstack((X, np.array(x).flatten()))
            Theta = np.vstack((Theta, np.array(theta).flatten()))
            if kk > 0:
                U = np.vstack((U, uk.reshape((1, nu))))

        # Compute control for last state
        try:
            uk = self.u.compute_control_step(self.u_nom, x, kk)
        except:

            print(traceback.format_exc())
            print('WARNING: Control computation failed, exiting closed-loop simulation')
            return X, U
        U = np.vstack((U, uk.reshape((1, nu))))

        if self.u.infeasibilities:
            print('Infeasibilities of safety filter occured at the following time steps: '+str(self.u.infeasibilities))
        else:
            print('No infeasibilities occurred.')

        return X, U, Theta



class DummyControl:
    '''Used to create same control class as in the safety_filter for an arbitrary control'''

    def __init__(self):
        self.name = 'dummy'
        self.infeasibilities = []

    def compute_control_step(self, unom, x, k):
        '''Given unom, x, k output the unom value evaluated at x,k ie. unom(x,k)
        unom: function taking variables x,k and outputing nominal contrl value
        x: current state
        k: current time step
        '''

        return unom(x, k)

class DummyControlAdaptive:
    '''Used to create same control class as in the safety_filter for an arbitrary control'''

    def __init__(self):
        self.name = 'dummy'
        self.infeasibilities = []

    def compute_control_step(self, unom, x, k, theta):
        '''Given unom, x, k output the unom value evaluated at x,k ie. unom(x,k)
        unom: function taking variables x,k and outputing nominal contrl value
        x: current state
        k: current time step
        '''

        return unom(x, k, theta)

def filter_policy(policy):

    new_policy = {}

    for key, val in policy.items():

        if 'block' not in key:
            # Find the index of the substring to remove
            index = key.find('callable.')
            if index != -1:
                # Remove the substring and everything up to it
                new_key = key[index + len('callable.'):]
                new_policy[new_key] = val
            else:
                new_policy[key] = val

    return new_policy



class DPC_control():
    def __init__(self, nu, policy, params, exogenous_variables={}, umin=[], umax=[], norm_mean=None, norm_stds=None):

        umin = np.array(umin).flatten()
        umax = np.array(umax).flatten()
        self.means = norm_mean
        self.stds = norm_stds
        if not len(umin) == 0:
            # self.policy = blocks.MLP_bounds(params['mlp_in_size'],nu, bias=True,
            #                         linear_map=torch.nn.Linear,
            #                         nonlin=torch.nn.ReLU, #activations['gelu'], #
            #                         hsizes=[params['hsizes'] for ii in range(params['nh'])],
            #                         min=torch.from_numpy(umin),
            #                         max=torch.from_numpy(umax),
            #                         method=params['bound_method'],
            #                                 )
            self.policy = blocks.MLP_bounds(params['mlp_in_size'], nu, bias=True,
                                            nonlin=activations['gelu'], #torch.nn.ReLU,  #
                                            hsizes=[params['hsizes'] for ii in range(params['nh'])],
                                            min=torch.from_numpy(umin),
                                            max=torch.from_numpy(umax),
                                            )
        else:
            self.policy = blocks.MLP(params['mlp_in_size'], nu, bias=True,
                                            linear_map=torch.nn.Linear,
                                            nonlin=activations['gelu'], #torch.nn.ReLU,
                                            hsizes=[params['hsizes'] for ii in range(params['nh'])],
                                            )


        self.policy.load_state_dict(filter_policy(policy))
        self.exogenous_variables = exogenous_variables


    def __call__(self, x, k):
        '''Given x, k output the policy value evaluated at x,k '''

        X = [torch.from_numpy(x.flatten()).type(torch.float)]

        for dx in self.exogenous_variables.values():
            X.append(dx[:,k])

        X = torch.cat(X, dim=0)


        # Normalize features if means, stds data provided
        if not self.means is None:
            X = self.normalize_features(X)

        u = self.policy(X)

        return u.detach().numpy().reshape((-1,1))

    def normalize_features(self, *inputs):
        x = torch.cat(inputs, dim=-1)
        return (x - self.means) / self.stds

    def update_exogenous_variables(self, new_ex):
        '''Update exogenous variables terms

        new_ex: dictionary of exogenous variables whose key must be in self.exogenous_variables

        '''

        for key, val in new_ex.items():
            if key in self.exogenous_variables:
                self.exogenous_variables[key] = val



class NNModel():
    def __init__(self, nx, nu, modelpth, params):

        self.nx = nx
        self.nu = nu
        self.problem = torch.load(modelpth)
        self.params = params

        # Set nsteps to 1
        self.problem.nodes[0].nsteps = 1

    def __call__(self, x, k, u):
        x_input = self.params['x_input_key']
        u_input = self.params['u_input_key']
        data = {x_input: torch.from_numpy(x.flatten().reshape((1, 1, self.nx))).type(torch.float),
                u_input: torch.from_numpy(u.flatten().reshape((1, 1, self.nu))).type(torch.float)}

        yout = self.problem.nodes[0](data)

        return yout[x_input][0,-1,:].detach().numpy().reshape((self.nx,1))
