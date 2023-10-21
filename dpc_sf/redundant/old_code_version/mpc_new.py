"""
The MPC_Base class contains all the common parts of the CasADI MPC for the 3 control problems:

1. reference point tracking (MPC_Point_Ref)
2. reference trajectory tracking (MPC_Traj_Ref)
3. reference point tracking with obstacle constraints (MPC_Point_Ref_Obstacle)

"""

import casadi as ca
import numpy as np
from abc import ABC
import utils.pytorch_utils as ptu

class MPC_Base(ABC):
    def __init__(
            self, 
            N, 
            dt, 
            interaction_interval, 
            n, 
            m, 
            dynamics, 
            state_ub, 
            state_lb, 
            integrator_type='euler', # 'euler', 'RK4'
        ): 
        self.N = N      # horizon
        self.dt = dt    # sampling time
        self.ii = interaction_interval # how often the MPC will interact (10 = every 10 simulation steps)
        self.n = n      # num states
        self.m = m      # num inputs
        self.dynamics = dynamics
        self.integrator_type = integrator_type

        self.opti = ca.Opti()
        self.X = self.opti.variable(n, N*self.ii+1)
        self.U = self.opti.variable(m, N+1) # final input plays no role

        # perform first optimisation
        # --------------------------

        # define weighting matrices
        self.Q = ca.MX.zeros(self.n,self.n)
        self.Q[0,0] =   1 # x
        self.Q[1,1] =   1 # y
        self.Q[2,2] =   1 # z
        self.Q[3,3] =   0 # q0
        self.Q[4,4] =   0 # q1
        self.Q[5,5] =   0 # q2
        self.Q[6,6] =   0 # q3
        self.Q[7,7] =   1 # xdot
        self.Q[8,8] =   1 # ydot
        self.Q[9,9] =   1 # zdot
        self.Q[10,10] = 1 # p
        self.Q[11,11] = 1 # q
        self.Q[12,12] = 1 # r
        self.Q[13,13] = 0 # wM1
        self.Q[14,14] = 0 # wM2
        self.Q[15,15] = 0 # wM3
        self.Q[16,16] = 0 # wM4

        self.R = ca.MX.eye(self.m) * 1

        # constrain optimisation to the system dynamics over the horizon
        if self.integrator_type == 'euler':
            for k in range(self.N):
                input = self.U[:,k]
                sdot_k = self.dynamics(self.X[:,k], input)
                self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot_k * self.dt)

        elif self.integrator_type == 'RK4':
            for k in range(self.N):
                k1 = self.dynamics(self.X[:,k], self.U[:,k])
                k2 = self.dynamics(self.X[:,k] + self.dt / 2 * k1, self.U[:,k])
                k3 = self.dynamics(self.X[:,k] + self.dt / 2 * k2, self.U[:,k])
                k4 = self.dynamics(self.X[:,k] + self.dt * k3, self.U[:,k])
                x_next = self.X[:,k] + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
                self.opti.subject_to(self.X[:,k+1] == x_next)

        # apply state constraints
        for k in range(self.N):
            self.opti.subject_to(self.X[:,k] < state_ub)
            self.opti.subject_to(self.X[:,k] > state_lb)

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # solve
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6
        } # silence!
        self.opti.solver('ipopt', opts)

    def get_predictions(self):
        return self.opti.value(self.X), self.opti.value(self.U)
    
#     def dynamics_constraints(self, X, U, opt):
#         '''Defines the constraints related to the ode of the system dynamics
#         X: system state
#         U: system input
#         opt: Casadi optimization class'''
#         if self.integrator_type == 'RK4':
#             for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
    #             # Runge-Kutta 4 integration
    #             k1 = self.f(X[:, k], 0, U[:, k])
    #             k2 = self.f(X[:, k] + self.dT / 2 * k1, 0, U[:, k])
    #             k3 = self.f(X[:, k] + self.dT / 2 * k2, 0, U[:, k])
    #             k4 = self.f(X[:, k] + self.dT * k3, 0, U[:, k])
    #             x_next = X[:, k] + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #             opt.subject_to(X[:, k + 1] == x_next)  # close the gaps
# 
#         if self.integrator_type == 'Euler':
#             for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
    #             x_next = X[:, k] + self.dT * self.f(X[:, k], 0, U[:, k])
    #             opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

    
""" (why is this green in vscode?)
The MPC_Point_Ref will take in the current state and a single reference state and find the MPC
action which will minimise its cost.
"""

# class MPC_Point_Ref(MPC_Base):
#     def __init__(self, N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb, return_type):
#         super().__init__(N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb)
# 
#         self.return_type = return_type
# 
#         # define start condition (dummy)
#         state0 = np.array([0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375])
#         self.init = self.opti.parameter(n,1)
#         self.opti.set_value(self.init, state0)
#         self.opti.subject_to(self.X[:,0] == self.init)
# 
#         # define reference (dummy)
#         reference = state0
#         reference[0] = 2
#         self.ref = self.opti.parameter(n,1)
#         self.opti.set_value(self.ref, reference)
# 
#         # define the cost for this type of MPC
#         self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted
# 
#         # solve the mpc once, so that we can do it repeatedly in a method later
#         sol = self.opti.solve()
#     
#     def cost_discounted(self, state, reference, input):
#         state_error = reference - state
#         gamma = 0.99 # standard for RL 0.99
#         cost = ca.MX(0)
#         # lets get cost per timestep:
#         for k in range(self.N * self.ii + 1):
#             timestep_input = input[:,round(k/self.ii)]
#             timestep_state_error = state_error[:,k]
#             cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)*gamma**k
#         return cost
# 
#     def __call__(self, state, reference):
#         # point to point control, from state to stationary reference
# 
#         # define start condition
#         self.opti.set_value(self.init, state.tolist())
#         # define cost w.r.t reference
#         self.opti.set_value(self.ref, reference)
#         # solve
#         sol = self.opti.solve().value(self.U)[:,0]
#         if self.return_type == 'torch':
#             return ptu.from_numpy(sol)
#         elif self.return_type == 'numpy':
#             return sol     
    


"""  """

class MPC_Point_Ref(MPC_Base):
    def __init__(self, N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb, return_type='torch', obstacle=True, integrator_type='euler'):
        super().__init__(N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb, integrator_type=integrator_type)

        self.return_type = return_type
        self.obstacle = obstacle

        # define cylinder obstacle
        self.radius = 0.5
        self.x_pos = 1
        self.y_pos = 1

        # obstacle avoid at every step of the MPC 0 -> N inclusive
        if obstacle == True:
            # apply the constraint from 2 timesteps in the future as the quad has relative degree 2
            # to ensure it is always feasible!
            for k in range(interaction_interval*self.N-1):
                multiplier = 1 + k * dt * 0.5
                current_x, current_y = self.X[0,k+2], self.X[1,k+2]
                self.opti.subject_to(self.is_in_cylinder(current_x, current_y, multiplier))
                
        
        # define start condition (dummy)
        state0 = np.array([0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375])
        self.init = self.opti.parameter(n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # define reference (dummy)
        reference = state0
        reference[0] = 2
        self.ref = self.opti.parameter(n,1)
        self.opti.set_value(self.ref, reference)

        # define the cost for this type of MPC
        self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted

        # solve the mpc once, so that we can do it repeatedly in a method later
        sol = self.opti.solve()

    # cylinder enlargens in the future to stop us colliding with cylinder,
    # mpc expects it to be worse than it is.
    def is_in_cylinder(self, X, Y, multiplier):
        return self.radius ** 2 * multiplier <= (X - self.x_pos)**2 + (Y - self.y_pos)**2
    
    def distance2cylinder(self, state):
        return np.sqrt((state[0] - self.x_pos)**2 + (state[1] - self.y_pos)**2) - self.radius
    
    def cost_discounted(self, state, reference, input):
        state_error = reference - state
        gamma = 0.99 # standard for RL 0.99
        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N * self.ii + 1):
            timestep_input = input[:,round(k/self.ii)]
            timestep_state_error = state_error[:,k]

            # standard quadratic cost
            cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)*gamma**k
            
            # soft constraint of the cylinder, distance to cylinder
            # if self.obstacle:
            #     dist = self.distance2cylinder(timestep_state)
            #     cost += 1e-2 / (ca.fmax(0,dist) + 1e-16)

        return cost
    
    def __call__(self, state, reference):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state.tolist())

        # THIS WARM START FEATURE IS KILLER
        # warm start all future states towards the reference in a straight line
        # try linear set of points between ref and state
        reference_stack = np.array([np.linspace(state[i], reference[i], self.N) for i in range(self.n)])
        self.opti.set_initial(self.X[:,1:], reference_stack)

        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference.tolist())
        # solve
        sol = self.opti.solve().value(self.U)[:,0]
        if self.return_type == 'torch':
            return ptu.from_numpy(sol)
        elif self.return_type == 'numpy':
            return sol     

"""  """

class MPC_Traj_Ref(MPC_Base):
    def __init__(self, N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb, reference_traj, return_type='torch', integrator_type='euler'):
        super().__init__(N, dt, interaction_interval, n, m, dynamics, state_ub, state_lb, integrator_type=integrator_type)

        self.traj = reference_traj
        self.return_type = return_type
        # define start condition (dummy)
        state0 = [0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]
        self.init = self.opti.parameter(n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # define dummy reference (n x N)
        reference = np.zeros([n,N*interaction_interval+1])
        self.ref = self.opti.parameter(n,N*interaction_interval+1)
        self.opti.set_value(self.ref, reference)

        # cost function
        self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted

        # solve the mpc once, so that we can do it repeatedly in a method later
        sol = self.opti.solve()

    def cost_discounted(self, state, reference, input):
        state_error = reference - state
        gamma = 0.99 # standard for RL 0.99
        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N * self.ii + 1):
            timestep_input = input[:,round(k/self.ii)]
            timestep_state_error = state_error[:,k]
            cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)*gamma**k
        return cost

    def __call__(self, state, time):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state.tolist())

        # THIS WARM START FEATURE IS KILLER
        # warm start all future states towards the reference in a straight line
        # try linear set of points between ref and state
        ref = self.traj(time + self.N * self.dt)
        reference_stack = np.array([np.linspace(state[i], ref[i], self.N) for i in range(self.n)])
        self.opti.set_initial(self.X[:,1:], reference_stack)

        # define cost w.r.t reference
        times = np.arange(time, time+self.dt*(self.N*self.ii+1), self.dt)
        # get the reference for these times
        reference = []
        for k in range(self.N*self.ii+1):
            reference.append(self.traj(times[k]))
        reference = np.vstack(reference).T
        self.opti.set_value(self.ref, reference)

        # solve
        sol = self.opti.solve().value(self.U)[:,0]
        if self.return_type == 'torch':
            return ptu.from_numpy(sol)
        elif self.return_type == 'numpy':
            return sol