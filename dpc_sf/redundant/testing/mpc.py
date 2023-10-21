"""
The MPC_Base class contains all the common parts of the CasADI MPC for the 3 control problems:

1. reference point tracking (MPC_Point_Ref)
2. reference trajectory tracking (MPC_Traj_Ref)
3. reference point tracking with obstacle constraints (MPC_Point_Ref_Obstacle)

"""

import casadi as ca
import numpy as np
from abc import ABC
from utils.stateConversions import sDes2state

class MPC_Base(ABC):
    def __init__(self, N, sim_Ts, interaction_interval, n, m, quad): 
        self.N = N      # horizon
        self.sim_Ts = sim_Ts    # sampling time
        self.ii = interaction_interval # how often the MPC will interact (10 = every 10 simulation steps)
        self.n = n      # num states
        self.m = m      # num inputs
        self.dynamics = quad.casadi_state_dot
        self.quad = quad

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
        self.Q[3,3] =   1 # q0
        self.Q[4,4] =   1 # q1
        self.Q[5,5] =   1 # q2
        self.Q[6,6] =   1 # q3
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
        k = 0
        for i in range(self.N):
            # make sure the system follows its dynamics
            input = self.U[:,i]
            for j in range(self.ii):
                sdot_k = self.dynamics(self.X[:,k], input)
                self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot_k * self.sim_Ts)

                # constrain the states of the system at every time step
                self.opti.subject_to(self.X[:,k] < quad.casadi_constraints['upper'])
                self.opti.subject_to(self.X[:,k] > quad.casadi_constraints['lower'])

                k += 1

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # solve
        opts = {'ipopt.print_level':0, 'print_time':0} # silence!
        self.opti.solver('ipopt', opts)

    def get_predictions(self):
        return self.opti.value(self.X), self.opti.value(self.U)
    
""" (why is this green in vscode?)
The MPC_Point_Ref will take in the current state and a single reference state and find the MPC
action which will minimise its cost.
"""

class MPC_Point_Ref(MPC_Base):
    def __init__(self, N, sim_Ts, interaction_interval, n, m, quad):
        super().__init__(N, sim_Ts, interaction_interval, n, m, quad)

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

    def __call__(self, state, reference):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)
        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference)
        # solve
        return self.opti.solve()


"""  """

class MPC_Point_Ref_Obstacle(MPC_Base):
    def __init__(self, N, sim_Ts, interaction_interval, n, m, quad):
        super().__init__(N, sim_Ts, interaction_interval, n, m, quad)

        # define cylinder obstacle
        radius = 0.5
        x_pos = 1
        y_pos = 1

        # obstacle avoid at every step of the MPC 0 -> N inclusive
        for k in range(interaction_interval*self.N+1):
            self.opti.subject_to(
                radius**2 <=
                (self.X[0,k] - x_pos)**2 +
                (self.X[1,k] - y_pos)**2
            )

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
    
    def __call__(self, state, reference):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)
        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference)
        # solve
        return self.opti.solve()        

"""  """

class MPC_Traj_Ref(MPC_Base):
    def __init__(self, N, sim_Ts, interaction_interval, n, m, quad, reference_traj):
        super().__init__(N, sim_Ts, interaction_interval, n, m, quad)

        self.traj = reference_traj

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
        self.opti.set_value(self.init, state)

        # define cost w.r.t reference
        times = np.arange(time, time+self.sim_Ts*(self.N*self.ii+1), self.sim_Ts)
        # get the reference for these times
        reference = []
        for k in range(self.N*self.ii+1):
            sDes = self.traj.desiredState(times[k], self.sim_Ts, self.quad)
            reference.append(sDes2state(sDes))
        reference = np.vstack(reference).T
        self.opti.set_value(self.ref, reference)

        # solve
        return self.opti.solve()