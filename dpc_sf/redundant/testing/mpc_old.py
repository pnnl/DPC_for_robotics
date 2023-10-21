import casadi as ca
import numpy as np
from utils.stateConversions import sDes2state

# I should probably have an MPC base class at this point lol - I just hate seeing inheritance in 
# someone elses code personally, I immediately get scared.

class MPC_Point_Ref:
    def __init__(self, N, Ts, n, m, dynamics, state_constraints):
        self.N = N      # horizon
        self.Ts = Ts    # sampling time
        self.n = n      # num states
        self.m = m      # num inputs
        self.dynamics = dynamics

        self.opti = ca.Opti()
        self.X = self.opti.variable(n, N+1)
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

        # define start condition (dummy)
        state0 = [0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]
        self.init = self.opti.parameter(n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # constrain optimisation to the system dynamics over the horizon
        for k in range(self.N):
            # make sure the system follows its dynamics
            sdot = self.dynamics(self.X, self.U)
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot[:,k] * self.Ts)

            # constrain the states of the system at every time step
            self.opti.subject_to(self.X[:,k] < state_constraints['upper'])
            self.opti.subject_to(self.X[:,k] > state_constraints['lower'])

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # define state constraints (obstacles/flight envelope)
        # self.obstacle = self.opti.parameter() # so it is changeable later
        # self.opti.subject_to()

        # define dummy reference
        reference = state0
        reference[0] = 2
        self.ref = self.opti.parameter(n,1)
        self.opti.set_value(self.ref, reference)

        # Choose cost function
        # self.opti.minimize(self.cost_no_discount(self.X, self.ref, self.U)) # not discounted
        self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted

        # solve
        opts = {'ipopt.print_level':0, 'print_time':0} # silence!
        self.opti.solver('ipopt', opts)
        sol = self.opti.solve()

    def cost_no_discount(self, state, reference, input):
        # in this cost we penalise the state error at every timestep equally without discount
        state_error = reference - state

        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N):
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input

        return cost
    
    def cost_discounted(self, state, reference, input):
        state_error = reference - state
        gamma = 0.99 # standard for RL 0.99
        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N):
            timestep_input = input[:,k]
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
    
class MPC_Point_Ref_Obstacle:
    def __init__(self, N, Ts, n, m, dynamics, state_constraints):
        self.N = N      # horizon
        self.Ts = Ts    # sampling time
        self.n = n      # num states
        self.m = m      # num inputs
        self.dynamics = dynamics

        self.opti = ca.Opti()
        self.X = self.opti.variable(n, N+1)
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

        # define start condition (dummy)
        state0 = [0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]
        self.init = self.opti.parameter(n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # define what we need for an ellipse obstacle
        # adapted from: https://github.com/pnnl/deps_arXiv2020/blob/master/Example_4/2D_obstacle_avoidance_csadi.py
        radius = 0.1
        x_pos = 1
        y_pos = 1
        #z_pos = 0.5
        x_stretch = 1
        y_stretch = 1
        #z_stretch = 1

        # constrain optimisation to the system dynamics over the horizon
        for k in range(self.N):
            # make sure the system follows its dynamics
            sdot = self.dynamics(self.X, self.U)
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot[:,k] * self.Ts)

            # constrain the states of the system at every time step
            self.opti.subject_to(self.X[:,k] < state_constraints['upper'])
            self.opti.subject_to(self.X[:,k] > state_constraints['lower'])

            # obstacle avoid at every step of the MPC
            self.opti.subject_to(
                radius**2 <=
                x_stretch*(self.X[0,k] - x_pos)**2 +
                y_stretch*(self.X[1,k] - y_pos)**2
            )

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # define dummy reference
        reference = state0
        reference[0] = 2
        self.ref = self.opti.parameter(n,1)
        self.opti.set_value(self.ref, reference)

        # Choose cost function
        # self.opti.minimize(self.cost_no_discount(self.X, self.ref, self.U)) # not discounted
        self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted

        # solve
        opts = {'ipopt.print_level':0, 'print_time':0} # silence!
        self.opti.solver('ipopt', opts)
        sol = self.opti.solve()

    def cost_no_discount(self, state, reference, input):
        # in this cost we penalise the state error at every timestep equally without discount
        state_error = reference - state

        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N):
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input
            
        return cost
    
    def cost_discounted(self, state, reference, input):
        state_error = reference - state
        gamma = 0.99 # standard for RL 0.99
        cost = ca.MX(0)
        xc = 1
        yc = 1
        distance_to_obstacle = ca.sqrt((state[0,:]-xc)**2+(state[1,:]-yc)**2)
        buffer_zone = 0.5
        infeasible_zone = 0.1

        # lets get cost per timestep:
        for k in range(self.N):
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)*gamma**k
            cost += ca.if_else(distance_to_obstacle[k] < buffer_zone, 1/(distance_to_obstacle[k]-infeasible_zone), 0)

        return cost
    
    def __call__(self, state, reference):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)

        # define cost w.r.t reference
        self.opti.set_value(self.ref, reference)

        # solve
        return self.opti.solve()

class MPC_Traj_Ref:
    def __init__(self, N, Ts, n, m, quad, reference_traj):
        self.N = N      # horizon
        self.Ts = Ts    # sampling time
        self.n = n      # num states
        self.m = m      # num inputs
        self.dynamics = quad.casadi_state_dot
        self.t = 0 # initialise time for trajectory reference tracking
        self.traj = reference_traj
        self.quad = quad

        self.opti = ca.Opti()
        self.X = self.opti.variable(n, N+1)
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

        # define start condition (dummy)
        state0 = [0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]
        self.init = self.opti.parameter(n,1)
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # constrain optimisation to the system dynamics over the horizon
        for k in range(self.N):
            # make sure the system follows its dynamics
            sdot = self.dynamics(self.X, self.U)
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + sdot[:,k] * self.Ts)

            # constrain the states of the system at every time step
            self.opti.subject_to(self.X[:,k] < quad.casadi_constraints['upper'])
            self.opti.subject_to(self.X[:,k] > quad.casadi_constraints['lower'])

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # define state constraints (obstacles/flight envelope)
        # self.obstacle = self.opti.parameter() # so it is changeable later
        # self.opti.subject_to()

        # define dummy reference (n x N)
        reference = np.zeros([n,N+1])

        self.ref = self.opti.parameter(n,N+1)
        self.opti.set_value(self.ref, reference)

        # Choose cost function
        # self.opti.minimize(self.cost_no_discount(self.X, self.ref, self.U)) # not discounted
        self.opti.minimize(self.cost_discounted(self.X, self.ref, self.U)) # discounted

        # solve
        opts = {'ipopt.print_level':0, 'print_time':0} # silence!
        self.opti.solver('ipopt', opts)
        sol = self.opti.solve()

    def cost_no_discount(self, state, input, time):
        # in this cost we penalise the state error at every timestep equally without discount
        state_error = reference - state

        cost = ca.MX(0)
        # lets get cost per timestep:
        for k in range(self.N):
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input

        return cost
    
    def cost_discounted(self, state, reference, input):
        gamma = 0.99 # standard for RL 0.99
        state_error = reference - state

        cost = ca.MX(0)
        # lets get cost over horizon:
        for k in range(self.N):
            state_error = reference - state
            timestep_input = input[:,k]
            timestep_state_error = state_error[:,k]
            cost += (timestep_state_error.T @ self.Q @ timestep_state_error + timestep_input.T @ self.R @ timestep_input)*gamma**k

        return cost
    
    def __call__(self, state, time):
        # point to point control, from state to stationary reference

        # define start condition
        self.opti.set_value(self.init, state)

        # define cost w.r.t reference
        times = np.arange(time, time+self.Ts*(self.N+1), self.Ts)
        # get the reference for these times
        reference = []
        for k in range(self.N+1):
            sDes = self.traj.desiredState(times[k], self.Ts, self.quad)
            reference.append(sDes2state(sDes))
        reference = np.vstack(reference).T
        self.opti.set_value(self.ref, reference)

        # solve
        return self.opti.solve()


# example usage
# ---------------------

if __name__ == "__main__":

    from quad import Quadcopter

    quad = Quadcopter()

    N = 20
    Ts = 0.1
    n = 17
    m = 4
    dynamics = quad.casadi_state_dot

    # state is at x = 0, reference at x = 2
    state = [0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]
    reference = [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 522.9847412109375, 522.9847412109375, 522.9847412109375, 522.9847412109375]

    mpc = MPC(N, Ts, n, m, dynamics)

    sol = mpc(state, reference)

    # movement along x axis, as desired in prediction
    print(f'predicted closed loop x timehistory: {sol.value(mpc.X)[0,:]}')
    print(f'control action to take: {sol.value(mpc.U)[:,0]}') # first action to take



        
