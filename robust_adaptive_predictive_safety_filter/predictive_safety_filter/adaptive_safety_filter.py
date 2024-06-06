from casadi import *
import numpy as np
import traceback
from predictive_safety_filter.safety_filter import SafetyFilter
from predictive_safety_filter.safety_filter import integrate_adaptive


class AdaptiveSafetyFilter(SafetyFilter):
    def __init__(self, x0, theta0, f, params, constraints, options=None):
        '''
        theta0: (np.array size (ntheta, 1)) Initial model parameter estimate
        '''

        self.theta = theta0
        self.n_theta = len(self.theta)
        try:
            self.terminal_conditions = params['terminal_conditions']
        except:
            print('No terminal conditions specified')
            raise(SystemError)
        if 'N' not in params:
            params['N'] = params['terminal_conditions'][0]['N']
        if 'Ld' not in params['robust_margins']:
            params['robust_margins']['Ld'] = params['terminal_conditions'][0]['Ld']
        if 'Ld_terminal' not in params['robust_margins']:
            params['robust_margins']['Ld_terminal'] = params['terminal_conditions'][0]['Ld']

        options['time-varying'] = True

        super().__init__(x0=x0, f=f, params=params, constraints=constraints, options=options)

        print('N = '+str(self.N))
        print('Ld = '+str(self.rob_marg['Ld']))
        print('Ld terminal = '+str(self.rob_marg['Ld_terminal']))


    def dynamics_constraints(self, X, U, opt, k0=0):
        '''Defines the constraints related to the ode of the system dynamics
        X: system state
        U: system input
        opt: Casadi optimization class
        k0: initial time
        '''

        # Loop over control intervals (these are the equality constraints associated with the dynamics)
        for k in range(self.N):
            x_next = integrate_adaptive(self.f, X[:, k], k, U[:, k],  self.theta, self.dT, self.integrator_type)
            opt.subject_to(X[:, k + 1] == x_next)

    def compute_robust_margin_terminal(self):
        '''Given dictionary of robustness margin terms, compute the robustness margin for the terminal constraint
        '''
        return self.rob_marg['Lhf_x']*self.rob_marg['Ld_terminal']*self.rob_marg['Lf_x']**(self.N-1)

    def check_constraint_satisfaction(self, unom, x0, k0=0):
        '''Check if constraints are satisfied for the future trajectory for given nominal control
         at the state x0 at time k0
         unom: function nominal control, unom(x,k)
         x0: initial state
         k0: initial time
         '''

        # Loop over control intervals (these are the equality constraints associated with the dynamics)
        x = x0
        for k in range(self.N):

            # Integrate system
            try:
                u = unom(x, k + k0)
            except:
                return 1, 'unom evaluation failed'
            x = integrate_adaptive(self.f, x, k, u, self.theta, self.dT, self.integrator_type)

            # Check input constraints
            for iu in range(self.ncu):
                check_input = self.input_constraints[iu](np.array(u).reshape(self.nu,1))
                if check_input > 0.0:
                    return 1, 'input_constraint_violation'

            # Check state constraints (TO DO: add a check before the for loop to check initial condition is in safe set)
            for ic in range(self.nc):
                rob_marg = self.compute_robust_margin(k)
                check_states = self.state_constraints[ic](x, k + k0) + rob_marg
                if check_states > 0.0:
                    return 1, 'state_constraint_violation'

        # Check terminal constraint
        rob_marg_term = self.compute_robust_margin_terminal()
        check_term = self.hf(x, k + k0 + 1.0) + rob_marg_term
        if check_term > 0.0:
            return 1, 'terminal_constraint_violation'

        return 0, 'no_violation'

    def update_model(self, theta, Ld):
        '''
        theta: (np.array of size (nthetax1)) Model parameter used in f(x,u,theta)
        Ld: (float) Error bound at time k
        '''

        self.theta = theta
        self.rob_marg['Ld'] = Ld

        if self.terminal_conditions:
            self.update_terminal_conditions(Ld)


    def update_terminal_conditions(self, Ld):
        '''
        N: (int) New prediction horizon, N>1
        delta: (float) Error bound used in terminal set condition
        '''

        for ii in range(len(self.terminal_conditions)):

            # If the current worst-case bound is smaller than the terminal bound, update to the terminal bound and horizon
            if self.terminal_conditions[ii]['Ld'] >= Ld:

                self.rob_marg['Ld_terminal'] = self.terminal_conditions[ii]['Ld']

                # Note the in the terminal conditions must be larger/equal to the current horizon. For shorter horizons a feasibility check is needed
                self.N = max(self.N,self.terminal_conditions[ii]['N'])

                # Reset warm start variables
                self.initialize_warm_start_variables()
