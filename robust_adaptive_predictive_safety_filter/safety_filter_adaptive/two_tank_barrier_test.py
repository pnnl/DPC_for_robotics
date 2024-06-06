from casadi import *
import matplotlib.pyplot as plt
from predictive_safety_filter.safety_filter import integrate_adaptive
from tqdm import tqdm

class TestBarrierFunction():

    def __init__(self, model, u, dt, params, hf, integrator_type='cont'):
        self.model = model
        self.u = u
        self.dt = dt
        self.hf = hf
        self.params = params
        self.integrator_type = integrator_type

    def eval(self, N, Ld_terminal, x1_range, x2_range, theta1_range, theta2_range, Lf_theta=None, thetabound = None):
        '''Sample points from x1_range and x2_range and check if a control input is available to satisfy the hf condition'''


        print('Checking barrier function...')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        figh = plt.figure()
        axh = figh.add_subplot(111, projection='3d')
        max_hf_check =- np.inf
        rob_marg = self.params['Lhf_x']*Ld_terminal*self.params['Lf_x']**(N-1)

        if Lf_theta and thetabound:
            rob_marg += self.params['Lhf_x']*Lf_theta*thetabound

        print('rob marg = '+str(rob_marg))
        print('theta1 = '+str(theta1_range))
        print('tehta2 = '+str(theta2_range))
        print('robustness terms: '+str(self.params))

        for x1 in tqdm(x1_range):
            for x2 in x2_range:
                for theta1 in theta1_range:
                    for theta2 in theta2_range:

                        bool_val = False
                        x = np.array([x1, x2])
                        theta = np.array([theta1, theta2])

                        # Only perform the check if we are in the sublevel set of hf
                        if self.hf(x, 0) <= 0:

                            # Check if a zero control satisfies the conditions
                            f_x = integrate_adaptive(self.model, x, 0, np.array([0, 0]), theta, self.dt, integrator_type=self.integrator_type)
                            if self.hf(f_x, 0) <= -rob_marg:
                                ax.scatter(x1, x2, 0, color='b')
                                axh.scatter(x1, x2, self.hf(f_x, 0) + rob_marg, color='b')

                            # If a zero control does not satisfy the condition, compute the optimal control
                            else:
                                u = self.u(x, 0, theta)
                                u1 = u[0]
                                u2 = u[1]

                                # Check if the optimal control satisfies the hf condition
                                f_x = integrate_adaptive(self.model, x, 0, np.array([u1, u2]), theta, self.dt, integrator_type=self.integrator_type)
                                hf_check = self.hf(f_x, 0) + rob_marg
                                ax.scatter(x1, x2, u1, color='r', )
                                ax.scatter(x1, x2, u2, color='g')
                                axh.scatter(x1, x2, hf_check, color='m')
                                max_hf_check = max([max_hf_check, hf_check])

                                # If the condition is not satisfies, print the violating points
                                if hf_check > 0.0:
                                    print('hf violated at x1 = ' + str(x1) + ', x2 = ' + str(x2) + ', theta1 = '+str(theta1)+ ', theta2 = '+str(theta2))
                                    print('hf_check = ' + str(hf_check))
                                    print('bool val ' + str(bool_val))

        print('max hf_check = '+str(max_hf_check))
        print('Check complete.')
        # Plot the results for verification
        plt.figure(fig)
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('u')
        plt.figure(figh)
        plt.xlabel('x1')
        plt.ylabel('x2')
        axh.set_zlabel('hf + Lhx Ld (Lfx)^N-1')
        plt.show()

    def check_constraint_conditions(self, N, Ld, Nsim, hf_level_plus, hf_level_minus):

        # Compute constraints with robust margins and plot to test feasibility
        X_rob_upper_term = []
        X_rob_lower_term = []
        rob_marg_terminal = Ld*(self.params['Lf_x'])**(N-1)
        for kk in range(Nsim):

            X_rob_upper_term.append(hf_level_plus(0) - rob_marg_terminal)
            X_rob_lower_term.append(hf_level_minus(0) + rob_marg_terminal)

        t = range(N)
        plt.figure()
        plt.plot(range(Nsim), self.model.xmax[-1] * np.ones(Nsim), 'k', label='safe set (X) upper')
        plt.plot(range(Nsim), self.model.xmin[-1] * np.ones(Nsim), 'k', label='safe set (X) lower')
        plt.plot(X_rob_upper_term, 'b--', label='terminal set (Xf) upper')
        plt.plot(X_rob_lower_term, 'b', label='terminal set (Xf) lower')

        for ii in range(0, Nsim, 50):
            time_x = []
            X_rob_upper = []
            X_rob_lower = []
            for kk in range(N):
                time_x.append(ii + kk)
                rob_marg = self.params['Lh_x']*Ld*sum([ self.params['Lf_x']**j for j in range(kk) ])
                X_rob_upper.append(self.model.xmax[-1] - rob_marg)
                X_rob_lower.append(self.model.xmin[-1] + rob_marg)

            plt.plot(time_x, X_rob_upper, 'r--')
            plt.plot(time_x, X_rob_lower, 'r')

        plt.xlabel('Time step')
        plt.ylabel('State')
        plt.title('Constraint Sets for N = '+str(N)+', Ld = '+str(Ld))
        plt.legend()
        plt.show()

class ValidControl():

    def __init__(self,  dt, r, rho):
        self.dt = dt
        self.r = r
        self.rho = rho


    def __call__(self, x, k, theta):

        x1 = x[0]
        x2 = x[1]
        c1 = theta[0]
        c2 = theta[1]
        r1 = self.r[0]
        r2 = self.r[1]

        y1 = -(x1 - r1) / self.dt + c2 * sqrt(x1)
        y2 = -(x2 - r2) / self.dt - c2 * sqrt(x1) + c2 * sqrt(x2)

        self.opti = Opti()
        self.y1 = self.opti.parameter()
        self.y2 = self.opti.parameter()
        self.u1 = self.opti.variable()
        self.u2 = self.opti.variable()

        self.opti.subject_to(self.u1 >= 0.0)
        self.opti.subject_to(self.u2 >= 0.0)
        self.opti.subject_to(self.u1 <= 1.0)
        self.opti.subject_to(self.u2 <= 1.0)

        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {"print_level": 1}
        self.opti.solver("ipopt", self.p_opts, self.s_opts)

        self.opti.minimize((c1 * (1 - self.u2) * self.u1 - self.y1) ** 2 + self.rho * (c1 * self.u1 * self.u2 - self.y2) ** 2)

        self.opti.set_value(self.y1, y1)
        self.opti.set_value(self.y2, y2)
        sol = self.opti.solve()
        u1 = sol.value(self.u1)
        u2 = sol.value(self.u2)

        u = np.array([u1, u2])

        return u

