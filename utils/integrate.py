import numpy as np

def generate_variable_timesteps(Ts, Tf_hzn, N):

    # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
    dts = [dt_1 + i * d for i in range(N)]
    return dts

# for variable timestep prediction horizons
def generate_variable_times(t_start, timesteps):

    times = [t_start]
    for dt in timesteps:
        times.append(times[-1] + dt)
    times = np.array(times)  # Exclude the starting time
    return times


def euler(f, x, u, Ts, *args):
    return x + Ts * f(x, u, *args)


def RK4(f, x, u, Ts, *args):
    k1 = f(x, u, *args)
    k2 = f(x + Ts / 2 * k1, u, *args)
    k3 = f(x + Ts / 2 * k2, u, *args)
    k4 = f(x + Ts * k3, u, *args)
    return x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
