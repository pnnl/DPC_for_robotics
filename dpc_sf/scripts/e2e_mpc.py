from dpc_sf.dynamics.params import params as quad_params, params
import numpy as np
from tqdm import tqdm
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.control.mpc.mpc import MPC_Point_Ref, MPC_Traj_Ref
from dpc_sf.control.mpc.mpc_optimized import MPC
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.control.trajectory.trajectory import equation_reference
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.dynamics.eom_pt import QuadcopterPT
import matplotlib.pyplot as plt

def e2e_test_opt(
        test='wp_traj', 
        backend='mj', 
        save_trajectory=True, 
        save_dir='data/mpc_timehistories/',
        save_name=None,
        plot_prediction=True,
        Ts=0.1,
        Ti=0.0,
        Tf=1.0,
        N=30,
        Tf_hzn = 3.0,
    ):

    print(f'conducting e2e mj mpc: {test}')
    integrator = "euler"

    # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
    dts_init = [dt_1 + i * d for i in range(N)]

    if test == 'wp_p2p':
        reference = waypoint_reference('wp_p2p', average_vel=0.1, set_vel_zero=False)
        obstacle_opts = {'r': 0.5, 'x': 1, 'y': 1}
    elif test == 'wp_traj':
        reference = waypoint_reference('wp_traj', average_vel=1.0, set_vel_zero=False)
        obstacle_opts = None
    elif test == 'fig8':
        reference = equation_reference(test, average_vel=1.0)
        obstacle_opts = None

    state = quad_params["default_init_state_np"]
    quadCA = QuadcopterCA(params=quad_params)

    if backend == 'mj':
        quad = QuadcopterMJ(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator=integrator,
            xml_path="quadrotor_x.xml",
            write_path="media/mujoco/",
            render='matplotlib' # render='online_mujoco'
        )
        # quad.start_online_render()
    elif backend == 'eom':
        quad = QuadcopterPT(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator='euler',
        )

    ctrl = MPC(N, Ts, Tf_hzn, dts_init, quadCA, integrator, obstacle_opts)

    ctrl_pred_x = []
    for t in tqdm(np.arange(Ti, Tf, Ts)):
        # print(t)
        # for waypoint navigation stack the end reference point
        
        if test == 'wp_p2p':
            # for wp_p2p
            r = np.vstack([reference(quad.t)]*(N+1)).T

        elif test == 'wp_traj' or 'fig8':
            # for wp_traj (only constant dts)
            def compute_times(t_start, timesteps):
                times = [t_start]
                for dt in timesteps:
                    times.append(times[-1] + dt)
                return np.array(times)  # Exclude the starting time

            times = compute_times(t, dts_init)
            r = np.vstack([reference(time) for time in times]).T

        cmd = ctrl(quad.state, r)
        quad.step(cmd)

        ctrl_predictions = ctrl.get_predictions() 
        ctrl_pred_x.append(ctrl_predictions[0])

    ctrl_pred_x = np.stack(ctrl_pred_x)

    # make sure that the number of frames rendered is never too large!
    num_steps = int(Tf / Ts)
    max_frames = 500
    def compute_render_interval(num_steps, max_frames):
        render_interval = 1  # Start with rendering every frame.
        # While the number of frames using the current render interval exceeds max_frames, double the render interval.
        while num_steps / render_interval > max_frames:
            render_interval *= 2
        return render_interval
    render_interval = compute_render_interval(num_steps, max_frames)

    print(f"animating {backend} {test} with render interval of {render_interval}")
    if plot_prediction is True:
        quad.animate(
            state_prediction=ctrl_pred_x, 
            render_interval=render_interval
        )
    else:
        quad.animate(
            state_prediction=None,
            render_interval=render_interval
        )


# end to end test
def e2e_test(
        test='wp_traj', 
        backend='mj', 
        save_trajectory=True, 
        save_dir='data/mpc_timehistories/',
        save_name=None,
        plot_prediction=True,
        Ts=0.1,
        Ti=0.0,
        Tf=0.2,
        N=30
    ):

    print(f'conducting e2e mj mpc: {test}')
    integrator = 'euler' # 'euler', 'RK4'

    if test == 'wp_p2p':
        reference = waypoint_reference(test, average_vel=0.1, set_vel_zero=False)
    elif test == 'wp_traj':
        reference = waypoint_reference(test, average_vel=1.0, set_vel_zero=False)
    else:
        reference = equation_reference(test, average_vel=1.0)

    state = np.array([
        0,                  # x
        0,                  # y
        0,                  # z
        1,                  # q0
        0,                  # q1
        0,                  # q2
        0,                  # q3
        0,                  # xdot
        0,                  # ydot
        0,                  # zdot
        0,                  # p
        0,                  # q
        0,                  # r
        522.9847140714692,  # wM1
        522.9847140714692,  # wM2
        522.9847140714692,  # wM3
        522.9847140714692   # wM4
    ])

    quadCA = QuadcopterCA(params=params)

    if backend == 'mj':
        quad = QuadcopterMJ(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator='euler',
            xml_path="quadrotor_x.xml",
            write_path="media/mujoco/",
            render='matplotlib' # render='online_mujoco'
        )
        # quad.start_online_render()
    elif backend == 'eom':
        quad = QuadcopterPT(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator='euler',
        )

    if test == 'wp_p2p':
        ctrl = MPC_Point_Ref(
            N=N,
            dt=Ts,
            interaction_interval=1,
            n=17,
            m=4,
            dynamics=quadCA.state_dot,
            state_ub=params['ca_state_ub'],
            state_lb=params['ca_state_lb'],
            return_type='numpy',
            obstacle=True,
            integrator_type=integrator
        )
    elif test == 'wp_traj' or test == 'fig8':
        ctrl = MPC_Traj_Ref(
            N=N,
            dt=Ts,
            interaction_interval=1, 
            n=17, 
            m=4, 
            dynamics=quadCA.state_dot,
            state_ub=params["ca_state_ub"],
            state_lb=params["ca_state_lb"],
            reference_traj=reference,
            return_type='numpy',
            integrator_type=integrator
        )

    print(f"running {test}")
    ctrl_pred_x = []
    for t in tqdm(np.arange(Ti, Tf, Ts)):
        # print(t)
        if test == 'wp_p2p':
            cmd = ctrl(quad.state, reference(quad.t))
        else:
            cmd = ctrl(quad.state, quad.t)
        quad.step(cmd)

        ctrl_predictions = ctrl.get_predictions() 
        ctrl_pred_x.append(ctrl_predictions[0])

    # while quad.t < quad.Tf:
    #     print(quad.t)
    #     if test == 'wp_p2p':
    #         cmd = ctrl(quad.state, reference(quad.t))
    #     else:
    #         cmd = ctrl(quad.state, quad.t)
    #     quad.step(cmd)
    # 
    #     ctrl_predictions = ctrl.get_predictions() 
    #     ctrl_pred_x.append(ctrl_predictions[0])

    if save_trajectory:
        print("saving the state and input histories...")
        x_history = np.stack(quad.state_history)
        u_history = np.stack(quad.input_history)
        if save_name is None:
            np.savez(
                file = f"{save_dir}/xu_{test}_{backend}_{str(Ts)}.npz",
                x_history = x_history,
                u_history = u_history
            )
        elif save_name is not None:
            np.savez(
                file = f"{save_dir}/{save_name}",
                x_history = x_history,
                u_history = u_history
            )

    ctrl_pred_x = np.stack(ctrl_pred_x)

    # make sure that the number of frames rendered is never too large!
    num_steps = int(Tf / Ts)
    max_frames = 500
    def compute_render_interval(num_steps, max_frames):
        render_interval = 1  # Start with rendering every frame.
        # While the number of frames using the current render interval exceeds max_frames, double the render interval.
        while num_steps / render_interval > max_frames:
            render_interval *= 2
        return render_interval
    render_interval = compute_render_interval(num_steps, max_frames)

    print(f"animating {backend} {test} with render interval of {render_interval}")
    if plot_prediction is True:
        quad.animate(
            state_prediction=ctrl_pred_x, 
            render_interval=render_interval
        )
    else:
        quad.animate(
            state_prediction=None,
            render_interval=render_interval
        )

    # quad.reset(state=params["default_init_state_np"])

if __name__ == '__main__':
    # run through all mpc tests
    e2e_test_opt('wp_p2p', 'eom')
    e2e_test_opt('wp_traj', 'eom')
    e2e_test_opt('fig8', 'eom')
    e2e_test_opt('wp_p2p', 'mj')
    e2e_test_opt('wp_traj', 'mj')
    e2e_test_opt('fig8', 'mj')
    # e2e_test('wp_traj', 'mj')
    # e2e_test('wp_p2p', 'mj')
    # e2e_test('fig8', 'mj')
    # e2e_test('wp_p2p', 'eom')
    # e2e_test('wp_traj', 'eom')
    # e2e_test('fig8', 'eom')

