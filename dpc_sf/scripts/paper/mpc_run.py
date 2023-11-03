"""
script to run all MPC examples as they will be seen in the paper. This
will take a ***very*** long time to run for some of them, even with my
optimisations of the optimisation.

This will save all the MPC timehistories as well for later analysis.
"""

from dpc_sf.control.mpc.run import run_mpc



run_mpc(
    task='wp_p2p',
    backend='mj', 
    Ts=0.001, 
    N=2000, 
    Tf_hzn=2.0, 
    Tf=5.0
)

run_mpc(
    task='fig8',
    backend='mj', 
    Ts=0.001, 
    N=3000, 
    Tf_hzn=3.0, 
    Tf=15.0
)

run_mpc(
    task='wp_traj', 
    backend='mj', 
    Ts=0.001, 
    N=30, 
    Tf_hzn=3.0, 
    Tf=15.0
)


