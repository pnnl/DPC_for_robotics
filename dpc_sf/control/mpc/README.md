# Nonlinear Model Predictive Control

## State Reference

Using CasADI IPOPT there is a state reference MPC implemented in mpc.py. It uses a discounted cost over the horizon, and samples once for every 10 simulation steps, and provides a single input command to be applied over the next 10 simulation steps as it is currently implemented.

There are some things I found to work better and worse however:
- Much like the controller provided by the original code, at a simulation timestep of much greater than 0.001s the quadcopter appears very difficult to control for reasons I have yet to investigate
- The MPC requires a horizon of about 3 seconds ahead of where it currently is before it becomes useful, this means that at a simulation timestep of 1e-03, and an MPC sampling timestep of 1e-02, it needs a horizon of 300 ahead to achieve this currently. I hypothesise that a much longer horizon would be massively beneficial to performance, but an MPC sampling timestep of more than 1e-02 seems to result in poor performance.

Ideas to improve the horizon issue:
- DONE: Discounted cost - already applied, minor improvements
- DONE: Variable timesteps in the prediction horizon
- DONE: Having the MPC predictions use the simulations true timestep rather than its own - not yet applied, but would definitely improve performance

## Trajectory Reference

Again using CasADI IPOPT and the same MPC prediction and simulation timestep, however now I pass the MPC an a-priori known reference trajectory, which has a desired position and velocity at any given time. The mpc then aims to diminish the cost of discrepancy from this trajectory. This is different from the state reference point to point technique above as in the MPC prediction stage, the reference changes in time, meaning that the MPC is able to anticipate and predict where the trajectory is going within its horizon. This leads to smooth convergence to the reference trajectory as can be seen in gifs/rework_parity_testing/traj_ref.gif.

## State Reference Obstacle Avoidance

This adds to the state reference MPC, by including positional constraints to the MPC. In gifs/rework_sampling_mpc/obs_avoid.gif you can see this in action.

## Quality of Life Improvements To Do

- DONE: Have the prediction step of the MPC have the same number of steps as the simulation, despite the MPC only interacting with it less
- DONE: Be able to choose the MPC sampling rate whilst retaining perfect predictions
