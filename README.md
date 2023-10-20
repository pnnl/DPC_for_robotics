# Differentiable Predictive Control with Safety Filter Applied to Robotics


This repo aims to augment a differentiable predictive controller (DPC) with a safety filter to allow for guaruntees of performance. The application is a 17 dimensional quadrotor (13 primary states, 4 actuator states, described below). The quadrotor has 3 simulations, one in PyTorch to allow for integration with learning, one in CASADI to allow for MPC, and one in Mujoco to allow for a validation environment.

## Analytical Quadrotor

The equations of motion were based on [this repo](https://github.com/bobzwik/Quadcopter_SimCon), which derived them from first principles using the SymPy physics.mechanics module.

The states of the system are:

$$\pmb{x} = \{x, y, z, q_0, q_1, q_2, q_3, \dot{x}, \dot{y}, \dot{z}, p_b, q_b, r_b, \omega_{M1}, \omega_{M2}, \omega_{M3}, \omega_{M4}\}$$

where:
- $x$, $y$, $z$ are the cartesian coordinates of the drone in the inertial frame (relative to the world)
- $q_0$, $q_1$, $q_2$, $q_3$ are the attitude quaternion of the drone in the inertial frame (relative to the world)
- $\dot{x}$, $\dot{y}$, $\dot{z}$ are the cartesian velocities of the drone in the inertial frame (relative to the world)
- $p_b$, $q_b$, $r_b$ are the rotational rates in the body frame (relative to the drone)
- $\omega_{M1}$, $\omega_{M2}$, $\omega_{M3}$, $\omega_{M4}$, are the angular velocities of the 4 rotors
    - Front left = M1
    - Front right = M2
    - Back right = M3
    - Back left = M4

The inputs to the system are:

$$\pmb{u} = \{ \tau_{M1}, \tau_{M2}, \tau_{M3}, \tau_{M4} \}$$

where:
- $\tau_{Mx}$ is the torque applied to the rotors of the quadrotor

### Assumptions

- The moments of inertia of the rotor are purely about the body axes of the quadrotor, in other words:
    - $I_{xz}$, $I_{xy}$, $I_{yz}$ = 0
    - $I_{xx}$, $I_{yy}$, $I_{zz}$ $\neq$ 0
- The rotors are dragless, their rotational rate remains the same unless a control action torque is applied to them.
- Aerodynamics are simple, there is no ground effect, no tip vortices, no non-homogenous body drag (same in all orientations), and so on. The only contributions from aerodynamics are shown below:
    - Rotor thrust = $k_{th} \omega^2$
    - Rotor torque = $k_{to} \omega^2$
    - Drone body drag = $C_d (\dot{x}^2 + \dot{y}^2 + \dot{z}^2) = C_d V^2$

## Mujoco Quadrotor



## Nonlinear Model Predictive Control

### State Reference

Using CasADI IPOPT there is a state reference MPC implemented in mpc.py. It uses a discounted cost over the horizon, and samples once for every 10 simulation steps, and provides a single input command to be applied over the next 10 simulation steps as it is currently implemented.

There are some things I found to work better and worse however:
- Much like the controller provided by the original code, at a simulation timestep of much greater than 0.001s the quadcopter appears very difficult to control for reasons I have yet to investigate
- The MPC requires a horizon of about 3 seconds ahead of where it currently is before it becomes useful, this means that at a simulation timestep of 1e-03, and an MPC sampling timestep of 1e-02, it needs a horizon of 300 ahead to achieve this currently. I hypothesise that a much longer horizon would be massively beneficial to performance, but an MPC sampling timestep of more than 1e-02 seems to result in poor performance.

Ideas to improve the horizon issue:
- DONE: Discounted cost - already applied, minor improvements
- Variable timesteps in the prediction horizon - not yet applied due to some code being required, but definitely doable
- DONE: Having the MPC predictions use the simulations true timestep rather than its own - not yet applied, but would definitely improve performance

For now it works when the MPC prediction timestep is exactly the same as the simulation at 0.1 seconds, with a prediction horizon of 3 seconds.

### Trajectory Reference

Again using CasADI IPOPT and the same MPC prediction and simulation timestep, however now I pass the MPC an a-priori known reference trajectory, which has a desired position and velocity at any given time. The mpc then aims to diminish the cost of discrepancy from this trajectory. This is different from the state reference point to point technique above as in the MPC prediction stage, the reference changes in time, meaning that the MPC is able to anticipate and predict where the trajectory is going within its horizon. This leads to smooth convergence to the reference trajectory as can be seen in gifs/rework_parity_testing/traj_ref.gif.

### State Reference Obstacle Avoidance

This adds to the state reference MPC, by including positional constraints to the MPC. In gifs/rework_sampling_mpc/obs_avoid.gif you can see this in action.

### Quality of Life Improvements To Do

- DONE: Have the prediction step of the MPC have the same number of steps as the simulation, despite the MPC only interacting with it less
- DONE: Be able to choose the MPC sampling rate whilst retaining perfect predictions 


## DPC

# DPC_for_robotics
# DPC_for_robotics
# DPC_for_robotics
# DPC_for_robotics
# DPC_for_robotics
