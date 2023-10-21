# Description

This directory contains 3 separate copies of dynamics:

1. PyTorch
2. CasADI
3. Mujoco

The states of the system for all 3 are:

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

## PyTorch and CasADI Models

The equations of motion for the PyTorch and CasADI are the same and were based on [this repo](https://github.com/bobzwik/Quadcopter_SimCon), which derived them from first principles using the SymPy physics.mechanics module.

### PyTorch (eom_pt.py)

These dynamics are used for the equations of motion simulation. They contain a state_dot method, which recieves tensors for state and input, and output tensors, and is completely differentiable. It also contains a step function which operates on numpy arrays to allow for consistency with the Mujoco environment.

### CasADI (eom_ca.py)

These dynamics are for the MPC, a carbon copy of the PyTorch ones using CasADI functions instead.

### Assumptions

- The moments of inertia of the rotor are purely about the body axes of the quadrotor, in other words:
    - $I_{xz}$, $I_{xy}$, $I_{yz}$ = 0
    - $I_{xx}$, $I_{yy}$, $I_{zz}$ $\neq$ 0
- The rotors are dragless, their rotational rate remains the same unless a control action torque is applied to them.
- Aerodynamics are simple, there is no ground effect, no tip vortices, no non-homogenous body drag (same in all orientations), and so on. The only contributions from aerodynamics are shown below:
    - Rotor thrust = $k_{th} \omega^2$
    - Rotor torque = $k_{to} \omega^2$
    - Drone body drag = $C_d (\dot{x}^2 + \dot{y}^2 + \dot{z}^2) = C_d V^2$

## Mujoco Model (mj.py, quadrotor_x.xml)

The Mujoco model is not defined using equations of motion, but through an XML file of bodies, actuators, frames of reference and so on. This is found in the quadrotor_x.xml file. This model is used as a pseudo real life analogue to use as a test bed for the controllers produced using the "known" dynamics in PyTorch and CasADI.
