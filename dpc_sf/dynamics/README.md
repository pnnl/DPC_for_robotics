# Description

It is my hope that the dynamics are now set in stone in quad_dynamics. This directory contains 3 separate sets of dynamics:

1. PyTorch
2. CasADI
3. Mujoco

## PyTorch (eom_pt.py)

These dynamics are used for the equations of motion simulation. They contain a state_dot method, which recieves tensors for state and input, and output tensors, and is completely differentiable. It also contains a step function which operates on numpy arrays to allow for consistency with the Mujoco environment.

## CasADI (eom_ca.py)

These dynamics are for the MPC, a carbon copy of the PyTorch ones using CasADI functions instead.

## Mujoco (mj.py, quadrotor_x.xml)

These dynamics are not defined by the equations of motion, but instead an xml file "quadrotor_x.xml". These are to be a pseudo real life analogue to use as a test bed for the controllers produced using the "known" dynamics in PyTorch and CasADI.