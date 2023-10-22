# Differentiable Predictive Control with Safety Filtration Applied to Robotics

This repo aims to augment a differentiable predictive controller (DPC) with a safety filter to allow for guaruntees of performance. The application is a 17 dimensional quadrotor (13 primary states, 4 actuator states, described below). The quadrotor has 3 simulations, one in PyTorch to allow for integration with learning, one in CASADI to allow for MPC, and one in Mujoco to allow for a validation environment.

## DPC