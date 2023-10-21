This mpc_old.py file contains the original version of the MPC, of which there were three versions:

1. reference point tracking
2. reference trajectory tracking
3. reference point tracking with obstacle avoidance

These three control problems required very different things from CasADI, but still had substantial overlap. Firstly making them classes is required so that we do not need to constantly recreate the CasADI optimisation problem, instead we can instantiate the problem once in the constructor, and then change the 'parameters' of the optimisation, like reference to track, and the current state to start from, on the fly using methods.

I also had a problem with one of the types of control - the obstacle avoidance one, and that led me to isolating the differences between the classes to debug it. This was accelerated dramatically by reusing the common parts of the classes that I know to be correct in one MPC_Base class and creating the others by inheriting from it. This is one of the very few instances in which I would make an inheritance.

To keep things clear to someone who might be seeing this code for the first time however I have kept the non inheritance code around in this folder. I might also add other stuff which is not used anymore, but might be useful in here.