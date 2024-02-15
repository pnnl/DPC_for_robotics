import casadi as ca
import l4casadi as l4c
import torch
import neuromancer as nm
import numpy as np

from time import time

from utils.integrate import euler, RK4, generate_variable_timesteps


mlp_state_dict = torch.load('data/nav_policy.pth')

# Define a function to adjust the layer names
def adjust_layer_names(key):
    # Remove 'callable.' and replace 'linear' with 'hidden_layers'
    new_key = key.replace('callable.linear', 'hidden_layers')
    
    # Split the key to check for layer index adjustments
    parts = new_key.split('.')
    
    if parts[0] == 'hidden_layers':
        index = int(parts[1])  # Get the index of the hidden layer
        
        # Rename 'hidden_layers.0' to 'input_layer'
        if index == 0:
            new_key = new_key.replace('hidden_layers.0', 'input_layer')
        
        # Rename 'hidden_layers.4' to 'output_layer'
        elif index == 4:
            new_key = new_key.replace('hidden_layers.4', 'output_layer')
        
        # Adjust the indices of the hidden layers to start at 0 again
        else:
            new_key = f'hidden_layers.{index - 1}' + '.' + '.'.join(parts[2:])
    
    return new_key

# Create a new state dict with the modified keys
l4c_state_dict = {adjust_layer_names(key): value for key, value in mlp_state_dict.items()}

# Load into l4casadi model
mlp = l4c.naive.MultiLayerPerceptron(6+2,20,3,4,'ReLU')
mlp.load_state_dict(l4c_state_dict)
unom = l4c.L4CasADi(mlp, model_expects_batch_dim=True)

class SafetyFilter:
    def __init__(self, dynamics, Ts, Tf_hzn, N, quad_params, integrator) -> None:
        """ 
        dynamics    - the quad dynamics themselves in casadi
        Ts          - timestep
        Tf_hzn      - the predictive horizon final time
        N           - the number of steps in the predictive horizon (variable in length according to generate_variable_timesteps)
        quad_params - the literal quad parameters
        """

        self.dynamics = dynamics
        dts = generate_variable_timesteps(Ts, Tf_hzn, N)
        self.quad_params = quad_params
        self.integrator = integrator
        self.N = N

        self.nx = 17
        self.nu = 4
        self.nvc = 2

        self.cylinder_position = np.array([[1.,1.]])

        

