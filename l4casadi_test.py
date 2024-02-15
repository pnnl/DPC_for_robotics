"""
This is a complete reformulation of the predictive safety filter for Johns purposes - it is a testing ground
"""

import casadi as ca
import l4casadi as l4c
import torch
from time import time
import neuromancer as nm

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

# Create a new state dict with the modified keys
nm_state_dict = {
    key.replace('callable.', ''): value
    for key, value in mlp_state_dict.items()
    if 'callable.' in key
}

nm_mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20])
nm_mlp.load_state_dict(nm_state_dict)
mlp = l4c.naive.MultiLayerPerceptron(6+2,20,3,4,'ReLU')
mlp.load_state_dict(l4c_state_dict)
l4c_model = l4c.L4CasADi(mlp, model_expects_batch_dim=True)

x_sym = ca.MX.sym('x', 8, 1)
y_sym = l4c_model(x_sym)
f = ca.Function('y', [x_sym], [y_sym])
df = ca.Function('dy', [x_sym], [ca.jacobian(y_sym, x_sym)])
# ddf = ca.Function('ddy', [x_sym], [ca.hessian(y_sym, x_sym)[0]])

x_pt = torch.tensor([[0.], [2.], [0.], [4], [0.], [2.], [0.], [4]])
x = ca.DM([[0.], [2.], [0.], [4], [0.], [2.], [0.], [4]])

tic = time()
print(l4c_model(x))
toc = time()
print(toc-tic)
print(nm_mlp(x_pt.T))

print('fin')