import ml_casadi.torch as mc
import casadi as ca
import numpy as np
import torch

size_in = 6
size_out = 3
model = mc.TorchMLCasadiModuleWrapper(
    torch_module,
    input_size=size_in,
    output_size=size_out)
    
casadi_sym_inp = ca.MX.sym('inp',size_in)
casadi_sym_out = model.approx(casadi_sym_inp, order=1)
casadi_func = ca.Function('model_approx_wrapper',
                          [casadi_sym_inp, model.sym_approx_params(order=1, flat=True)],
                          [casadi_sym_out])

inp = np.ones([1, size_in])  # torch needs batch dimension
casadi_param = model.approx_params(inp, order=1, flat=True)  # order=2
casadi_out = casadi_func(inp.transpose(-2, -1), casadi_param)   # transpose for vector rep. expected by casadi

t_out = model(torch.tensor(inp, dtype=torch.float32))

print(casadi_out)
print(t_out)