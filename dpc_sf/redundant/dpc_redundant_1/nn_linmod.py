"""
script to linearise a neural network in pytorch using functorch
"""
import os
import torch
from neuromancer.modules.blocks import MLP
import numpy as np
from dpc_sf.dynamics.params import params
import torch.optim as optim
from dpc_sf.dynamics.jac_pt import QuadcopterJac
from torchdiffeq import odeint_adjoint as odeint
from dpc_sf.dynamics.eom_deq import QuadcopterDiffEq
import time

mlp = MLP(
    insize=21,
    outsize=17,
    hsizes=[4]*4
)

jac_func = torch.func.jacrev(mlp.forward)

x = torch.randn(17)
u = torch.randn(4)

linearisation = jac_func(torch.hstack([x,u]))

A = linearisation[:,0:17]
B = linearisation[:, 17:]

# we could just learn the difference between learned systems linearisation and the true linearisation
# true_jac = QuadcopterJac()
# 
# # lets use torchdiffeq as its the backend of psl
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
# # 
# data_size=1000
# batch_size = 20
# batch_time = 10
# x0 = params["default_init_state_pt"]
# u0 = torch.zeros(4)
# true_y0 = torch.hstack([x0, u0])
# t = torch.linspace(0.,5.,data_size)
# 
# quad = QuadcopterDiffEq()
# 
# with torch.no_grad():
#     true_y = odeint(quad, true_y0, t, method='dopri5')
# 
# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
#     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
# 
# def makedirs(dirname):
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)
# 
# class RunningAverageMeter(object):
#     """Computes and stores the average and current value"""
# 
#     def __init__(self, momentum=0.99):
#         self.momentum = momentum
#         self.reset()
# 
#     def reset(self):
#         self.val = None
#         self.avg = 0
# 
#     def update(self, val):
#         if self.val is None:
#             self.avg = val
#         else:
#             self.avg = self.avg * self.momentum + val * (1 - self.momentum)
#         self.val = val
# 
# if __name__ == '__main__':
# 
#     ii = 0
# 
#     func = mlp.to(device)
#     
#     optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
#     end = time.time()
# 
#     time_meter = RunningAverageMeter(0.97)
#     
#     loss_meter = RunningAverageMeter(0.97)
# 
#     niters = 2000
#     for itr in range(1, niters + 1):
#         optimizer.zero_grad()
#         batch_y0, batch_t, batch_y = get_batch()
#         pred_y = odeint(func, batch_y0, batch_t).to(device)
#         loss = torch.mean(torch.abs(pred_y - batch_y))
#         loss.backward()
#         optimizer.step()
# 
#         time_meter.update(time.time() - end)
#         loss_meter.update(loss.item())
# 
#         if itr % 20 == 0:
#             with torch.no_grad():
#                 pred_y = odeint(func, true_y0, t)
#                 loss = torch.mean(torch.abs(pred_y - true_y))
#                 print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
#                 # visualize(true_y, pred_y, func, ii)
#                 ii += 1
# 
#         end = time.time()