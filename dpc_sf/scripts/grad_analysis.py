import torch
import copy

from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss

from dpc_sf import dynamics
from dpc_sf import utils 
from dpc_sf.utils import pytorch_utils as ptu

"""
We have three pieces of the DPC controlled system (ignoring the NN):
- loss function
- PI velocity controller
- Dynamics
"""
 
# Loss function
l = None

# PI velocity controller
c = None

# Dynamics
f = dynamics.eom_pt.state_dot_pt

# state space position to investigate
x_hover = dynamics.params.params["default_init_state_pt"]
x_rs1 = ptu.from_numpy(utils.random_state.random_state())
x_rs2 = ptu.from_numpy(utils.random_state.random_state())
# u = quad_dynamics.params.params["cmd_hover_pt"]
u = ptu.create_tensor([0,0,0,0])

"""
We know that the Dynamics are correctly differentiable by the analytical jacobian,
lets test that again to ensure that the random states are not erroneous somehow...
"""

def validate_jacobian(f, x, u):

    print('validating jacobian...')

    # find the true analytical jacobian A and B matrices for a default state
    jac_func = dynamics.jac_pt.QuadcopterJac(params=dynamics.params.params)
    A_gt, B_gt = jac_func(x)

    # find the PyTorch gradients through the dynamics
    A, B = torch.func.jacrev(f, argnums=(0,1))(x, u)

    # validate they are equivalent
    assert (A-A_gt).abs().max() <= 1e-05
    assert (B-B_gt).abs().max() <= 1e-08

    print('passed')

validate_jacobian(f, x_hover, u)
validate_jacobian(f, x_rs1, u)
validate_jacobian(f, x_rs2, u)

"""
Now that we have validated that the jacobian is correct, let us look at it 
from some positions in the state space:
- hover
- random state 1
- random state 2
"""

def find_gradients(f, x, u):

    print('finding the gradients of position provided...')
    # find the PyTorch gradients through the dynamics
    A, B = torch.func.jacrev(f, argnums=(0,1))(x, u)

    print("----------------------------------------------------------------------------")
    print(f"xd, yd, zd: \n {A[:3,:]}")
    print("----------------------------------------------------------------------------")
    print(f"q0d, q1d, q2d, q3d: \n {A[3:7,:]}")
    print("----------------------------------------------------------------------------")
    print(f"xdd, ydd, zdd: \n {A[7:10,:]}")
    print("----------------------------------------------------------------------------")
    print(f"pd, qd, rd: \n {A[10:13,:]}")
    print("----------------------------------------------------------------------------")
    print(f"wM1d, wM2d, wM3d, wM4d: \n {A[13:17,:]}")
    print(f"B matrix influence: \n {B[13:17,:]}")
    print("----------------------------------------------------------------------------")
    print('done')
    print("----------------------------------------------------------------------------")

    return A, B

A, B = find_gradients(f, x_hover, u)
_, _ = find_gradients(f, x_rs1, u)
_, _ = find_gradients(f, x_rs2, u)



"""
Next let us look at the gradients of the input to the output through the graph
"""

def dudx(f, x, u):
    u.requires_grad_(True)

    # Compute the output by passing x and u through the function f
    dx = f(x, u)

    x_next = x + dx * 0.1

    # Compute gradients of y with respect to u
    x_next.backward(torch.ones_like(x_next))

    print(f"gradient of u w.r.t x: {u.grad}")

dudx(f, x_rs1, copy.deepcopy(u))

"""
Next let us look at the gradients of the input to the loss function
"""

def dudL(f, x, u, r):
    u.requires_grad_(True)

    # Compute the output by passing x and u through the function f
    dx = f(x, u)

    x_next = x + dx * 0.1

    error = r - x_next

    pos_error = error[0:3]

    # Compute gradients of y with respect to u
    pos_error.backward(torch.ones_like(pos_error))

    print(f"gradient of u w.r.t L: {u.grad}")

dudL(f, x_rs1, copy.deepcopy(u), x_hover)
dudL(f, x_rs1, copy.deepcopy(u), x_rs2)

"""
Next let us look at the gradients of the input passed through the low level controller to the loss
"""

def dudL_ll(f, x, u, r):
    pass

# template for a neuromancer problem for later
def create_nm_problem():

    sys = dynamics.eom_dpc.QuadcopterDPC(
        params=dynamics.params.params,
        nx=17,
        nu=4,
        ts=0.1,
        normalize=True,
        mean=None,
        var=None,
        include_actuators=True
    )

    sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'])

    cl_system = System([sys_node], nsteps=10)

    # u = variable('U')
    x = variable('X')
    r = variable('R')

    tracking_loss = 5.0 * (x[:,1:,10:13] == r[:,:,10:13]) ^ 2

    loss = PenaltyLoss(
        objectives=[tracking_loss], 
        constraints=[]
    )

    problem = Problem(
        nodes = [cl_system],
        loss = loss
    )
    return problem

print('fin')