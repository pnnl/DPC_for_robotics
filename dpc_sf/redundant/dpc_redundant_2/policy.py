import torch

from neuromancer.modules.blocks import MLP
from neuromancer.modules.activations import SoftExponential
import neuromancer.slim as slim

from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.normalisation import denormalize_pt

class Policy(torch.nn.Module):

    def __init__(
            self, 
            insize=34, 
            outsize=3,
            hsize=64,
            n_layers=3,
            mlp_umin=-1,
            mlp_umax=1, 
            Ts=0.1,
            normalize=True,
            mean = None,
            var = None,
            bs = 1,
        ):

        super().__init__()

        # the MLP for learning DPC
        self.net = MLP_bounds_MIMO(
            insize=insize, 
            outsize=outsize, 
            hsizes=[hsize] * n_layers,
            min=[mlp_umin] * outsize, 
            max=[mlp_umax] * outsize,
        )

        self.mean = mean
        self.var = var
        self.bs = bs

        # the low level controller for orientation
        self.vel_sp_2_w_cmd = XYZ_Vel(Ts=Ts, bs=bs)


    def forward(self, x_norm, r_norm):
        """
        Expects normalised state and reference signals
        """

        # the features being input to the policy are the same as before
        features = torch.cat([x_norm, r_norm], dim=-1)

        # the action will be a desired [thrust, pitch, roll]
        vel_sp = self.net(features)

        # input desired action into the low level P controller to get desired w
        x = denormalize_pt(x_norm, means=ptu.from_numpy(self.mean), variances=ptu.from_numpy(self.var))

        w_cmd = self.vel_sp_2_w_cmd(x, vel_sp)

        return w_cmd


def sigmoid_scale(x, min, max):
    return (max - min) * torch.sigmoid(x) + min

def relu_clamp(x, min, max):
    x = x + torch.relu(-x + min)
    x = x - torch.relu(x - max)
    return x

class MLP_bounds_MIMO(MLP):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """
    bound_methods = {'sigmoid_scale': sigmoid_scale,
                    'relu_clamp': relu_clamp}

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        min=[0],
        max=[1.0],
        method='sigmoid_scale',
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__(insize=insize, outsize=outsize, bias=bias,
                         linear_map=linear_map, nonlin=nonlin,
                         hsizes=hsizes, linargs=linargs)
        assert len(min) == outsize, f'min and max ({min}, {max}) should have the same size as the output ({outsize})'
        assert len(min) == len(max), f'min ({min}) and max ({max}) should be of the same size'

        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        self.method = self._set_method(method)

    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), \
                f'Method, {method} must be a key in {self.bound_methods} ' \
                f'or a differentiable callable.'
            return method

    def forward(self, x):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)
    
