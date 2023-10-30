import torch
import torch.nn as nn

from neuromancer.dynamics import ode
from neuromancer.modules import blocks

from dpc_sf.utils import pytorch_utils as ptu

# Functions:
def posVel2cyl(state, cyl, radius):
    x = state[:, 0:1]
    y = state[:, 2:3]
    xc = cyl[:, 0:1]
    yc = cyl[:, 1:2]

    dx = x - xc
    dy = y - yc

    # Calculate the Euclidean distance from each point to the center of the cylinder
    distance_to_center = (dx**2 + dy**2) ** 0.5
    
    # Subtract the radius to get the distance to the cylinder surface
    distance_to_cylinder = distance_to_center - radius

    xdot = state[:, 1:2]
    ydot = state[:, 3:4]

    # Normalize the direction vector (from the point to the center of the cylinder)
    dx_normalized = dx / (distance_to_center + 1e-10)  # Adding a small number to prevent division by zero
    dy_normalized = dy / (distance_to_center + 1e-10)

    # Compute the dot product of the normalized direction vector with the velocity vector
    velocity_to_cylinder = dx_normalized * xdot + dy_normalized * ydot

    return distance_to_cylinder, velocity_to_cylinder

class Dynamics(ode.ODESystem):
    def __init__(self, insize, outsize, x_std=0.0) -> None:
        super().__init__(insize=insize, outsize=outsize)

        A = ptu.tensor([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        B = ptu.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.f = lambda x, u: x @ A.T + u @ B.T
        self.in_features = insize
        self.out_features = outsize
        
        # noise definitions
        self.x_std = x_std

    def ode_equations(self, x, u):
        # x = xu[:,0:6]
        # u = xu[:,6:9]
        # add noise if required
        x = x + torch.randn(x.shape, device=ptu.device) * self.x_std
        return self.f(x,u)
    
class processP2PMemSeqPolicyInput(torch.nn.Module):
    def __init__(self, minibatch_size=10, nx=6, radius=0.5, past_hzn=10) -> None:
        super().__init__()
        self.radius = radius
        self.minibatch_size = minibatch_size
        self.nx = nx
        self.past_hzn = past_hzn
        self.x_history      = ptu.create_zeros([self.minibatch_size, self.nx * self.past_hzn])
        self.r_history      = ptu.create_zeros([self.minibatch_size, self.nx * self.past_hzn])
        self.c_pos_history  = ptu.create_zeros([self.minibatch_size, self.past_hzn])
        self.c_vel_history  = ptu.create_zeros([self.minibatch_size, self.past_hzn])

        self.idx = 0

    def reset_histories(self, grad):
        print('resetting histories')
        self.x_history      = ptu.create_zeros([self.minibatch_size, self.nx * self.past_hzn])
        self.r_history      = ptu.create_zeros([self.minibatch_size, self.nx * self.past_hzn])
        self.c_pos_history  = ptu.create_zeros([self.minibatch_size, self.past_hzn])
        self.c_vel_history  = ptu.create_zeros([self.minibatch_size, self.past_hzn])
        self.idx = 0
        return grad

    def forward(self, x, r, cyl=None):

        c_pos, c_vel = posVel2cyl(x, cyl, self.radius)

        # fill out the zeros in the past sequence
        if self.idx < self.past_hzn:
            self.x_history[:, self.idx*self.nx:(self.idx+1)*self.nx] = x
            self.r_history[:, self.idx*self.nx:(self.idx+1)*self.nx] = r
            self.c_pos_history[:, self.idx:self.idx+1] = c_pos
            self.c_vel_history[:, self.idx:self.idx+1] = c_vel

        # implement rolling window to adjust sequence
        else:
            # Shift all values in x_history and r_history backward by self.nx positions
            self.x_history[:, :-self.nx] = self.x_history[:, self.nx:]
            self.x_history[:, -self.nx:] = x

            self.r_history[:, :-self.nx] = self.r_history[:, self.nx:]
            self.r_history[:, -self.nx:] = r

            # Shift all values in c_pos_history and c_vel_history backward by 1 position
            self.c_pos_history[:, :-1] = self.c_pos_history[:, 1:]
            self.c_pos_history[:, -1] = c_pos

            self.c_vel_history[:, :-1] = self.c_vel_history[:, 1:]
            self.c_vel_history[:, -1] = c_vel
            
        output = torch.hstack([
            self.x_history, 
            self.r_history, 
            self.c_pos_history, 
            self.c_vel_history
        ])

        # Register the hook to the output tensor.
        if output.requires_grad:
            output.register_hook(self.reset_histories)

        # increment call counter
        self.idx += 1

        return output

class processP2PPolicyInput(torch.nn.Module):
    def __init__(self, radius=0.5) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, x, r, cyl=None):
        # we want to pass the 
        e = r - x
        if cyl is not None:
            c_pos, c_vel = posVel2cyl(x, cyl, self.radius)
            return torch.hstack([e, c_pos, c_vel])
        else:
            return e

class processFig8TrajPolicyInput(torch.nn.Module):
    def __init__(self, use_error=True) -> None:
        super().__init__()
        self.use_error = use_error

    def forward(self, x, r, p):
        # we want to pass the 
        e = r - x
        if self.use_error is True:
            return torch.hstack([e,p])
        elif self.use_error is False:
            return torch.hstack([x,e,p])
        
class processP2PTrajPolicyInput(torch.nn.Module):
    def __init__(self, use_error=True) -> None:
        super().__init__()
        self.use_error = use_error

    def forward(self, x, r):
        # we want to pass the 
        e = r - x
        if self.use_error is True:
            return e
        elif self.use_error is False:
            return torch.hstack([x,e])

class radMultiplier(torch.nn.Module):
    def __init__(self, Ts, bs=1) -> None:
        super().__init__()
        self.Ts = Ts
        self.bs = bs

    def forward(self, i):
        # multiplier = 1 + idx * Ts * 0.5
        m = 1 + i * self.Ts * 0.01 # 0.15 # 0.5
        i = i + 1
        return i, m

class StateSeqTransformer(blocks.Block):
    """
    Transformer that takes in a "sequence" of states at the current time and produces an input.

    WARNING: Can use a LOT of memory - think 32 GB +
    """
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Embedding(input_dim, d_model)  # Assuming a max sequence length of 1000 for simplicity
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # No decoder for this simple setup
            dim_feedforward=dim_feedforward,
            batch_first=True  # Batch dimension first for simplicity
        )
        
        self.output = nn.Linear(d_model, output_dim)  # outputs are of size 3

    def block_eval(self, x):
        # x: [minibatch, states]
        batch_size, seq_length = x.size()
        
        # Reshape x to add a sequence dimension
        x = x.unsqueeze(1) # This will reshape x to [minibatch, 1, states]
        
        # Embedding
        embedded = self.embedding(x)
        
        # The embedded shape will now be [minibatch, 1, d_model], 
        # but we want it to be [minibatch, seq_length, d_model]
        embedded = embedded.repeat(1, seq_length, 1)
        
        # Add positional encodings
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        positional_emb = self.positional_encoding(positions)
        
        if positional_emb.shape[1] != embedded.shape[1]:
            raise ValueError(f"Positional embeddings shape {positional_emb.shape} does not match embedded sequence shape {embedded.shape}.")
        
        embedded += positional_emb
        
        # Pass through transformer; since we don't have a decoder, we only use the encoder
        memory = self.transformer.encoder(embedded)
        
        # For simplicity, take the output of the transformer for the last position (most recent input) as our output
        output = self.output(memory[:, -1, :])
        
        return output# .unsqueeze(1)  # shape: [minibatch, 1, 3]
    
class MemSeqTransformer(blocks.Block):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, sequence_length):
        super().__init__()

        # hyperparameters

        # linear layer maps the inputs to the embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # embedding with a memory of all of the past hzn and future hzn, with representation in d_model
        self.positional_encoding = nn.Embedding(sequence_length, d_model)

        # pytorch prebuilt transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # No decoder for this simple setup
            dim_feedforward=dim_feedforward,
            batch_first=True  # Batch dimension first for simplicity
        )

        # map transformer d_model to a usable output
        self.output = nn.Linear(d_model, output_dim)  # outputs are of size 3

    def block_eval(self, obs):
        # typical form of observation for this:
        # obs = [x_past, u_past, x_current, r_current, r_future]
        pass

    
class BimodalPolicy(torch.nn.Module):
    def __init__(self,
            insize=6+2,
            outsize=4, 
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=torch.nn.ReLU,
            hsizes=[20, 20, 20, 20]
        ) -> None:
        super().__init__()

        self.mode = []
        self.mode.append()
