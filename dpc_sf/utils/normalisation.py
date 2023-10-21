"""
functions to normalise and unnormalise arrays given means and variances
"""
import torch
import numpy as np
import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.dynamics.params import params

def normalize_np(
        state: np.ndarray, 
        means: np.ndarray = params["state_mean"], 
        variances: np.ndarray = params["state_var"], 
        epsilon=1e-8, 
        clip_state=10.0) -> np.ndarray:
    
    return np.clip((state - means) / np.sqrt(variances + epsilon), -clip_state, clip_state)

def denormalize_np(
        state: np.ndarray,
        means: np.ndarray = params["state_mean"], 
        variances: np.ndarray = params["state_var"], 
        epsilon=1e-8) -> np.ndarray:
    
    return state * np.sqrt(variances + epsilon) + means

def normalize_rl(
        state: np.ndarray, 
        means: np.ndarray = params["rl_state_mean"], 
        variances: np.ndarray = params["rl_state_var"], 
        epsilon=1e-8, 
        clip_state=10.0) -> np.ndarray:
    
    return np.clip((state - means) / np.sqrt(variances + epsilon), -clip_state, clip_state)

def denormalize_rl(
        state: np.ndarray,
        means: np.ndarray = params["rl_state_mean"], 
        variances: np.ndarray = params["rl_state_var"], 
        epsilon=1e-8) -> np.ndarray:
    
    return state * np.sqrt(variances + epsilon) + means


def normalize_pt(
        state: torch.Tensor, 
        means: torch.Tensor = ptu.from_numpy(params["state_mean"]), 
        variances: torch.Tensor = ptu.from_numpy(params["state_var"]), 
        epsilon=1e-8, 
        clip_state=10.0) -> torch.Tensor:
    
    return torch.clip((state - means) / torch.sqrt(variances + epsilon), min=-clip_state, max=clip_state)

def denormalize_pt(
        state: torch.Tensor, 
        means: torch.Tensor = ptu.from_numpy(params["state_mean"]), 
        variances: torch.Tensor = ptu.from_numpy(params["state_var"]), 
        epsilon=1e-8) -> torch.Tensor:
    
    return state * torch.sqrt(variances + epsilon) + means

def normalise_dict(
        state_dot: torch.Tensor, 
        means: torch.Tensor = ptu.from_numpy(params["state_dot_mean"]), 
        variances: torch.Tensor = ptu.from_numpy(params["state_dot_var"]), 
        epsilon=1e-8, 
        clip_state=10.0,
        norm_keys = ['R', 'X', 'xn']
    ) -> torch.Tensor:
    # expand means and variances to expected shape
    means = means.unsqueeze(0).unsqueeze(1)
    variances = variances.unsqueeze(0).unsqueeze(1)

    norm_state_dot = {}
    for key in norm_keys:
        try:
            if isinstance(state_dot[key], np.ndarray):
                norm_state_dot[key] = torch.clip((ptu.from_numpy(state_dot[key]) - means) / torch.sqrt(variances + epsilon), min=-clip_state, max=clip_state)
            elif isinstance(state_dot[key], torch.Tensor):
                norm_state_dot[key] = torch.clip((state_dot[key] - means) / torch.sqrt(variances + epsilon), min=-clip_state, max=clip_state)
        except:
            continue
        
    return norm_state_dot


def normalize_nm(
        state_dot: np.ndarray, 
        means: np.ndarray = params["state_dot_mean"], 
        variances: np.ndarray = params["state_dot_var"], 
        epsilon=1e-8, 
        clip_state=10.0) -> np.ndarray:
    """
    In NeuroMANCER we expect tensors of shape (batch, rollout length, states),
    we wish to normalise every state, so everything in (batch, rollout).

    Further this function expects the state_dot rather than the state to normalise
    """
    assert len(state_dot.shape) == 3

    # expand means and variances to expected shape
    means = means[None,:][None,:]
    variances = variances[None,:][None,:]

    return np.clip((state_dot - means) / np.sqrt(variances + epsilon), a_min=-clip_state, a_max=clip_state)

def denormalize_nm(
        state_dot: np.ndarray, 
        means: np.ndarray = params["state_dot_mean"], 
        variances: np.ndarray = params["state_dot_var"], 
        epsilon=1e-8) -> np.ndarray:
    """
    In NeuroMANCER we expect tensors of shape (batch, rollout length, states),
    we wish to normalise every state, so everything in (batch, rollout)

    Further this function expects the state_dot rather than the state to denormalise
    """
    assert len(state_dot.shape) == 3

    # expand means and variances to expected shape
    means = means[None,:][None,:]
    variances = variances[None,:][None,:]

    return state_dot * np.sqrt(variances + epsilon) + means

# example usage:
if __name__ == '__main__':

    state_raw = params['default_init_state_np']
    means = params["state_mean"]
    variances = params["state_var"]
    state_norm = normalize_np(state_raw, means, variances)
    samples = np.random.normal(means, np.sqrt(variances), size=(10000, len(means)))

    # NumPy Testing
    # -------------
    maxs = samples.max(axis=0)
    mins = samples.min(axis=0)
    assert np.abs(denormalize_np(normalize_np(maxs, means, variances), means, variances) \
                - maxs).max() < 1e-10
    assert np.abs(denormalize_np(normalize_np(mins, means, variances), means, variances) \
                - mins).max() < 1e-10
    
    # PyTorch Testing
    # ---------------
    maxs_pt = ptu.from_numpy(maxs)
    mins_pt = ptu.from_numpy(mins)
    means_pt = ptu.from_numpy(means)
    variances_pt = ptu.from_numpy(variances)
    assert torch.abs(denormalize_pt(normalize_pt(maxs_pt, means_pt, variances_pt), means_pt, variances_pt) \
                - maxs_pt).max() < 1e-4
    assert torch.abs(denormalize_np(normalize_pt(mins_pt, means_pt, variances_pt), means_pt, variances_pt) \
                - mins_pt).max() < 1e-4
    
    print('fin')
