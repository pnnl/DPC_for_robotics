import torch

# Load the saved state_dict
filename = "policy/DPC/wp_p2p.pth"
loaded_state_dict = torch.load(filename)

# Create a new dictionary to store the modified state_dict
policy_state_dict = {}

# Process keys as described
for key, value in loaded_state_dict.items():
    if "callable." in key:
        new_key = key.split("nodes.0.nodes.1.")[-1]
        policy_state_dict[new_key] = value
    else:
        # This will keep other keys as they are. Remove if not desired.
        policy_state_dict[key] = value

# Optionally save the modified state_dict back to disk
save_filename = "policy/DPC/wp_p2p.pth"
torch.save(policy_state_dict, save_filename)

print(f"Modified state_dict saved to {save_filename}")
