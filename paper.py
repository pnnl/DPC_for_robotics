import torch
import numpy as np
import utils.pytorch as ptu
import matplotlib.pyplot as plt

ptu.init_dtype()
ptu.init_gpu()

data = np.load("data/xu_sf_p2p_mj_0.001.npz")
x, u, r = data['x_history'], data['u_history'], data['r_history']

# Create a (1, 1) subplot
fig, ax = plt.subplots(1, 1)

# Plot the time history of the quadcopter
ax.plot(x[:,0], x[:,1], label='Quadcopter Path')

# Plot a circle at (1, 1) with radius 0.5
circle = plt.Circle((1, 1), 0.5, color='red', fill=False)
ax.add_artist(circle)

# Set axes to have equal scale
ax.set_aspect('equal', adjustable='box')

# Set labels and title (optional)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Quadcopter 2D Position Time History')

# Add legend (optional)
ax.legend()

# Save the plot as an SVG file
plt.savefig('data/paper/quadcopter_path.svg', format='svg')
plt.close(fig)

distances = np.sqrt((x[:,0] - 1)**2 + (x[:,1] - 1)**2) - 0.5

print('fin')