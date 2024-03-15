"""
generate a python env with version 3.10, and pip install -e . on this and u
should be good to go.

ensure the mujoco binary is also installed onto the system you are running onto in the proper way:
https://github.com/google-deepmind/mujoco
"""

from setuptools import setup, find_packages

setup(name='dpc_for_robotics',
    version='0.0.1',
    url='https://github.com/pnnl/dpc_for_robotics/',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'casadi',
        'neuromancer',  # downgrade torch to cpu only for l4casadi to work
        'matplotlib',
        'mujoco',       # for the uncertain black-box simulation, not really necessary
        'tqdm',         # just for pretty loops in a couple places
        'imageio',      # just for DPC callback animations
        'osqp',         # for testing purposes on convex hull operations - can probably be removed later.
        'scikit-build', # for l4casadi
        'mujoco-python-viewer',
        'imageio',
        'imageio-ffmpeg',
        # 'l4casadi --no-build-isolation', # run with this flag separately
        'gymnasium',    # for RL
    ],
    packages=find_packages(
        include=[
            'dpc_for_robotics.*'
        ]
    ),
)
