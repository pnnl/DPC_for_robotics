from setuptools import setup, find_packages

extras = {
    'mujoco': [],
}

# dependency
all_dependencies = []
for group_name in extras:
    all_dependencies += extras[group_name]
extras['all'] = all_dependencies

setup(name='dpc_sf',
      version='0.0.1',
      url='https://gitlab.pnnl.gov/dadaist/dpc_safety_filter/',
      author='John Viljoen',
      author_email='johnviljoen2@gmail.com',
      install_requires=[
          'matplotlib',
          'scipy',
          'numpy',
      ],
      packages=find_packages(
          include=[
              'dpc_sf',
              'dpc_sf.*',
              'utils'
          ]
      ),
      extras_require=extras,
      )
