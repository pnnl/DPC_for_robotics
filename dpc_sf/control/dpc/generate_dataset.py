import torch
import numpy as np

from neuromancer.dataset import DictDataset
from dpc_sf.utils import pytorch_utils as ptu

# Dataset Generation Functions
# ----------------------------

"""
Algorithm for dataset generation:

- we randomly sample 3 points for each rollout, quad start state, quad reference state, cylinder position
- if a particular datapoint has a start or reference state that is contained within the random cylinder 
  we discard that datapoint and try again. The output of this is a "filtered" dataset in "get_filtered_dataset"
- I also have a validation function to check the filtered datasets produced here, but you can ignore that
- these filtered datasets are then wrapped in DictDatasets and then wrapped in torch dataloaders

NOTE:
- the minibatch size used in the torch dataloader can play a key role in reducing the final steady state error
  of the system
    - If the minibatch is too large we will not get minimal steady state error, minibatch of 10 has proven good
"""

class DatasetGenerator:
    def __init__(
            self,
            p2p_dataset = 'uniform_random',
            task='wp_p2p', # 'wp_p2p', 'wp_traj', 'fig8'
            x_range=6., 
            r_range=6., 
            cyl_range=6.,
            radius=0.5,
            batch_size=1000,
            minibatch_size=100,
            nstep=100,
            sample_type='uniform',
            nx=6,
            validate_data=True,
            Ts=0.1,
            shuffle_dataloaders = False,
            average_velocity = 0.5,
        ) -> None:

        self.p2p_dataset = p2p_dataset
        self.task = task
        self.x_range = x_range
        self.r_range = r_range
        self.cyl_range = cyl_range
        self.radius = radius
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.nstep = nstep
        self.sample_type = sample_type
        self.nx = nx
        self.validate_data = validate_data
        self.Ts = Ts
        self.shuffle_dataloaders = shuffle_dataloaders
        self.average_velocity=average_velocity

    # Obstacle Avoidance Methods:
    # ---------------------------

    def is_inside_cylinder(self, x, y, cx, cy):
        """
        Check if a point (x,y) is inside a cylinder with center (cx, cy) and given radius.
        """
        distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance < self.radius
    
    def get_cyl_wp_p2p_dataset4(self):
        """
        The same as 3, but with symmetric sampling in an attempt to hide the modality problem
        """
        n = self.batch_size  # Replace with your actual number of points

        R = torch.cat([torch.cat([torch.tensor([[[2., 0, 2, 0, 1, 0]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Cyl = torch.cat([torch.cat([torch.tensor([[[1.,1]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Idx = torch.cat([torch.vstack([torch.tensor([0.0])]).unsqueeze(1)]*n, dim=0)
        M = torch.cat([torch.ones([1, 1, 1])]*n, dim=0)


        # Calculate half of n points, rounding up if n is odd
        half_n = (n + 1) // 2

        # Existing constants
        cylinder_center = torch.tensor([2.0, 2.0])
        cylinder_radius = torch.sqrt(torch.tensor(2**2 + 2**2))
        z_lower = -1
        z_upper = 1

        # Generate random angles for polar coordinates
        angles = torch.normal(mean=torch.tensor(1.25*torch.pi), std=torch.tensor(1), size=(half_n,))

        # Reflect the angles for symmetry
        reflected_angles = 2 * 1.25 * torch.pi - angles  # Replace 1.25*torch.pi with the actual angle you want to reflect around

        # Concatenate original and reflected angles
        all_angles = torch.cat([angles, reflected_angles])

        # If n is odd, remove one of the reflected points
        if n % 2 != 0:
            all_angles = all_angles[:-1]

        # Convert polar to Cartesian coordinates
        x = cylinder_radius * torch.cos(all_angles) + cylinder_center[0]
        y = cylinder_radius * torch.sin(all_angles) + cylinder_center[1]

        # Generate uniformly distributed Z values
        z = z_lower + (z_upper - z_lower) * torch.rand(n)

        # Stack x, y, and z to get the final coordinates
        xdot, ydot, zdot = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        X = torch.stack((x, xdot, y, ydot, z, zdot), dim=1).unsqueeze(1)

        return {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': Idx,
            'M': M
        }

    def get_cyl_wp_p2p_dataset3(self):
        """
        here we generate random samples in a cylinder around the desired end point, instead
        of the line in dataset2
        """
        X = []
        R = []
        Cyl = []

        n = self.batch_size  # Replace with your actual number of points

        R = torch.cat([torch.cat([torch.tensor([[[2., 0, 2, 0, 1, 0]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Cyl = torch.cat([torch.cat([torch.tensor([[[1.,1]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Idx = torch.cat([torch.vstack([torch.tensor([0.0])]).unsqueeze(1)]*n, dim=0)
        M = torch.cat([torch.ones([1, 1, 1])]*n, dim=0)

        cylinder_center = torch.tensor([2.0, 2.0])  # Center of the cylinder
        cylinder_radius = torch.sqrt(torch.tensor(2**2 + 2**2))  # Radius of the cylinder to pass through the origin
        z_lower = -1  # Lower bound for Z values
        z_upper = 1  # Upper bound for Z values

        # Generate random angles for polar coordinates
        # if generate with uniform dist
        # angles = 2 * torch.pi * torch.rand(n)  # uniform angles between [0, 2π]
        # if generate with normal dist
        angles = torch.normal(mean=torch.tensor(1.25*torch.pi), std=torch.tensor(1), size=(n,))# (2 * torch.pi * torch.randn(n)) + 0.75 * torch.pi

        # Convert polar to Cartesian coordinates
        x = cylinder_radius * torch.cos(angles) + cylinder_center[0]
        y = cylinder_radius * torch.sin(angles) + cylinder_center[1]

        # Generate uniformly distributed Z values
        z = z_lower + (z_upper - z_lower) * torch.rand(n)

        # Stack x, y, and z to get the final coordinates
        xdot, ydot, zdot = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        X = torch.stack((x, xdot, y, ydot, z, zdot), dim=1).unsqueeze(1)
        
        return {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': Idx,
            'M': M
        }


    def get_cyl_wp_p2p_dataset2(self):
        """
        Generate a far simpler dataset, far more akin to the problem I am trying to solve

        turned out this wasnt representative enough of the problem, as when applying to the
        full mujoco environment we ended up having issues where it would overshoot the mark
        due to the delay in application of the actions, which we left the space we trained in
        and then the system didnt come back because of it.
        """

        X = []
        R = []
        Cyl = []
        
        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")

        x_lower = -1  # Replace with your actual lower bound
        x_upper = 1   # Replace with your actual upper bound
        n = self.batch_size  # Replace with your actual number of points

        # Generate n linearly spaced x values between x_lower and x_upper
        x_values = x_lower + (x_upper - x_lower) * torch.rand(n) 
        # Calculate corresponding y values using the line equation y = -x
        y_values = -x_values
        z_values = torch.zeros(n)
        xdot_values = torch.zeros(n)
        ydot_values = torch.zeros(n)
        zdot_values = torch.zeros(n)
        points = torch.stack((x_values, xdot_values, y_values, ydot_values, z_values, zdot_values), dim=-1)
        X = points.unsqueeze(1)  # Example to make it (1, 1, n, 2)
        
        R = torch.cat([torch.cat([torch.tensor([[[2., 0, 2, 0, 1, 0]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Cyl = torch.cat([torch.cat([torch.tensor([[[1.,1]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Idx = torch.cat([torch.vstack([torch.tensor([0.0])]).unsqueeze(1)]*n, dim=0)
        M = torch.cat([torch.ones([1, 1, 1])]*n, dim=0)
        
        return {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': Idx,
            'M': M
        }

    def get_cyl_wp_p2p_dataset(self):    
        """
        Generate a filtered dataset of samples where none of the state or reference points are inside a cylinder.

        Returns:
        - dict: A dictionary containing:
            'X': Tensor of state data of shape [batch_size, 1, nx].
            'R': Tensor of reference data of shape [batch_size, nstep + 1, nx].
            'Cyl': Tensor of cylinder data of shape [batch_size, nstep + 1, 2].
            'Idx': Zero tensor indicating starting index for each sample in batch.
            'M': Tensor with ones indicating starting multiplier for each sample.
        """
        X = []
        R = []
        Cyl = []
        
        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")
        while len(X) < self.batch_size:

            x_sample, r_sample, cyl_sample = self.get_random_state()
            inside_cyl = False
            
            # Check if any state or reference point is inside the cylinder.
            for t in range(self.nstep + 1):
                if self.is_inside_cylinder(x_sample[0, 0, 0], x_sample[0, 0, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                    inside_cyl = True
                    break
                if self.is_inside_cylinder(r_sample[0, t, 0], r_sample[0, t, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                    inside_cyl = True
                    break
            
            if not inside_cyl:
                X.append(x_sample)
                R.append(r_sample)
                Cyl.append(cyl_sample)
        
        # Convert lists to tensors.
        X = torch.cat(X, dim=0)
        R = torch.cat(R, dim=0)
        Cyl = torch.cat(Cyl, dim=0)
        
        return {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': ptu.create_zeros([self.batch_size,1,1]), # start idx
            'M': torch.ones([self.batch_size, 1, 1]), # start multiplier
        }

    def validate_dataset(self, dataset):
        X = dataset['X']
        R = dataset['R']
        Cyl = dataset['Cyl']
        
        batch_size = X.shape[0]
        nstep = R.shape[1] - 1

        print("validating dataset...")
        for i in range(batch_size):
            for t in range(nstep + 1):
                # Check initial state.
                if self.is_inside_cylinder(X[i, 0, 0], X[i, 0, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                    return False, f"Initial state at index {i} lies inside the cylinder."
                # Check each reference point.
                if self.is_inside_cylinder(R[i, t, 0], R[i, t, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                    return False, f"Reference at time {t} for batch index {i} lies inside the cylinder."

        return True, "All points are outside the cylinder."

    # Trajectory Reference Methods:
    # -------------------------------

    def fig8(self, t, A=4, B=4, C=4, Z=-5, average_vel=1.0):

        # accelerate or decelerate time based on velocity desired
        t *= average_vel

        # Position
        x = A * np.cos(t)
        y = B * np.sin(2*t) / 2
        z = C * np.sin(t) + Z  # z oscillates around the plane Z

        # Velocities
        xdot = -A * np.sin(t)
        ydot = B * np.cos(2*t)
        zdot = C * np.cos(t)

        return ptu.from_numpy(np.array([x, y, z])).squeeze(), ptu.from_numpy(np.array([xdot, ydot, zdot])).squeeze()
    
    def generate_reference(self, mode='linear', average_velocity=0.5):
        """
        Generate a reference dataset.
        Parameters:
        - nstep: Number of steps
        - nx: Dimensionality of the reference
        - Ts: Time step
        - r_range: Range for random sampling
        - mode: 'linear' for straight line references, 'sinusoidal' for sinusoidal references
        """
        if mode == 'linear':
            start_point = self.r_range * torch.randn(1, 1, self.nx)
            end_point = self.r_range * torch.randn(1, 1, self.nx)

            pos_sample = []
            for dim in range(3):  # Only interpolate the positions (x, y, z)
                interpolated_values = torch.linspace(start_point[0, 0, dim], end_point[0, 0, dim], steps=self.nstep+1)
                pos_sample.append(interpolated_values)

            pos_sample = torch.stack(pos_sample, dim=-1)
            # Calculate the CORRECT velocities for our timestep
            vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / self.Ts

            # For the last velocity, we duplicate the last calculated velocity
            vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

            return pos_sample, vel_sample


        elif mode == 'sinusoidal':
            # randomise the initial time so as to look across the trajectory without such a long 
            # nstep prediction length.
            t_start = np.random.rand(1) * 15
            times = np.linspace(t_start, t_start + self.nstep * self.Ts, self.nstep + 1)  # Generate time values
            A = (np.random.rand(1) - 0.5) * 2 + 4
            B = (np.random.rand(1) - 0.5) * 2 + 4
            C = (np.random.rand(1) - 0.5) * 2 + 4
            Z = (np.random.rand(1) - 0.5) * 2 - 5
            average_vel = average_velocity # np.random.rand(1) * 2

            pos_sample = []
            vel_sample = []
            paras_sample = []
            T_total = self.Ts * (self.nstep)
            for t in times:
                pos, vel = self.fig8(t=t, A=A, B=B, C=C, Z=Z, average_vel=average_vel)
                paras = ptu.from_numpy(np.hstack([A,B,C,Z]))
                # pos[2] *= -1 # NED
                pos_sample.append(pos)
                paras_sample.append(paras)
                
                # vel_sample.append(vel)

            pos_sample = torch.stack(pos_sample)
            paras_sample = torch.stack(paras_sample)

            vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / self.Ts

            # For the last velocity, we duplicate the last calculated velocity
            vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

            return pos_sample, vel_sample, paras_sample
            # 
            # vel_sample = torch.stack(vel_sample)


            # print('fin')


    def get_linear_wp_traj_dataset(self):
        X = []
        R = []
        
        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")
        while len(X) < self.batch_size:
            x_sample = self.x_range * torch.randn(1, 1, self.nx)
            x_sample[:,:,0] *= 2.5
            # x_sample[:,:,0] += 2

            pos_sample, vel_sample = self.generate_reference(mode='linear')
            pos_sample *= 2.5
            pos_sample -= 2

            # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
            r_sample = torch.zeros(1, self.nstep+1, self.nx)
            r_sample[0, :, 0] = pos_sample[:, 0]
            r_sample[0, :, 1] = vel_sample[:, 0]
            r_sample[0, :, 2] = pos_sample[:, 1]
            r_sample[0, :, 3] = vel_sample[:, 1]
            r_sample[0, :, 4] = pos_sample[:, 2]
            r_sample[0, :, 5] = vel_sample[:, 2]
            
            X.append(x_sample)
            R.append(r_sample)
        
        # Convert lists to tensors.
        X = torch.cat(X, dim=0)
        R = torch.cat(R, dim=0)
        
        return {
            'X': X,
            'R': R,
            'Idx': ptu.create_zeros([self.batch_size,1,1]), # start idx
        }
    
    # sinusoidal trajectory reference functions:
    def get_sinusoidal_traj_dataset(self, average_velocity):
        X = []
        R = []
        P = []

        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")
        while len(X) < self.batch_size:
            
            x_sample = (torch.rand(1, 1, self.nx) - 0.5) * 8
            # set velocities to zero
            x_sample[:,:,1::2] *= 0
            # offset Z to be closer to the trajectories
            x_sample[:,:,4] -= 5

            pos_sample, vel_sample, paras_sample = self.generate_reference(mode='sinusoidal', average_velocity=average_velocity)

            # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
            r_sample = torch.zeros(1, self.nstep+1, self.nx)
            r_sample[0, :, 0] = pos_sample[:, 0]
            r_sample[0, :, 1] = vel_sample[:, 0]
            r_sample[0, :, 2] = pos_sample[:, 1]
            r_sample[0, :, 3] = vel_sample[:, 1]
            r_sample[0, :, 4] = pos_sample[:, 2]
            r_sample[0, :, 5] = vel_sample[:, 2]
            
            X.append(x_sample)
            R.append(r_sample)
            P.append(paras_sample.unsqueeze(0))
        
        # Convert lists to tensors.
        X = torch.cat(X, dim=0)
        R = torch.cat(R, dim=0)
        P = torch.cat(P, dim=0)
        
        return {
            'X': X,
            'R': R,
            'P': P,
            'Idx': ptu.create_zeros([self.batch_size,1,1]), # start idx
        }


    # Shared Methods
    # --------------

    def get_random_state(self):
        if self.sample_type == 'normal':
            x_sample = self.x_range * torch.randn(1, 1, self.nx)
            r_sample = torch.cat([self.r_range * torch.randn(1, 1, self.nx)] * (self.nstep + 1), dim=1)
            cyl_sample = torch.cat([self.cyl_range * torch.randn(1, 1, 2)] * (self.nstep + 1), dim=1)
        elif self.sample_type == 'uniform':
            x_sample = 2 * self.x_range * (torch.rand(1, 1, self.nx) - 0.5)
            r_sample = torch.cat([2 * self.r_range * (torch.rand(1, 1, self.nx) - 0.5)] * (self.nstep + 1), dim=1)
            cyl_sample = torch.cat([2 * self.cyl_range * (torch.rand(1, 1, 2) - 0.5)] * (self.nstep + 1), dim=1)
        else:
            raise ValueError(f"invalid sample type passed: {self.sample_type}")
        # reference velocities should be zero here 
        r_sample[:,:,1::2] = 0.
        return x_sample, r_sample, cyl_sample

    def get_dictdatasets(self):
        
        if self.task == 'wp_p2p':
            if self.p2p_dataset == 'line':
                train_data = DictDataset(self.get_cyl_wp_p2p_dataset2(), name='train')
                dev_data = DictDataset(self.get_cyl_wp_p2p_dataset2(), name='dev')            
            elif self.p2p_dataset == 'uniform_random':
                train_data = DictDataset(self.get_cyl_wp_p2p_dataset(), name='train')
                dev_data = DictDataset(self.get_cyl_wp_p2p_dataset(), name='dev')
            elif self.p2p_dataset == 'cylinder_random':
                train_data = DictDataset(self.get_cyl_wp_p2p_dataset3(), name='train')
                dev_data = DictDataset(self.get_cyl_wp_p2p_dataset3(), name='dev')      
            elif self.p2p_dataset == 'cylinder_random_symmetric':
                train_data = DictDataset(self.get_cyl_wp_p2p_dataset4(), name='train')
                dev_data = DictDataset(self.get_cyl_wp_p2p_dataset4(), name='dev')                         
        elif self.task == 'wp_traj':
            train_data = DictDataset(self.get_linear_wp_traj_dataset(), name='train')
            dev_data = DictDataset(self.get_linear_wp_traj_dataset(), name='dev')
        elif self.task == 'fig8':
            train_data = DictDataset(self.get_sinusoidal_traj_dataset(average_velocity=self.average_velocity), name='train')
            dev_data = DictDataset(self.get_sinusoidal_traj_dataset(average_velocity=self.average_velocity), name='dev')  
        else:
            raise Exception

        return train_data, dev_data

    def get_loaders(self):

        train_data, dev_data = self.get_dictdatasets()

        if self.validate_data is True and (self.task == 'wp_p2p'):
            self.validate_dataset(train_data.datadict)
            self.validate_dataset(dev_data.datadict)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.minibatch_size,
                                                collate_fn=train_data.collate_fn, shuffle=self.shuffle_dataloaders)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=self.minibatch_size,
                                                collate_fn=dev_data.collate_fn, shuffle=self.shuffle_dataloaders)
        
        return train_loader, dev_loader