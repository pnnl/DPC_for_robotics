"""
Script which contains all hyperparams for the DPC training argparsed
"""

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Command line options for your program')

# Add arguments
parser.add_argument('--radius',                     type=float, default=0.50,                     help='Radius of the cylinder to be avoided')
parser.add_argument('--save_path',                  type=str,   default="data/policy/DPC_p2p/",   help='Path to save the policies')
parser.add_argument('--media_path',                 type=str,   default="data/media/dpc/images/", help='Path to save the policies')
parser.add_argument('--nstep',                      type=int,   default=100,                      help='Number of timesteps in the horizon')
parser.add_argument('--epochs',                     type=int,   default=10,                       help='Number of training epochs per iteration')
parser.add_argument('--iterations',                 type=int,   default=1,                        help='Number of iterations')
parser.add_argument('--lr',                         type=float, default=0.05,                     help='Learning rate')
parser.add_argument('--Ts',                         type=float, default=0.1,                      help='Timestep')
parser.add_argument('--minibatch_size',             type=int,   default=10,                       help='Autograd per minibatch size')
parser.add_argument('--batch_size',                 type=int,   default=5000,                     help='Batch size')
parser.add_argument('--x_range',                    type=float, default=3.,                       help='Multiplier for the initial states')
parser.add_argument('--r_range',                    type=float, default=3.,                       help='Multiplier for the reference end points')
parser.add_argument('--cyl_range',                  type=float, default=3.,                       help='Multiplier for the cylinder center location')
parser.add_argument('--use_rad_multiplier',         type=bool,  default=False,                    help='Increase radius over the horizon')
parser.add_argument('--train',                      type=bool,  default=True,                     help='Train the model')
parser.add_argument('--use_terminal_state_penalty', type=bool,  default=False,                    help='Use terminal state penalty in the cost function')
parser.add_argument('--Q_con',                      type=float, default=1_000_000.,               help='Cost of violating the cylinder constraint')
parser.add_argument('--Q_terminal',                 type=float, default=10.0,                     help='Cost of violating the terminal constraint')
parser.add_argument('--delta_terminal',             type=float, default=0.1,                      help='Terminal constraint initial radius')
parser.add_argument('--x_noise_std',                type=float, default=0.0,                      help='Terminal constraint initial radius')
# parser.add_argument('--u_noise_scale',              type=float, default=0.1,                      help='Terminal constraint initial radius')
parser.add_argument('--use_custom_callback',        type=bool,  default=True,                     help='Terminal constraint initial radius')
parser.add_argument('--lr_multiplier',              type=float, default=0.2,                      help='Terminal constraint initial radius')
parser.add_argument('--nstep_multiplier',           type=float, default=1,                        help='Terminal constraint initial radius')
parser.add_argument('--use_old_datasets',           type=bool,  default=False,                    help='Terminal constraint initial radius')
parser.add_argument('--sample_type',                type=str,   default='uniform',                 help='Terminal constraint initial radius')
parser.add_argument('--barrier_type',               type=str,   default='softexp',                 help='Terminal constraint initial radius')
parser.add_argument('--barrier_alpha',              type=float, default=0.05,                     help='Terminal constraint initial radius')
parser.add_argument('--use_cyl_constraint',         type=bool,  default=True,                    help='Terminal constraint initial radius')
parser.add_argument('--validate_data',              type=bool,  default=False,                    help='Terminal constraint initial radius')
parser.add_argument('--task',                       type=str,   default='wp_traj',                    help='Terminal constraint initial radius')
parser.add_argument('--Qpos',                       type=float, default=5.00,                    help='Terminal constraint initial radius')
parser.add_argument('--Qvel',                       type=float, default=5.00,                    help='Terminal constraint initial radius')
parser.add_argument('--Qtermscale',                 type=float, default=30.00,                    help='Terminal constraint initial radius')
parser.add_argument('--R',                          type=float, default=0.1,                    help='Terminal constraint initial radius')
parser.add_argument('--optimizer',                  type=str,   default='adagrad',                    help='Terminal constraint initial radius')
parser.add_argument('--fig8_observe_error',         type=bool,  default=True,                    help='Terminal constraint initial radius')
parser.add_argument('--p2p_dataset',                type=str,  default='cylinder_random',                    help='Terminal constraint initial radius')
parser.add_argument('--shuffle_dataloaders',        type=bool,  default=False,                    help='Terminal constraint initial radius')
parser.add_argument('--fig8_dataset',               type=str,  default='uniform_random',                    help='Terminal constraint initial radius')
parser.add_argument('--fig8_average_velocity',      type=float,  default=0.25,                    help='Terminal constraint initial radius')
parser.add_argument('--p2p_bimodal_policy',         type=bool,  default=False,                    help='Terminal constraint initial radius')


# Parse the arguments
args = parser.parse_args()

radius                      = args.radius                    
save_path                   = args.save_path                 
media_path                  = args.media_path                 
nstep                       = args.nstep                     
epochs                      = args.epochs                    
iterations                  = args.iterations                
lr                          = args.lr                        
Ts                          = args.Ts                        
minibatch_size              = args.minibatch_size            
batch_size                  = args.batch_size                
x_range                     = args.x_range                   
r_range                     = args.r_range                   
cyl_range                   = args.cyl_range                 
use_rad_multiplier          = args.use_rad_multiplier        
train                       = args.train                     
use_terminal_state_penalty  = args.use_terminal_state_penalty
Q_con                       = args.Q_con                     
Q_terminal                  = args.Q_terminal                
delta_terminal              = args.delta_terminal   
x_noise_std                 = args.x_noise_std         
use_custom_callback         = args.use_custom_callback         
lr_multiplier               = args.lr_multiplier         
nstep_multiplier            = args.nstep_multiplier    
use_old_datasets            = args.use_old_datasets
sample_type                 = args.sample_type # 'uniform', 'normal'
barrier_type                = args.barrier_type 
barrier_alpha               = args.barrier_alpha 
use_cyl_constraint          = args.use_cyl_constraint
validate_data               = args.validate_data
Qpos                        = args.Qpos
R                           = args.R
optimizer                   = args.optimizer
fig8_observe_error          = args.fig8_observe_error

# Convert args namespace to dictionary
args_dict = vars(args)