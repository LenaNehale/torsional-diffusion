from argparse import ArgumentParser
import os
from pathlib import Path
if 'SCRATCH' in os.environ:
    SCRATCH=Path(os.environ['SCRATCH'])
else:
    SCRATCH = Path(__file__).resolve().parent.parent

def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=SCRATCH / 'torsional-diffusion/workdir/test_run', help='Folder in which to save model and logs')
    #parser.add_argument('--restart_dir', default=SCRATCH / 'torsional-diffusion/workdir/boltz_T300' ,  type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--restart_dir', default = None ,  type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--cache', type=str, default=SCRATCH / 'torsional-diffusion/cache/test_run', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--std_pickles', type=str, default=SCRATCH / 'torsional-diffusion/DRUGS/standardized_pickles', help='Folder in which the pickle are put after standardisation/matching')
    parser.add_argument('--split_path', type=str, default=SCRATCH / 'torsional-diffusion/DRUGS/split_boltz_10k.npy', help='Path of file defining the split')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Whether to use wandb')

    
    #data arguments      
    parser.add_argument('--data_dir', type=str, default= SCRATCH / 'torsional-diffusion/DRUGS/drugs/', help='Folder containing original conformers')
    parser.add_argument('--init_positions_path', type=str, default="/home/mila/l/lena-nehale.ezzine/ai4mols/torsional-diffusion/data/md_trajs_dict.pkl", help='Path to the initial positions of conformers')
    parser.add_argument('--use_synthetic_aug', type=bool, default= False, help='Whether to use synthetic augmentation')

    parser.add_argument('--train_smis', type=str, default=   "C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl  COC=O  c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N"  , help='train SMILES strings for which to generate conformers')
    parser.add_argument('--limit_train_mols', type=int, default=None, help='Limit to the number of molecules in dataset, 0 uses them all')
    parser.add_argument('--val_smis', type=str, default=   " CCS  C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl  COc1ccccc1  CC(=C)c1ccccc1  CCc1cccc2c1cccc2 "  , help='val SMILES strings for which to generate conformers')
    parser.add_argument('--dataset', type=str, default='drugs', help='drugs or qm9')
    parser.add_argument('--n_smis_batch', type=int, default=None, help='Number of SMILES strings per batch')
    parser.add_argument('--n_local_structures', type=int, default=1, help= 'Number of local structures per smile to sample')  #TODO change!
    parser.add_argument('--max_n_local_structures', type=int, default=1, help= 'Max Number of local structures per smile to sample from')  #set to +inf for mle/diffusion!
    parser.add_argument('--gt_data_path', type=str, default=None, help='Path to the ground truth data')
    
    
    
    #gflownet arguments
    parser.add_argument('--train_mode', type=str, default='gflownet', help='Training mode for GflowNets')
    parser.add_argument('--grad_acc', type=bool, default=True, help='Whether or not to use gradient accumulation')
    parser.add_argument('--p_expl', type=float, default=0.2, help='Exploration probability for GflowNets')
    parser.add_argument('--p_replay', type=float, default=0.2, help='Replay probability for GflowNets')
    parser.add_argument('--energy_fn', type=str, default='mmff', help='Energy function for GflowNets')
    parser.add_argument('--logrew_clamp', type=float, default=-1e5, help='Clamping value for log rewards')
    parser.add_argument('--rew_temp', type=float, default= 0.001987204118 * 298.15 , help='Temperature for rewards')
    parser.add_argument('--replay_buffer_size', type=int, default=500, help='Size of the replay buffer')
    parser.add_argument('--batch_size_train', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sigma_min', type=float, default=0.01*3.14, help='Minimum sigma used for training')
    parser.add_argument('--sigma_max', type=float, default=3.14, help='Maximum sigma used for training')
    parser.add_argument('--diffusion_steps', type=int, default=20, help='Number of diffusion steps')

    # eval args
    parser.add_argument('--batch_size_eval', type=int, default=1024, help='Batch size for evaluation')
    parser.add_argument('--num_points', type=int, default=30, help='Number of points for evaluation')
    parser.add_argument('--num_back_trajs', type=int, default=8, help='Number of backward trajectories for computing logpT')
    parser.add_argument('--ode', type=bool, default=False, help='Whether to use ODE for computing logpT')


    # other training arguments
    parser.add_argument('--num_sgd_steps', type=int, default=2048, help='Number of SGD steps for one epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of plateau scheduler')
    parser.add_argument('--boltzmann_weight', action='store_true', default=True, help='Whether to sample conformers based on B.w.')

    # Feature arguments
    parser.add_argument('--in_node_features', type=int, default=74, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
    parser.add_argument('--in_edge_features', type=int, default=4, help='Dimension of edge feature (do not change)')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
    parser.add_argument('--radius_embed_dim', type=int, default=50, help='Dimension of embedding of distances')
    
    # Model arguments
    parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=32, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=8, help='Number of hidden features per node of orser >0')
    parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')

    # Boltzmann training arguments
    parser.add_argument('--boltzmann_training', action='store_true', default=False, help='Set to true for torsional Boltzmann training')
    parser.add_argument('--boltzmann_confs', type=int, default=32, help='Number of conformers to generate at each resampling step')
    parser.add_argument('--boltzmann_steps', type=int, default=20, help='Number of inference steps used by the resampler')
    parser.add_argument('--likelihood', type=str, default='full', help='Method to evaluate likelihood: full (default) or hutch')
    parser.add_argument('--temp', type=int, default=300, help='Temperature used for Boltzmann weight')
    parser.add_argument('--adjust_temp', action='store_true', default=False, help='Whether to perform the temperature annealing during training')

    args = parser.parse_args()
    return args
