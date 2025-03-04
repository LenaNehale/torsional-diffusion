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
    parser.add_argument('--data_dir', type=str, default= SCRATCH / 'torsional-diffusion/DRUGS/drugs/', help='Folder containing original conformers')
    parser.add_argument('--std_pickles', type=str, default=SCRATCH / 'torsional-diffusion/DRUGS/standardized_pickles', help='Folder in which the pickle are put after standardisation/matching')
    parser.add_argument('--split_path', type=str, default=SCRATCH / 'torsional-diffusion/DRUGS/split_boltz_10k.npy', help='Path of file defining the split')
    parser.add_argument('--dataset', type=str, default='drugs', help='drugs or qm9')
    parser.add_argument('--n_local_structures', type=int, default=32, help= 'Number of local structures per smile to sample')  
    parser.add_argument('--max_n_local_structures', type=int, default=2001, help= 'Max Number of local structures per smile to sample from')  #TODO  set to +inf for mle/diffusion!

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--init_positions_path', type=str, default="/home/mila/l/lena-nehale.ezzine/ai4mols/torsional-diffusion/data/md_trajs_dict.pkl", help='Path to the initial positions of conformers')

    
    #GflowNet arguments
    parser.add_argument('--train_smis', type=str, default=   "CC(C)C=C COc1ccccc1 CCCCCBr CCOC(=O)c1ccc(cc1)O CC(C)CO CC(C)O CCOP(=S)(OCC)SCSCC CC=C(C)C CCCCCCBr CCOC(=O)c1ccccc1 COc1cccc(c1O)OC COC(OC)OC CCCCC(=O)OCC CC(C)(C)C(=O)OC CCC[C@@H](C)O CC(C)OC(C)C COc1ccccc1N CCCCCCCCC=O CC(=CCC/C(=C/CO)/C)C CCCCCCl CC[C@H](C)c1ccccc1 CCCCCCCC(=O)C CCOC(=O)CCC(=O)OCC CC(C)COC(=O)C CC[C@H](C)O CC(=C)C=C CCSC CC(=O)OCC(COC(=O)C)OC(=O)C CC(C)CCO CC[C@H](C(C)C)O CCCc1ccc(c(c1)OC)O CCCCCCCC(=O)OC CC(=CCC/C(=C\\CO)/C)C Cn1cnc2c1c(=O)n(c(=O)n2C)C Cc1cccc(c1C)O C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl COc1ccccc1OC CC(C)(C)Br CC(C)CCOC=O Cc1cc2ccccc2cc1C C1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O CCCCC=O CCCc1ccccc1 CC(C)c1ccccc1 CCCCCCN CCCOCCC C[C@@H]1CCCC[C@@H]1C CC(=C)c1ccccc1 CCc1cccc(c1)O C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O CCOP(=O)(OCC)OCC CC(=O)Oc1ccccc1C(=O)O CCc1cccc2c1cccc2 c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N c1ccc(c(c1)C(F)(F)F)C(F)(F)F CCCCOCCCC CC(C)(/C=N\\OC(=O)NC)SC COCC(OC)(OC)OC CN(C)C(=O)Nc1ccccc1 CCCC(C)(C)C c1cc(cc(c1)O)C#N CCOCCO c1ccc(cc1)O[C@@H](C(F)F)F CCC c1ccc(cc1)CCCO COC(=O)C(F)(F)F CCS CC(C)(C)C C[C@H](CC(C)C)O CCC(C)CC CCCCCCC COC=O"  , help='train SMILES strings for which to generate conformers')
    parser.add_argument('--val_smis', type=str, default=   " CCS  C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl  COc1ccccc1  CC(=C)c1ccccc1  CCc1cccc2c1cccc2 "  , help='val SMILES strings for which to generate conformers')

    parser.add_argument('--gt_data_path', type=str, default=None, help='Path to the ground truth data')
    parser.add_argument('--train_mode', type=str, default='gflownet', help='Training mode for GflowNets')
    parser.add_argument('--p_expl', type=float, default=0.0, help='Exploration probability for GflowNets')
    parser.add_argument('--p_replay', type=float, default=0.0, help='Replay probability for GflowNets')
    parser.add_argument('--energy_fn', type=str, default='mmff', help='Energy function for GflowNets')
    parser.add_argument('--logrew_clamp', type=float, default=-1e5, help='Clamping value for log rewards')
    parser.add_argument('--rew_temp', type=float, default= 0.001987204118 * 298.15 , help='Temperature for rewards')
    parser.add_argument('--replay_buffer_size', type=int, default=500, help='Size of the replay buffer')
    parser.add_argument('--batch_size_train', type=int, default=16, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--n_smis_batch', type=int, default=2, help='Number of SMILES strings per batch')
    parser.add_argument('--num_sgd_steps', type=int, default=2048, help='Number of SGD steps for one epoch')
    parser.add_argument('--num_points', type=int, default=10, help='Number of points for evaluation')
    parser.add_argument('--num_trajs', type=int, default=8, help='Number of backward trajectories for computing logpT')
    parser.add_argument('--diffusion_steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--grad_acc', type=bool, default=True, help='Whether or not to use gradient accumulation')
    


    # Training arguments
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Whether to use wandb')
    parser.add_argument('--log_gfn_metrics', action='store_true', default=False, help='Whether to log GFN metrics')
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of plateau scheduler')
    parser.add_argument('--sigma_min', type=float, default=0.01*3.14, help='Minimum sigma used for training')
    parser.add_argument('--sigma_max', type=float, default=3.14, help='Maximum sigma used for training')
    parser.add_argument('--limit_train_mols', type=int, default=80, help='Limit to the number of molecules in dataset, 0 uses them all')
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
