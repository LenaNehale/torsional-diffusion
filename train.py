import math, os, torch, yaml
from torch.utils.data import Subset 
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader, make_dataset_from_smi
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from gflownet.gfn_train import gfn_sgd, get_logpT 
from gflownet.replay_buffer import ReplayBufferClass
from utils.utils import get_model, get_optimizer_and_scheduler
from argparse import Namespace 
import copy 
import wandb
import pickle
from gflownet.make_eval_plots import * 

RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

"""
    Training procedures for both conformer generation and Botzmann generators
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def train(args, model, optimizer):

    if args.train_mode in ['diffusion', 'mle']:
        assert args.init_positions_path is not None, 'init_positions_path must be provided for diffusion and mle training'

    train_smis, val_smis = np.array(args.train_smis.split()), np.array(args.val_smis.split())
    args.limit_train_mols = len(train_smis)
    args.n_smis_batch = min(len(train_smis), 5)
    print('Set up replay buffer, seed and Wandb')
    if args.p_replay > 0:
        ReplayBuffer = ReplayBufferClass(smis_dataset = train_smis, max_size = args.replay_buffer_size)
    else:
        ReplayBuffer = None
    seed_everything(args.seed)
    exp_path = f"{args.train_mode}_{args.energy_fn}_{args.seed}_n_mols_{args.limit_train_mols}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_steps_{args.diffusion_steps}_max_n_local_structures_{args.max_n_local_structures}_use_synthetic_aug_{args.use_synthetic_aug}_lr_{args.lr}_T_{args.rew_temp}"
    root_path = args.root_path
    if args.limit_train_mols == 1 : 
        exp_path += f"_smi_{args.train_smis}" 
    if args.use_wandb:
        wandb.login()
        group = f"n_mols_{args.limit_train_mols}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_steps_{args.diffusion_steps}_lr_{args.lr}_T_{args.rew_temp}_nlayers_{args.num_conv_layers}_icml"
        if args.limit_train_mols == 1 : 
            group += f"_smi_{args.train_smis}" 
        run = wandb.init(project="gfn_torsional_diff", group = group[:127] )
        run.name = exp_path
    
    print("Starting GFN training ...")
    for k in tqdm(range(args.num_sgd_steps)):     
        if k % 1000 == 0:
            print('Saving the model ...')
            save_model(model, root_path, exp_path)
            
            
            print('Saving replay buffer ...') #TODO ugly, rewrite
            if ReplayBuffer is not None:
                positions_dict, tas_dict = ReplayBuffer.get_positions_and_tas(train_smis)
                save_rb(positions_dict, tas_dict, root_path, exp_path, sgd_step = k)
            else:
                positions_dict, tas_dict = None, None
            
            
            if args.run_eval:
                
                print('testing generalisation to other unseen local structures ...')
                av_smis, av_labels =  list(train_smis) + list(val_smis), ['train'] * len(train_smis) + ['val'] * len(val_smis)
                generated_stuff_all_ls = generate_stuff(model, av_smis, args.n_smis_batch, 1, args.diffusion_steps, args.rew_temp, args.logrew_clamp, args.energy_fn, args.device, args.sigma_min, args.sigma_max, args.init_positions_path, 500, np.inf, exp_path, sgd_step = k, train_mode = 'gflownet',  root_path = args.root_path )        
                plot_energies_hist(av_smis, av_labels, generated_stuff_all_ls, rew_temp = args.rew_temp, root_path = args.root_path , exp_path = exp_path , sgd_step=k,  save=True)
                torch.cuda.empty_cache()

                
                print('Generating gfn samples and learned energy landscape for the seen local structures...')
                generated_stuff = generate_stuff(model, train_smis, args.n_smis_batch, args.batch_size_eval, args.diffusion_steps, args.rew_temp, args.logrew_clamp, args.energy_fn, args.device, args.sigma_min, args.sigma_max, args.init_positions_path, args.n_local_structures, args.max_n_local_structures, exp_path, sgd_step = k, train_mode = 'gflownet', root_path = args.root_path)
                torch.cuda.empty_cache()
                assert args.n_local_structures == 1, 'for now plotting the learned energy landscape only works for n_local_structures = 1'
                plot_energy_samples_logpTs(model, train_smis, generated_stuff, args.energy_fn, args.logrew_clamp, args.init_positions_path, args.n_local_structures, args.max_n_local_structures, args.sigma_min, args.sigma_max,  args.diffusion_steps, args.device, args.num_points, args.num_back_trajs, args.rew_temp,  plot_energy_landscape = True, plot_sampled_confs = True, plot_pt = True, use_wandb = args.use_wandb, root_path = args.root_path, exp_path = exp_path, sgd_step = k, ode = args.ode, replay_tas= tas_dict )
                plot_kde_2d(generated_stuff, root_path, exp_path, sgd_step = k)

                
                
        
                print('Computing loglikelihood of a batch of MD data ...')
                logpTs = []
                for smi in train_smis:
                    subset = make_dataset_from_smi([smi], init_positions_path=args.init_positions_path, n_local_structures = 64, max_n_local_structures = np.inf) 
                    logpT = get_logpT(subset[smi], model, args.sigma_min, args.sigma_max,  args.diffusion_steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), ode = False, num_trajs = args.num_back_trajs) # effectuve bs: n_local_structures * num_back_trajs
                    logpTs.append(logpT)
                    print(f"LogpT {smi}: {logpT.mean()}")
                    if args.use_wandb:
                        wandb.log({f"logpT {smi}": logpT.mean()})
        
            
    
        
        
        idx_train = np.random.randint(0, args.limit_train_mols, size = args.n_smis_batch)
        subset = make_dataset_from_smi(train_smis[idx_train], init_positions_path=args.init_positions_path, n_local_structures = args.n_local_structures, max_n_local_structures = args.max_n_local_structures)
        results = gfn_sgd(model, subset, optimizer, device,  args.sigma_min, args.sigma_max, args.diffusion_steps, train = True, batch_size = args.batch_size_train ,  T=args.rew_temp,  logrew_clamp = args.logrew_clamp, energy_fn = args.energy_fn, train_mode = args.train_mode, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, p_expl = args.p_expl, p_replay = args.p_replay, grad_acc = args.grad_acc, use_synthetic_aug = args.use_synthetic_aug, sgd_step = k)      




if __name__ == '__main__':
    args = parse_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    if args.restart_dir:
        with open(f'{args.restart_dir}/model_parameters.yml') as f:
            args_old = Namespace(**yaml.full_load(f))

        model_ = get_model(args_old).to(device)
        state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
        model_.load_state_dict(state_dict, strict=True)

    else:
        model_ = get_model(args).to(device)
    model = copy.deepcopy(model_) # this is to  make sure that model_is not changed during training, and that we always initialize with the base diffuison model
    del model_
    numel = sum([p.numel() for p in model.parameters()])
    print(f'Model has {numel} parameters')

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    args.device = device
    train(args, model, optimizer)