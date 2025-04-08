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
    exp_path = f"{args.train_mode}_{args.energy_fn}_{args.seed}_limit_train_mols_{args.limit_train_mols}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_diffusion_steps_{args.diffusion_steps}_max_n_local_structures_{args.max_n_local_structures}_use_synthetic_aug_{args.use_synthetic_aug}_lr_{args.lr}"
    if args.limit_train_mols == 1 : 
        exp_path += f"_smi_{args.train_smis}" 
    if args.use_wandb:
        wandb.login()
        group = f"limit_train_mols_{args.limit_train_mols}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_diffusion_steps_{args.diffusion_steps}_lr_{args.lr}"
        if args.limit_train_mols == 1 : 
            group += f"_smi_{args.train_smis}" 
        run = wandb.init(project="gfn_torsional_diff", group = group )
        run.name = exp_path
    
    print("Starting GFN training ...")
    for epoch in range(args.n_epochs):        
        for k in tqdm(range(args.num_sgd_steps)):     
            if k % 5 == 0: 
                logpTs = []
                for smi in train_smis:
                    subset = make_dataset_from_smi([smi], init_positions_path=args.init_positions_path, n_local_structures = args.n_local_structures, max_n_local_structures = args.max_n_local_structures)
                    logpT = get_logpT(subset[smi], model, args.sigma_min, args.sigma_max,  args.diffusion_steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), ode = False, num_trajs = args.num_back_trajs)
                    logpTs.append(logpT)
                if args.use_wandb:
                    wandb.log({"logpT batch": torch.stack(logpTs).mean()})

            if k % 20 == 0:
                print('Saving the model ...')
                # Save the current model in a folder model_chkpts
                model_path = "/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFN/model_chkpts"
                if not os.path.exists(f'{model_path}'):
                    os.makedirs(f'{model_path}')
                torch.save(model.state_dict(), f'{model_path}/{exp_path}.pt')
                
                print('Saving replay buffer ...')
                if ReplayBuffer is not None:
                    replaybuffer_path = '/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFN/replay_buffer'
                    if not os.path.exists(replaybuffer_path):
                        os.makedirs(replaybuffer_path)
                    positions_dict, tas_dict = ReplayBuffer.get_positions_and_tas(train_smis)
                    pickle.dump([positions_dict, tas_dict], open(f'{replaybuffer_path}/{exp_path}.pkl', 'wb'))
                
                print('Generating gfn samples and learned energy landscape...')
                generated_stuff = generate_stuff(model, train_smis, args.n_smis_batch, args.batch_size_eval, args.diffusion_steps, args.rew_temp, args.logrew_clamp, args.energy_fn, args.device, args.sigma_min, args.sigma_max, args.init_positions_path, args.n_local_structures, args.max_n_local_structures, train_mode = 'gflownet')
                torch.cuda.empty_cache()
                plot_energy_samples_logpTs(model, train_smis, generated_stuff, args.energy_fn, args.logrew_clamp, args.init_positions_path, args.n_local_structures, args.max_n_local_structures, args.sigma_min, args.sigma_max,  args.diffusion_steps, args.device, args.num_points, args.num_back_trajs, args.rew_temp,  plot_energy_landscape = True, plot_sampled_confs = True, plot_pt = True, use_wandb = args.use_wandb, exp_path = exp_path, timestep = f"_time_{k}", ode = args.ode )
                
    
            idx_train = np.random.randint(0, args.limit_train_mols, size = args.n_smis_batch)
            subset = make_dataset_from_smi(train_smis[idx_train], init_positions_path=args.init_positions_path, n_local_structures = args.n_local_structures, max_n_local_structures = args.max_n_local_structures)
            results = gfn_sgd(model, subset, optimizer, device,  args.sigma_min, args.sigma_max, args.diffusion_steps, train = True, batch_size = args.batch_size_train ,  T=args.rew_temp,  logrew_clamp = args.logrew_clamp, energy_fn = args.energy_fn, train_mode = args.train_mode, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, p_expl = args.p_expl, p_replay = args.p_replay, grad_acc = args.grad_acc, use_synthetic_aug = args.use_synthetic_aug)      
        
        print("Epoch {}: Training Loss {}".format(epoch, torch.mean(results[0])))
    



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
