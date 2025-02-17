import math, os, torch, yaml
from torch.utils.data import Subset
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader, make_dataset_from_smi
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from gflownet.gfn_train import gfn_sgd 
from gflownet.gfn_metrics import log_gfn_metrics, log_gfn_metrics_cond
from gflownet.replay_buffer import ReplayBufferClass
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
from utils.boltzmann import BoltzmannResampler
from argparse import Namespace 
import copy 
import wandb
import pickle

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



def train(args, model, optimizer, scheduler):

    if args.data_name == 'geomdrugs':
        train_loader, val_loader = construct_loader(args, boltzmann_resampler=boltzmann_resampler)
        print('Set up dataset')
        if args.smis is None:
            dataset = train_loader.dataset
        
        else:
            assert len(args.smis) > args.n_smis_batch
            ixs = []
            for i in range(len(train_loader.dataset)):
                if train_loader.dataset[i].canonical_smi in args.smis:
                    ixs.append(i)
                    print(i)
            dataset = Subset(train_loader.dataset, ixs)   
    elif args.data_name == 'freesolv':
        #freesolv_smis = np.array(pickle.load(open("/home/mila/l/lena-nehale.ezzine/ai4mols/torsional-diffusion/freesolv_subset_valid_smis_workshop.pkl"  ,'rb')))
        freesolv_smis = args.smis.split()
        args.limit_train_mols = len(freesolv_smis)
        if args.fix_local_structures:
            assert args.init_positions_path is not None
            train_confs_init = make_dataset_from_smi(freesolv_smis, args.init_positions_path)
    else:
        raise ValueError('Dataset not recognized!')
        
    print('Set up replay buffer, seed and Wandb')
    #ReplayBuffer = ReplayBufferClass(max_size = args.replay_buffer_size)
    if args.data_name == 'freesolv':
        if args.p_replay > 0:
            ReplayBuffer = ReplayBufferClass(smis_dataset = freesolv_smis, max_size = args.replay_buffer_size)
        else:
            ReplayBuffer = None
    else:
        raise ValueError('Please define a smiles dataset for the replay buffer!')
    seed_everything(args.seed)
    #exp_path = f"{args.train_mode}_{args.energy_fn}_{args.seed}_limit_train_mols_{args.limit_train_mols}_dataset_{args.data_name}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_smis_{args.smis}"
    exp_path = f"{args.train_mode}_{args.energy_fn}_{args.seed}_limit_train_mols_{args.limit_train_mols}_dataset_{args.data_name}_p_replay_{args.p_replay}_p_expl_{args.p_expl}_diffusion_steps_{args.diffusion_steps}"
    if args.limit_train_mols == 1 : 
        exp_path += f"_smi_{args.smis}"
    if args.use_wandb:
        wandb.login()
        run = wandb.init(project="gfn_torsional_diff")
        run.name = exp_path
    

    print("Starting GFN training ...")
    for epoch in range(args.n_epochs):        
        if args.log_gfn_metrics:
            if args.data_name == 'geomdrugs':
                subset = Subset(dataset, [len(dataset) - i - 1 for i in range(len(dataset * 0.2))] ) # val subset
            elif args.data_name == 'freesolv':
                idx_val = np.arange(0, args.limit_train_mols // 5) + args.limit_train_mols
                subset = make_dataset_from_smi(np.array(freesolv_smis)[idx_val])
            log_gfn_metrics(model, subset, optimizer, device, args.sigma_min, args.sigma_max, args.diffusion_steps, batch_size=args.batch_size_eval, T=args.rew_temp, num_points=args.num_points, logrew_clamp=args.logrew_clamp, energy_fn=args.energy_fn, num_trajs = args.num_trajs, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, train_mode = args.train_mode, gt_data_path = args.gt_data_path, seed = args.seed)
            log_gfn_metrics_cond(model, dataset, optimizer, device, args.sigma_min, args.sigma_max, args.diffusion_steps, args.n_smis_batch, args.batch_size_eval, args.rew_temp  ,  args.logrew_clamp, args.energy_fn,  args.num_trajs, args.use_wandb, ReplayBuffer, args.train_mode, args.seed)
        #score = get_gt_score(gt_data_path, sigma_min, sigma_max, device, num_points, ix0, ix1, steps = 5)
        for k in tqdm(range(args.num_sgd_steps)): 
            if args.data_name == 'geomdrugs':
                subset_indices = np.random.choice(int(len(dataset)*0.8), args.n_smis_batch, replace=False)
                subset = Subset(dataset, subset_indices)
            elif args.data_name == 'freesolv':
                idx_train = np.random.randint(0, args.limit_train_mols, size = args.n_smis_batch)
                if args.fix_local_structures:
                    subset = np.array(train_confs_init)[idx_train]
                else:
                    subset = make_dataset_from_smi(np.array(freesolv_smis)[idx_train])
                subset = make_dataset_from_smi(np.array(freesolv_smis)[idx_train])
            results = gfn_sgd(model, subset, optimizer, device,  args.sigma_min, args.sigma_max, args.diffusion_steps, train = True, batch_size = args.batch_size_train ,  T=args.rew_temp,  logrew_clamp = args.logrew_clamp, energy_fn = args.energy_fn, train_mode = args.train_mode, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, p_expl = args.p_expl, p_replay = args.p_replay, grad_acc = args.grad_acc)
            
            if k % 100 == 0:
                print('Saving the model ...')
                # Save the current model in a folder model_chkpts
                model_path = "/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFNmodel_chkpts"
                if not os.path.exists(f'{model_path}'):
                    os.makedirs(f'{model_path}')
                torch.save(model.state_dict(), f'{model_path}/{exp_path}.pt')
                print('Model saved!')
                
                print('Saving replay buffer ...')
                if ReplayBuffer is not None:
                    replaybuffer_path = '/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFNreplay_buffer'
                    if not os.path.exists(replaybuffer_path):
                        os.makedirs(replaybuffer_path)
                    positions_dict = ReplayBuffer.get_positions(freesolv_smis)
                    pickle.dump(positions_dict, open(f'{replaybuffer_path}/{exp_path}.pkl', 'wb'))
                print('replay buffer saved!')
                '''
                print('Saving the conformers ...')
                # Save files for eval plots 
                conformers_rdkit = make_dataset_from_smi(freesolv_smis)
                _ , conformers_gen, _, _, logrews_gen, _, _ = gfn_sgd(model, conformers_rdkit  , optimizer, device,  args.sigma_min, args.sigma_max, args.diffusion_steps, train=False, batch_size = 2500, T= args.rew_temp , logrew_clamp = args.logrew_clamp, energy_fn=args.energy_fn, train_mode='gflownet', use_wandb = False, ReplayBuffer = None, p_expl = 0.0, p_replay = 0.0, grad_acc = False)

                confs_path = '/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFNconformers'
                if not os.path.exists(confs_path):
                    os.makedirs(confs_path)
                pickle.dump([conformers_gen , logrews_gen], open(f'{confs_path}/{exp_path}.pkl', 'wb'))
                print('Conformers saved!')

                
                # Save the replay buffer in pickle file
                if ReplayBuffer is not None:
                    if not os.path.exists('~/scratch/torsionalGFNreplay_buffer'):
                        os.makedirs('~/scratch/torsionalGFNreplay_buffer')
                    replay_buffer_path = f'~/scratch/torsionalGFNreplay_buffer/{exp_path}.pkl'
                    with open(replay_buffer_path, 'wb') as f:
                        pickle.dump(ReplayBuffer, f)
                '''
        print("Epoch {}: Training Loss {}".format(epoch, torch.mean(results[0])))
    '''
        val_loss, base_val_loss = test_epoch(model, val_loader, device) 
        print("Epoch {}: Validation Loss {} base loss {}".format(epoch, val_loss, base_val_loss))

        if scheduler:
            scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            #torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

        #torch.save({'epoch': epoch,'model': model.state_dict(),'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict(),}, os.path.join(args.log_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    '''


def boltzmann_train(args, model, optimizer, train_loader, val_loader, resampler):
    print("Starting Boltzmann training...")

    val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
    print(f"Initial val ESS: Mean {np.mean(val_ess):.4f} Median {np.median(val_ess):.4f}")
    best_val = val_ess

    for epoch in range(args.n_epochs):
        if args.adjust_temp:
            train_loader.dataset.boltzmann_resampler.temp = (3000 - args.temp) / (epoch + 1) + args.temp

        train_loss, base_train_loss = train_epoch(model, train_loader, optimizer, device)
        print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))
        if epoch % 5 == 0:
            val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
            print(f"Epoch {epoch} val ESS: Mean {np.mean(val_ess).item():.4f} Median {np.median(val_ess):.4f}")

            if best_val > val_ess:
                best_val = val_ess
                #torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

            #torch.save({'epoch': epoch,'model': model.state_dict(),'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict(),}, os.path.join(args.log_dir, 'last_model.pt'))


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

    # construct loader and set device
    if args.boltzmann_training:
        boltzmann_resampler = BoltzmannResampler(args, model)
    else:
        boltzmann_resampler = None

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # record parameters
    #yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    #save_yaml_file(yaml_file_name, args.__dict__) 
    args.device = device
    train(args, model, optimizer, scheduler)
