import math, os, torch, yaml
from torch.utils.data import Subset
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from gflownet.gfn_train import gfn_sgd 
from gflownet.gfn_metrics import log_gfn_metrics
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
from utils.boltzmann import BoltzmannResampler
from argparse import Namespace 
import copy 
import wandb

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



def train(args, model, optimizer, scheduler, train_loader, val_loader):

    print('Set up replay buffer, seed and Wandb')
     #ReplayBuffer = ReplayBufferClass(max_size = args.replay_buffer_size)
    ReplayBuffer = None
    seed_everything(args.seed)
    if args.use_wandb:
        wandb.login()
        run = wandb.init(project="gfn_torsional_diff")
        run.name = f"{args.train_mode}_{args.energy_fn}_{args.seed}_smi_{args.smis}_limit_train_mols_{args.limit_train_mols}"
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

    print("Starting GFN training ...")
    for epoch in range(args.n_epochs):
        if args.log_gfn_metrics:
            subset = Subset(dataset, list(range(5)))
            log_gfn_metrics(model, subset, optimizer, device, args.sigma_min, args.sigma_max, args.diffusion_steps, batch_size=args.batch_size_eval, T=args.rew_temp, num_points=args.num_points, logrew_clamp=args.logrew_clamp, energy_fn=args.energy_fn, num_trajs = args.num_trajs, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, train_mode = args.train_mode, gt_data_path = args.gt_data_path, seed = args.seed)
        #score = get_gt_score(gt_data_path, sigma_min, sigma_max, device, num_points, ix0, ix1, steps = 5)
        for _ in tqdm(range(args.num_sgd_steps)): 
            subset_indices = np.random.choice(len(dataset), args.n_smis_batch, replace=False)
            subset = Subset(dataset, subset_indices)
            results = gfn_sgd(model, subset, optimizer, device,  args.sigma_min, args.sigma_max, args.diffusion_steps, train = True, batch_size = args.batch_size_train ,  T=args.rew_temp,  logrew_clamp = args.logrew_clamp, energy_fn = args.energy_fn, train_mode = args.train_mode, use_wandb = args.use_wandb, ReplayBuffer = ReplayBuffer, p_expl = args.p_expl, p_replay = args.p_replay)
        print("Epoch {}: Training Loss {}".format(epoch, results[0]))
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
    train_loader, val_loader = construct_loader(args, boltzmann_resampler=boltzmann_resampler)

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # record parameters
    #yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    #save_yaml_file(yaml_file_name, args.__dict__) 
    args.device = device

    if args.boltzmann_training:
        boltzmann_train(args, model, optimizer, train_loader, val_loader, boltzmann_resampler)
    else:
        train(args, model, optimizer, scheduler, train_loader, val_loader)
