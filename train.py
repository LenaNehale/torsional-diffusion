import math, os, torch, yaml
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from gflownet.gfn_train import gfn_epoch, log_gfn_metrics
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
from utils.boltzmann import BoltzmannResampler
from argparse import Namespace
import copy
RDLogger.DisableLog('rdApp.*')

"""
    Training procedures for both conformer generation and Botzmann generators
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def train(args, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = math.inf
    best_epoch = 0

    print("Starting training (not boltzmann)...")
    for epoch in range(args.n_epochs):
        #train_loss, base_train_loss = train_epoch(model, train_loader, optimizer, device)
        #print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))
        sigma_max, sigma_min,steps, T =np.pi, 0.01 * np.pi, 20, 1.0
        conformers_train_gen, conformers_val_gen = log_gfn_metrics(model, train_loader, val_loader, optimizer, device, sigma_min, sigma_max, steps, n_trajs=4, T=T,  max_batches=3)
        train_loss, _, conformers = gfn_epoch(model, train_loader, optimizer, device,  sigma_min, sigma_max, steps, train = True, n_trajs = 4, max_batches=100, T=T)
        print("Epoch {}: Training Loss {}".format(epoch, train_loss))
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
    yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    if args.boltzmann_training:
        boltzmann_train(args, model, optimizer, train_loader, val_loader, boltzmann_resampler)
    else:
        train(args, model, optimizer, scheduler, train_loader, val_loader)
