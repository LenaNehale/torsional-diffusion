from diffusion.sampling import *

import math, os, torch, yaml
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
#from utils.boltzmann import BoltzmannResampler
from argparse import Namespace

import numpy as np
from tqdm import tqdm
import torch
import diffusion.torus as torus
import time


RDLogger.DisableLog('rdApp.*')

"""
    Training procedures for conformer generation using GflowNets.
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def sample_and_get_loss(conformers, model, device, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,ode=False, likelihood=None, pdb=None, energy_fn='mmff', T = 32):

    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)
    eps = 1 / steps    
    
    for batch_idx, data in enumerate(loader):
        bs = data.num_graphs
        n_torsion_angles = len(data.total_perturb[0]) 
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))
        dlogp = torch.zeros(bs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            #with torch.no_grad():
            data_gpu = model(data_gpu)
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()
            t = sigma_idx / steps   # t is really 1-t
            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood) # warning: this function changes data_gpu.pos
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            mean, std = g ** 2 * eps * score, g * np.sqrt(eps)
            
            # compute the forward and backward (in gflownet language) transitions logprobs
            for i in range(bs):
                start, end = i*n_torsion_angles, (i+1)*n_torsion_angles
                # in forward, the new mean is obtained using the score (see above)
                p_trajs_forward = torus.p_differentiable((perturb - mean)[start:end], std ) 
                logit_pf[i,sigma_idx ] += torch.log(p_trajs_forward).sum() 
                # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
                sigma_b = sigma_schedule[sigma_idx + 1]
                g_b = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
                std_b = g_b * np.sqrt(eps)
                p_trajs_backward = torus.p_differentiable(  perturb[start:end] , std_b) 
                logit_pb[i,sigma_idx ] += torch.log(p_trajs_backward).sum() 

            conf_dataset.apply_torsion_and_update_pos(data, perturb.detach().numpy())
            data_gpu.pos = data.pos.to(device)

            if pdb:
                for conf_idx in range(bs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()
        
        logit_pf = reduce(logit_pf, 'bs steps-> bs', 'sum' )
        logit_pb = reduce(logit_pb, 'bs steps-> bs', 'sum' )
        '''
        # Sanity check: compute exact likelihood of samples using the reverse ODE
        logp = torch.zeros(bs)
        data_likelihood = copy.deepcopy(data)
        data_likelihood_gpu = copy.deepcopy(data_gpu) 
        conf_dataset_likelihood = InferenceDataset(copy.deepcopy(conformers))
        for sigma_idx, sigma in enumerate(reversed(sigma_schedule[1:])):
            data_likelihood_gpu.node_sigma = sigma * torch.ones(data_likelihood.num_nodes, device=device)
            #with torch.no_grad():
            data_likelihood_gpu = model(data_likelihood_gpu)
            ## apply reverse ODE perturbation
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_likelihood_gpu.edge_pred.shape)
            score = data_likelihood_gpu.edge_pred.cpu()
            perturb =  - 0.5 * g ** 2 * eps * score
            conf_dataset_likelihood.apply_torsion_and_update_pos(data_likelihood, perturb.numpy()) # BUG! This changes the position in the conformers object
            ## compute dlogp
            div = divergence(model, data_likelihood, data_likelihood_gpu, method='hutch' if likelihood is None else likelihood)
            logp += -0.5 * g ** 2 * eps * div
            data_likelihood_gpu.pos = data_likelihood.pos.to(device)
        
        print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])
        '''
    #Get VarGrad Loss
    try:
        #assert all(x == data.name[0] for x in data.name) 
        assert all(x == data.canonical_smi[0] for x in data.canonical_smi)
    except:
        raise ValueError("Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs")
    #Computing logrews
    pos = rearrange(data.pos, '(bs n) d -> bs n d', bs=bs)
    z = rearrange(data.z, '(bs n) -> bs n', bs=bs)
    if energy_fn == 'mmff':
        logrews = - torch.Tensor([mmff_energy(pyg_to_mol(data.mol[i], conformers[i], mmff=True, rmsd=False)) for i in range(bs)])/T #previous: logrews = - torch.Tensor([mmff_energy(mol) for mol in data.mol])/T. This is wrong ! mol positions don't get updated during sampling
    else:
        raise ValueError("Energy function not implemented")       
    vargrad_quotients = logit_pf - logit_pb - logrews
    vargrad_loss = torch.var(vargrad_quotients)
    #print('vargrad loss', vargrad_loss)
    
    return conformers, vargrad_loss


def train_gfn_epoch(model, loader, optimizer, device):
    model.train()
    loss_tot = 0
    base_tot = 0

    for batch in tqdm(loader, total=len(loader)): # Here, loader is used to go through smiles
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = []        
        for i in range(len(batch)):
            # print time that it takes 
            data = batch[i]
            samples = []
            for _ in range(2):
                data_new = copy.deepcopy(data)
                samples.append(data_new)   
            samples = perturb_seeds(samples) # apply uniform noise to torsion angles  
            conformers, loss_smile = sample_and_get_loss(samples, model, device=device) #on-policy
            loss.append(loss_smile) 
        print('loss', torch.stack(loss).mean())
        torch.stack(loss).mean().backward() #Not possible because grads are taken from a table & need to detach gradients :( )
        optimizer.step()
        torch.cuda.empty_cache()
        loss_tot += torch.Tensor(loss).mean().item()


    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    print('train loss:', loss_avg)
    return loss_avg, base_avg



'''


def gfn_train(args, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = math.inf
    best_epoch = 0

    print("Starting training (not boltzmann)...")
    for epoch in range(args.n_epochs):
        train_loss, base_train_loss = train_epoch(model, train_loader, optimizer, device)
        print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))

        val_loss, base_val_loss = test_epoch(model, val_loader, device)
        print("Epoch {}: Validation Loss {} base loss {}".format(epoch, val_loss, base_val_loss))

        if scheduler:
            scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(args.log_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))



if __name__ == '__main__':
    args = parse_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    if args.restart_dir:
        with open(f'{args.restart_dir}/model_parameters.yml') as f:
            args_old = Namespace(**yaml.full_load(f))

        model = get_model(args_old).to(device)
        state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)

    else:
        model = get_model(args).to(device)

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
'''