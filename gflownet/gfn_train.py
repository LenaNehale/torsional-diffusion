from diffusion.sampling import * 
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import numpy as np
from tqdm import tqdm
import torch
import diffusion.torus as torus 
import time 
import contextlib
from utils.dataset import ConformerDataset
from utils.featurization import drugs_types
from torch_geometric.data import Data, Batch 
import torch_geometric


from rdkit.Chem import rdMolAlign, rdDepictor, Draw
from utils.standardization import fast_rmsd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
import itertools
import wandb

"""
    Training procedures for conformer generation using GflowNets.
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""





def sample_forward_trajs(conformers_input, model, train, sigma_min, sigma_max,  steps, device, p_expl, sample_mode):
    '''
    Sample forward trajectories.
    Args:
    - conformers_input (list): List of PyTorch geometric data objects representing conformers.
    - model (torch.nn.Module): Score model.
    - train (bool): Whether the model is in training mode.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - steps (int): Number of timesteps.
    - device (torch.device): CUDA or CPU device.
    Returns:
    - traj (list): List of PyTorch geometric batch objects representing a batch of conformers at each timestep of the trajectory.
    '''
    data = copy.deepcopy(Batch.from_data_list(conformers_input))
    data_gpu = copy.deepcopy(data).to(device)
    #data.total_perturb = torch.zeros(len(data)* data.mask_rotate[0].shape[0])
    bs, n_torsion_angles = len(data), data.mask_rotate[0].shape[0]
    sigma_schedule = 10 ** np.linspace( np.log10(sigma_max), np.log10(sigma_min), steps + 1 )
    eps = 1 / steps
    traj = [copy.deepcopy(data)]
    mols = [[pyg_to_mol(data.mol[i], data[i], copy=True) for i in range(bs)]]
    logit_pf = torch.zeros(bs, len(sigma_schedule))
    logit_pb = torch.zeros(bs, len(sigma_schedule))   
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data_gpu = model(data_gpu)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape) 
        score = data_gpu.edge_pred.cpu()        
        if train and random.random() < p_expl:
            noise_scale = 5 # Set to a higher value for more noise (hence more exploration)
        else:
            noise_scale = 1
        perturb = g**2 * eps * score + noise_scale * g * np.sqrt(eps) * z
        # Get PF and PB
        if not sample_mode:
            mean, std = g**2 * eps * score, g * np.sqrt(eps)
            p_trajs_forward = torus.p_differentiable( (perturb.detach() - mean), std)
            logit_pf[:, sigma_idx] += torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
            # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
            sigma_b = sigma_schedule[sigma_idx + 1]
            g_b = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            std_b = g_b * np.sqrt(eps)
            p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b)
            logit_pb[:, sigma_idx] += torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)
        new_pos = perturb_batch(data, perturb) 
        data.pos = new_pos 
        data_gpu.pos = data.pos.to(device)
        data.total_perturb = (data.total_perturb + perturb.detach() ) % (2 * np.pi)
        data_gpu.total_perturb = data.total_perturb.to(device)
        traj.append(copy.deepcopy(data))  
        mols.append([pyg_to_mol(data.mol[i], data[i], copy=True) for i in range(bs)])
    
    # get rmsds of noised conformers compared to traj[0]
    #rmsds = [get_rmsds(mols[i], mols[0]) for i in range(len(mols))]
    #print('RMSDs:', [np.mean(r) for r in rmsds])
    if not sample_mode:
        logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
        logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
    else:
        logit_pf, logit_pb = None, None
    return traj, logit_pf, logit_pb

def sample_backward_trajs(conformers_input, sigma_min, sigma_max,  steps):
    '''
    Sample backward trajectories.
    Args:
    - conformers_input (list): List of PyTorch geometric data objects representing conformers.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - steps (int): Number of timesteps.
    Returns:
    - traj (list): List of PyTorch geometric batch objects representing a batch of conformers at each timestep of the trajectory.
    '''
    data = copy.deepcopy(Batch.from_data_list(conformers_input))
    bs, n_torsion_angles = len(data), data.mask_rotate[0].shape[0]
    sigma_schedule = 10 ** np.linspace( np.log10(sigma_max), np.log10(sigma_min), steps + 1 )
    eps = 1 / steps
    traj = [copy.deepcopy(data)]
    for _, sigma_b in enumerate(reversed(sigma_schedule[1:])):
        g = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=[n_torsion_angles *  len(data.mol)])
        perturb = g * np.sqrt(eps) * z  
        new_pos = perturb_batch(data, perturb)  # the minus sign is because we are going backwards
        data.pos = new_pos 
        data.total_perturb = (data.total_perturb + perturb.detach() ) % (2 * np.pi)
        traj.append(copy.deepcopy(data))   
    #reverse traj
    traj = traj[::-1]
    return traj

def get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=True):
    '''
    Compute the logits of forward and backward trajectories (logits are not normalized).
    Args:
    - traj (list): List of PyTorch geometric batch objects representing a batch of comformers at each timestep of the trajectory.
    - model (torch.nn.Module): Score model.
    - device (torch.device): CUDA or CPU device.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - steps (int): Number of timesteps.
    - likelihood (bool): Whether or not to compute the marginal logpT for the final states, and the correlation between their logit_pf - logit_pb and logpT.
    - train (bool): Whether the model is in training mode.
    Returns:
    - logit_pf (torch.Tensor): Logits of forward trajectory.
    - logit_pb (torch.Tensor): Logits of backward trajectory.
    - logp (torch.Tensor): Log-likelihood of final states.
    '''
    #print('smi:', traj[-1][0].canonical_smi)
    bs = len(traj[-1])
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)
    eps = 1 / steps
    n_torsion_angles = len( traj[-1][0].total_perturb)
    logit_pf = torch.zeros(bs, len(sigma_schedule))
    logit_pb = torch.zeros(bs, len(sigma_schedule))   
    
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data = traj[sigma_idx]
        data_gpu = copy.deepcopy(data).to(device)
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data_gpu = model(data_gpu)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        score = data_gpu.edge_pred.cpu()       
        mean, std = g**2 * eps * score, g * np.sqrt(eps)
        perturb = traj[sigma_idx +1 ].total_perturb - traj[sigma_idx].total_perturb
        # compute the forward and backward (in gflownet language) transitions logprobs
        # in forward, the new mean is obtained using the score (see above)
        p_trajs_forward = torus.p_differentiable( (perturb.detach() - mean), std)
        logit_pf[:, sigma_idx] += torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
        # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
        sigma_b = sigma_schedule[sigma_idx + 1]
        g_b = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        std_b = g_b * np.sqrt(eps)
        p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b)
        logit_pb[:, sigma_idx] += torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)
        
    logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
    logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
    
    if likelihood:
        logp = get_logpT(traj[-1].to_data_list(), model, sigma_min, sigma_max,  steps, ode = False)
        #print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])
    else:
        logp = None
    return logit_pf, logit_pb, logp



def get_logrew(data, T , energy_fn = 'mmff', clamp = -1e5):
    if energy_fn == 'mmff':
        if type(data) == torch_geometric.data.data.Data:
            energies = [mmff_energy(pyg_to_mol(data.mol, data))]
        else:
            energies = [mmff_energy(pyg_to_mol(data.mol[i], data[i])) for i in range(len(data))]
        energies = torch.Tensor(energies)
        logrews = -  energies / T
    elif energy_fn == 'dummy':
        total_perturb = data.total_perturb
        bs = len(data)
        num_torsion_angles = total_perturb.reshape(bs, -1).shape[1]
        mean = torch.ones(num_torsion_angles)
        sigma = 0.05 
        mvn = multivariate_normal(mean = mean, cov = sigma * torch.eye(len(mean)), allow_singular = False, seed = 42)
        probs = mvn.pdf(total_perturb.reshape(bs, -1).cpu().numpy())
        if bs == 1 : 
            probs = [probs]
        logrews = torch.Tensor(np.log(probs)) / T
    else:
        raise  NotImplementedError(f"Energy function {energy_fn} not implemented!")
    logrews[logrews<clamp] = clamp
    return logrews

def get_loss(traj, logit_pf, logit_pb, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn="dummy", T=1.0, train=False, loss='vargrad', logrew_clamp = -1e5):
    '''
    Computes the vargrad/TB loss.
    Args:
    - traj (list): List of PyTorch geometric batch objects representing a batch of comformers at each timestep of the trajectory.
    - model (torch.nn.Module): Score model.
    - device (torch.device): CUDA or CPU device.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - steps (int): Number of timesteps.
    - likelihood (bool): Whether or not to compute the marginal logpT for the final states, and the correlation between their logit_pf - logit_pb and logpT.
    - energy_fn (str): Energy function to use ("mmff" or "dummy").
    - T (float): Temperature parameter for the energy fn, the higher the smoother the reward.
    - train (bool): Whether the model is in training mode.
    - loss (str): Loss function to use ("vargrad").
    - logrew_clamp (float): Clamp value for log rewards. Anything smaller than this value will be set to logrew_clamp.
    Returns:
    - data_list (list): List of sampled conformers.
    - vargrad_loss (torch.Tensor): Vargrad loss.
    - logit_pf (torch.Tensor): Logits of forward trajectory.
    - logit_pb (torch.Tensor): Logits of backward trajectory.
    - logrews (torch.Tensor): Log rewards.
    '''
    if logit_pf is None or logit_pb is None:
        logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=likelihood, train=train)
    data = traj[-1]    
    try:
        assert all( data[i].canonical_smi == data[0].canonical_smi for i in range(len(data)))
    except:
        raise ValueError( "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs" )
      
    logrews = get_logrew(data, T , energy_fn = energy_fn, clamp=logrew_clamp)
    #LogZ
    if loss == 'vargrad':
        logZ = (logit_pb + logrews/T - logit_pf).detach()
        logZ = reduce(logZ, "bs -> ", "mean")
    elif loss == 'TB':
        logZ = None
        raise NotImplementedError("NN to compute logZ not implemented yet")
    #Loss
    if loss == 'vargrad' or loss == 'TB':
        TB_loss = torch.pow(logZ + logit_pf - logit_pb - logrews/T, 2)
        TB_loss = reduce(TB_loss, "bs -> ", "mean")
    else:
        raise NotImplementedError(f"Loss {loss} not implemented!")

    
    '''
    if train : 
        gradients = torch.autograd.grad(logit_pf.sum(), model.parameters(), retain_graph=True, allow_unused=True)
        print('Num of Nans in gradients', len([g for g in gradients if torch.isnan(g).any() and g is not None]))
        if use_wandb:
            wandb.log({'grads mean': torch.mean([g.mean() for g in gradients if g is not None]).item()})
            wandb.log({'grads max': torch.max([g.max() for g in gradients if g is not None]).item()})
            wandb.log({'grads min': torch.min([g.min() for g in gradients if g is not None]).item()})
    '''
    return traj[-1].to_data_list(), TB_loss, logit_pf, logit_pb, logrews

    

def vargrad_loss_gradacc(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, pdb=None, energy_fn="dummy", T=1.0, logrew_clamp = -1e5):
    logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=likelihood, train=False)
    data = traj[-1]
    try:
        #assert all(x == data.name[0] for x in data.name)
        assert all( data[i].canonical_smi == data[0].canonical_smi for i in range(len(data)))
    except:
        raise ValueError( "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs" )
    # Computing logrews
    logrews = get_logrew(data, T , energy_fn = energy_fn, clamp=logrew_clamp)
    # Create the C matrix
    X = logit_pf - logit_pb.detach() - logrews/T
    C = X.unsqueeze(1) - X.unsqueeze(0)
    # Compute the gradloss 
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)
    eps = 1 / steps
    n_torsion_angles = len( traj[-1][0].total_perturb) 
    grad = None
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data = traj[sigma_idx]
        data_gpu = copy.deepcopy(data).to(device)
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        data_gpu = model(data_gpu)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        score = data_gpu.edge_pred.cpu()       
        mean, std = g**2 * eps * score, g * np.sqrt(eps)
        perturb = traj[sigma_idx +1 ].total_perturb - traj[sigma_idx].total_perturb
        # compute the forward and backward (in gflownet language) transitions logprobs
        # in forward, the new mean is obtained using the score (see above)
        p_trajs_forward = torus.p_differentiable( (perturb.detach() - mean), std)
        logit_pf = torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
        # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
        sigma_b = sigma_schedule[sigma_idx + 1]
        g_b = sigma_b * torch.sqrt( torch.tensor(2 * np.log(sigma_max / sigma_min)))
        std_b = g_b * np.sqrt(eps)
        p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b)
        logit_pb = torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)

        # Get gradient of logit_pf with respect to parameters
        grad_f = torch.autograd.grad((C.sum(axis = 1) * logit_pf).mean(), model.parameters(), retain_graph=True)
        grad_b = torch.autograd.grad((C.sum(axis = 1) * logit_pb).mean(), model.parameters(), retain_graph=True)
        grad = 4*(grad_f - grad_b) if grad is None else grad + 4*(grad_f - grad_b)
        #Remove logit_pf, logit_pb from the computation graph
        logit_pf = logit_pf.detach()
        logit_pb = logit_pb.detach()
        torch.cuda.empty_cache()
    return grad



def get_loss_diffusion(model, gt_data , sigma_min, sigma_max, device, train, use_wandb = False):      
    assert all( gt_data[i].canonical_smi ==  gt_data[0].canonical_smi for i in range(len(gt_data)))
    # choose a subset of k elements from dummy_data 
    gt_data = Batch.from_data_list(gt_data).to(device)
    data = copy.deepcopy(gt_data).to(device)
    # Noise data 
    sigma = np.exp(np.random.uniform(low=np.log(sigma_min), high=np.log(sigma_max)))
    data.node_sigma = (sigma * torch.ones(data.num_nodes)).to(device)
    torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=data.edge_mask.sum())
    data_cpu = copy.deepcopy(data).to('cpu') 
    data_cpu.pos = perturb_batch(data_cpu, torsion_updates) 
    data.pos = data_cpu.pos.to(device)
    data.edge_rotate = torch.tensor(torsion_updates).to(device)
    #predict score and compute loss
    model = model.to(device)
    with torch.no_grad() if not train else contextlib.nullcontext():   
        data = model(data)
    pred = data.edge_pred
    score = torus.score(
        data.edge_rotate.cpu().numpy(),
        data.edge_sigma.cpu().numpy())
    score = torch.tensor(score, device=pred.device)
    score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
    score_norm = torch.tensor(score_norm, device=pred.device)
    loss = ((score - pred) ** 2 / score_norm).mean()
    if use_wandb : 
        wandb.log({'gt score L1-norm': torch.abs(score).mean() })
        wandb.log({'unnormalized diffusion loss': ((score - pred) ** 2 ).mean() })
        wandb.log({'normalized diffusion loss': loss.item() })
    '''
    # check rmsd to ground truth data
    mols0 = [pyg_to_mol(data0.to('cpu')[i].mol, data0.to('cpu')[i], copy=True) for i in range(len(data0))]
    mols = [pyg_to_mol(data.to('cpu')[i].mol, data.to('cpu')[i], copy=True) for i in range(len(data))]
    rmsds = get_rmsds(mols0, mols)
    print('RMSDs:', np.mean(rmsds), 'sigma', sigma)
    '''
    return loss



def gfn_sgd(model, dataset, optimizer, device,  sigma_min, sigma_max, steps, train, T, batch_size, logrew_clamp, energy_fn, train_mode, use_wandb, ReplayBuffer, p_expl, p_replay):
    if train:
        model.train() # set model to training mode
    loss_tot = 0
    conformers = []
    logit_pfs = []
    logit_pbs = []
    logrews = []
    total_perturbs = []
    trajs = []

    
    for i in range(len(dataset)):
        smi  = dataset[i].canonical_smi
        gt_data = dataset[i].to(device)
        optimizer.zero_grad()
        if train_mode == 'on_policy':
            samples = [copy.deepcopy(gt_data) for _ in range(batch_size)]
            samples = perturb_seeds(samples)  # apply uniform noise to torsion angles
            data = Batch.from_data_list(samples)
            traj, logit_pf, logit_pb = sample_forward_trajs(samples, model, train, sigma_min, sigma_max,  steps, device, p_expl, sample_mode = True)
            
            if train and ReplayBuffer is not None:
                if len(ReplayBuffer) > int(batch_size*0.5): 
                    traj_replay, logrew_replay = ReplayBuffer.sample(int(batch_size*p_replay))
            else:
                traj_replay = None
            
            #traj_concat = concat(traj, traj_replay) TODO UNcomment for replay buffer training!
            traj_concat = traj
            confs, loss_smile, logit_pf, logit_pb, logrew = get_loss(traj_concat , logit_pf, logit_pb, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=train, loss='vargrad', logrew_clamp = logrew_clamp)
        
            if ReplayBuffer is not None:
                ReplayBuffer.update(traj, logrew[:len(traj[0])])
                print('ReplayBuffer mean logrew', torch.mean(ReplayBuffer.buffer_logrews).item())
                if use_wandb:
                    wandb.log({'ReplayBuffer mean logrew': torch.mean(ReplayBuffer.buffer_logrews).item()})

        elif train_mode == 'off_policy': 
            num_torsion_angles = len(gt_data.mask_rotate)
            torsion_angles_linspace = torch.linspace(0, 2*np.pi, 20 )
            # Sample randomly a subset of torsion angles in torsion_angles_linspace. Otherwise the batch size is too big and we run out of memory
            thetas = [torsion_angles_linspace for _ in range(num_torsion_angles)]
            if train: 
                thetas = [np.random.choice(torsion_angles_linspace, 2, replace=False) for _ in range(num_torsion_angles)]
            samples = []
            for torsion_update in itertools.product(*thetas):
                data = copy.deepcopy(gt_data)
                torsion_update = np.array(torsion_update)
                new_pos = modify_conformer(data.pos, data.edge_index.T[data.edge_mask], data.mask_rotate, torsion_update, as_numpy=False) 
                data.pos = new_pos
                data.total_perturb = torch.Tensor(torsion_update) % (2 * np.pi)
                samples.append(data)
            traj = sample_backward_trajs(samples, sigma_min, sigma_max,  steps)
            logit_pf, logit_pb = None, None
            confs, loss_smile, logit_pf, logit_pb, logrew = get_loss(traj, logit_pf, logit_pb, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=train, loss='vargrad', logrew_clamp = logrew_clamp)
        
        elif train_mode =='diffusion':
            #TODO: here, gt_data has only one element (because in the dataloader, we have gt_data.pos = gt_data.pos[0]. Add all other ground truth conformers, and see where they land in the energy landscape with bond legnths/angles taken from the 1st conf )
            loss_smile = get_loss_diffusion(model, gt_data, sigma_min, sigma_max, smi, device, train, use_wandb = use_wandb) 
            traj = None
            confs, logit_pf, logit_pb, logrew = gt_data, None, None, None
        
        elif train_mode == 'mle': 
            # sample backward trajectories
            traj = sample_backward_trajs(gt_data, sigma_min, sigma_max,  steps)
            traj_bis = sample_backward_trajs(gt_data, sigma_min, sigma_max,  steps)
            logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=train)
            logit_pf_bis, logit_pb_bis, logp_bis = get_log_p_f_and_log_pb(traj_bis, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=train)
            loss_kl = - logit_pf.mean()
            loss_consistency = torch.pow(logit_pf.detach() - logit_pb - logit_pf_bis + logit_pb_bis, 2).mean()
            print('KL loss', loss_kl, 'consistency loss', loss_consistency)
            #if use_wandb:
                    #wandb.log({'KL loss': loss_kl.item()})
                    #wandb.log({'consistency loss': loss_consistency.item()})
            loss_smile = loss_kl + 0.0 *loss_consistency
            confs = gt_data
            logrew = None
        else:
            raise NotImplementedError(f"Training mode {train_mode} not implemented!")
    
        loss_tot += loss_smile / len(dataset)
        conformers.append(confs)
        logit_pfs.append(logit_pf.detach() if logit_pf is not None else None)
        logit_pbs.append(logit_pb)
        logrews.append(logrew)
        
        if train_mode == 'on_policy' or train_mode == 'off_policy':
            total_perturbs.append(traj[-1].total_perturb)
        trajs.append(traj)
        
    if train:
        loss_tot.backward()  
        optimizer.step()          
    torch.cuda.empty_cache()
    if train:
        if  use_wandb:
            dict = {'on_policy': 'vargrad loss on-policy', 'off_policy': 'vargrad loss off-policy', 'diffusion': 'diffusion_loss', 'mle': 'mle loss'}
            loss_type = dict[train_mode]
            wandb.log({loss_type: loss_tot.item()})
    return loss_tot, conformers, logit_pfs, logit_pbs, logrews, total_perturbs, trajs

        



