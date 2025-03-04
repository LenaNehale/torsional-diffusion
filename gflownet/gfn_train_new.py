from utils.dataset import * 
from utils.torsion import perturb_batch
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np

import numpy as np
import torch
import diffusion.torus as torus 
import contextlib 
from torch_geometric.data import Batch 
import torch_geometric


from scipy.stats import multivariate_normal
import random
import itertools
import wandb
import time
from einops import reduce, rearrange, repeat
from rdkit.Chem import rdMolAlign
from utils.standardization import fast_rmsd

from gflownet.replay_buffer import concat

"""
    Training procedures for conformer generation using GflowNets.
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def sample_forward_trajs(conformers_input, model, train, sigma_min, sigma_max,  steps, device, p_expl, sample_mode, step_start = None , step_end = None):
    '''
    Sample forward trajectories. N.B. all conformers should come from the same molecular graph.
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
    data = copy.deepcopy(Batch.from_data_list(conformers_input)).to(device)
    for item in data:
        if type(data[item[0]]) == torch.Tensor:
            data[item[0]] = data[item[0]].to(device)
    #data.total_perturb = torch.zeros(len(data)* data.mask_rotate[0].shape[0])
    bs, n_torsion_angles = len(data), data.mask_rotate[0].shape[0]
    if step_start == None: 
        step_start = 0
    if step_end == None:
        step_end = steps
    sigma_schedule = 10 ** np.linspace( np.log10(sigma_max), np.log10(sigma_min), steps + 1 )
    sigma_schedule = sigma_schedule[step_start:step_end + 1]
    eps = 1 / steps
    traj = [copy.deepcopy(data)]
    #mols = [[pyg_to_mol(data.mol[i], data[i], copy=True) for i in range(bs)]]
    logit_pf = torch.zeros(bs, len(sigma_schedule)).to(device)
    logit_pb = torch.zeros(bs, len(sigma_schedule)).to(device)
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data = model(data)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=data.edge_pred.shape) 
        score = data.edge_pred.to(device)        
        if random.random() < p_expl:
            noise_scale = 5 # Set to a higher value for more noise (hence more exploration)
        else:
            noise_scale = 1
        perturb = g**2 * eps * score + (noise_scale * g * np.sqrt(eps) * z).to(device)
        # Get PF and PB
        if not sample_mode:
            mean, std = g**2 * eps * score, g * np.sqrt(eps)
            p_trajs_forward = torus.p_differentiable((perturb.detach() - mean), std)
            logit_pf[:, sigma_idx] += torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
            # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
            sigma_b = sigma_schedule[sigma_idx + 1]
            g_b = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            std_b = g_b * np.sqrt(eps)
            p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b)
            logit_pb[:, sigma_idx] += torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)
         
        #print('devices', [data[item[0]].device for item in data if type(data[item[0]]) == torch.Tensor])
        
        new_pos = perturb_batch(data.to('cpu'), perturb.to('cpu')).to(device)
        for item in data:
            if isinstance(data[item[0]], torch.Tensor) :
                data[item[0]] = data[item[0]].detach().to(device)
        
        data.pos = new_pos
        data.total_perturb = (data.total_perturb + perturb.detach() ) % (2 * np.pi)
        del data.edge_pred 
        torch.cuda.empty_cache()
        traj.append(copy.deepcopy(data))  
        
        #mols.append([pyg_to_mol(data.mol[i], data[i], copy=True) for i in range(bs)])
    
    # get rmsds of noised conformers compared to traj[0]
    #rmsds = [get_rmsds(mols[i], mols[0]) for i in range(len(mols))]
    #print('RMSDs:', [np.mean(r) for r in rmsds])
    if not sample_mode:
        logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
        logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
    else:
        logit_pf, logit_pb = None, None  
    return traj, logit_pf, logit_pb


def sample_backward_trajs(conformers_input, sigma_min, sigma_max,  steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), step_start = None , step_end = None):
    '''
    Sample backward trajectories. N.B. all conformers should come from the same molecular graph.
    Args:
    - conformers_input (list): List of PyTorch geometric data objects representing conformers.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - steps (int): Number of timesteps.
    Returns:
    - traj (list): List of PyTorch geometric batch objects representing a batch of conformers at each timestep of the trajectory.
    '''
    data = copy.deepcopy(Batch.from_data_list(conformers_input)).to(device)
    bs, n_torsion_angles = len(data), data.mask_rotate[0].shape[0]
    sigma_schedule = 10 ** np.linspace( np.log10(sigma_max), np.log10(sigma_min), steps + 1 )
    if step_start == None: 
        step_start = 0
    if step_end == None:
        step_end = steps
    sigma_schedule = sigma_schedule[step_start:step_end + 1]
    eps = 1 / steps
    traj = [copy.deepcopy(data)]
    for _, sigma_b in enumerate(reversed(sigma_schedule[1:])):
        g = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=[n_torsion_angles *  len(data.mol)])
        perturb = g * np.sqrt(eps) * z  
        data.pos = perturb_batch(data.to('cpu'), perturb)  # the minus sign is because we are going backwards
        data.total_perturb = (data.total_perturb + perturb.detach() ) % (2 * np.pi)
        traj.append(copy.deepcopy(data))   
        for item in data:
            if type(data[item[0]]) == torch.Tensor:
                data[item[0]] = data[item[0]].to(device)
    #reverse traj
    traj = traj[::-1]
    return traj

def get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=True, step_start = None , step_end = None):
    '''
    Compute the logits of forward and backward trajectories (logits are not normalized). N.B. all conformers should come from the same molecular graph.
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
    if step_start == None:
        step_start = 0
    if step_end == None:
        step_end = steps
    sigma_schedule = sigma_schedule[step_start:step_end + 1]
    eps = 1 / steps
    n_torsion_angles = len( traj[-1][0].total_perturb)
    logit_pf = torch.zeros(bs, len(sigma_schedule)).to(device)
    logit_pb = torch.zeros(bs, len(sigma_schedule)).to(device)   

    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data = traj[sigma_idx].to(device)
        data.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data = model(data)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        score = data.edge_pred      
        mean, std = g**2 * eps * score, (g * np.sqrt(eps)).to(device)
        perturb = traj[sigma_idx +1].total_perturb.to(device) - traj[sigma_idx].total_perturb.to(device)
        # compute the forward and backward (in gflownet language) transitions logprobs
        # in forward, the new mean is obtained using the score (see above)
        p_trajs_forward = torus.p_differentiable( (perturb.detach() - mean), std)
        logit_pf[:, sigma_idx] += torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
        # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
        sigma_b = sigma_schedule[sigma_idx + 1]
        g_b = sigma_b * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        std_b = (g_b * np.sqrt(eps)).to(device)
        p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b)
        logit_pb[:, sigma_idx] += torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)
        del data.edge_pred 
        torch.cuda.empty_cache()
        
    logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
    logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
    
    if likelihood:
        logp = get_logpT(traj[-1].to_data_list(), model, sigma_min, sigma_max,  steps, ode = False) 
        #print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])
    else:
        logp = None
    return logit_pf, logit_pb, logp

def get_logpT(conformers, model, sigma_min, sigma_max,  steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), ode = True, num_trajs = 10):
    '''
    Computes the log-likelihood of conformers using the reverse ODE (data -> noise).N.B. all conformers should come from the same molecular graph.
    Args:
    - conformers: list of pytorch geometric data objects representing conformers
    - model: score model
    - sigma_min, sigma_max: noise variance at timesetps (0,T)
    - steps: number of timesteps
    - device: cuda or cpu
    - ode: whether to use the reverse ODE or not
    - num_trajs: number of backward trajectories to sample
    Returns:
    - logp: log-likelihood of conformers under the model
    '''

    if ode: 
        sigma_schedule = 10 ** np.linspace( np.log10(sigma_max), np.log10(sigma_min), steps + 1)
        eps = 1 / steps
        data = copy.deepcopy(Batch.from_data_list(conformers))
        bs, n_torsion_angles = len(data), data.mask_rotate[0].shape[0]
        logp = torch.zeros(bs)
        #data.total_perturb = torch.zeros(bs * n_torsion_angles)
        data_gpu = copy.deepcopy(data).to(device)
        mols = [[pyg_to_mol(data[i].mol, data[i], copy=True) for i in range(len(data))]] # viz molecules trajs during denoising
        traj = [copy.deepcopy(data)]
        for sigma_idx, sigma in enumerate(reversed(sigma_schedule[1:])):
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)
            ## apply reverse ODE perturbation 
            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            score = data_gpu.edge_pred.cpu()
            perturb =  - 0.5 * g ** 2 * eps * score # minus is because we are going backwards
            new_pos = perturb_batch(data, perturb)
            data.pos = new_pos
            data.total_perturb = (data.total_perturb + perturb ) % (2 * np.pi)
            mols.append([pyg_to_mol(data[i].mol, data[i], copy=True) for i in range(len(data))])
            #data = copy.deepcopy(conf_dataset_likelihood.data) 
            data_gpu.pos =  data.pos.to(device) # has 2 more attributes than data: edge_pred and edge_sigma
            div = divergence(model, data, data_gpu, method='full') 
            logp += -0.5 * g ** 2 * eps * div
            data_gpu.pos = data.pos.to(device)
            traj.append(copy.deepcopy(data))
        # Get rmsds of noised conformers compared to traj[-1]
        #rmsds = [get_rmsds(mols[i], mols[-1]) for i in range(len(mols))]
        #print('RMSDs(mols[t], mols[0]) for t in [0, T]', [np.mean(r) for r in rmsds])

    else:
        conformers = [ x for x in conformers for _ in range(num_trajs)]
        traj = sample_backward_trajs(conformers, sigma_min, sigma_max,  steps)
        logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=False)
        logit_pf, logit_pb = logit_pf.reshape( -1, num_trajs), logit_pb.reshape( -1, num_trajs)      
        logp = torch.logsumexp(logit_pf - logit_pb, dim = -1)
        #Empty Cuda memory
        del logit_pf, logit_pb
        torch.cuda.empty_cache()
    return logp


def get_logrew(data, T , energy_fn = 'mmff', clamp = -1e5):
    if energy_fn == 'mmff':
        if type(data) == torch_geometric.data.data.Data:
            energies = [mmff_energy(pyg_to_mol(data.mol, data))]
        else:
            energies = [mmff_energy(pyg_to_mol(data[i].mol, data[i])) for i in range(len(data))]
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
      
    logrews = get_logrew(data, T , energy_fn = energy_fn, clamp=logrew_clamp).to(device)
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

    

def vargrad_loss_gradacc(traj, logit_pf, logit_pb, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn="mmff", T=1.0, logrew_clamp = -1e5):
    
    if logit_pf is None or logit_pb is None: 
        logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=likelihood, train=False)
    data = traj[-1]
    try:
        #assert all(x == data.name[0] for x in data.name)
        assert all( data[i].canonical_smi == data[0].canonical_smi for i in range(len(data)))
    except:
        raise ValueError( "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs" )
    # Computing logrews
    logrews = get_logrew(data, T , energy_fn = energy_fn, clamp=logrew_clamp).to(device)
    # Create the C vector: C = logZ + logit_pf - logit_pb - logrews/T
    logZ = (logit_pb + logrews/T - logit_pf).detach()
    logZ = reduce(logZ, "bs -> ", "mean")
    C = (logZ + logit_pf - logit_pb - logrews/T).detach()
    # Compute the gradloss 
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)
    eps = 1 / steps
    n_torsion_angles = len( traj[-1][0].total_perturb) 
    grad = None
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data = traj[sigma_idx].to(device)
        data.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        # with torch.grad == True: 
        with torch.set_grad_enabled(True):
            data = model(data)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        score = data.edge_pred
        mean, std = g**2 * eps * score, g * np.sqrt(eps)
        perturb = traj[sigma_idx +1 ].total_perturb - traj[sigma_idx].total_perturb
        # compute the forward and backward (in gflownet language) transitions logprobs
        # in forward, the new mean is obtained using the score (see above)
        p_trajs_forward = torus.p_differentiable( (perturb.detach() - mean), std.to(device))
        logit_pf_local = torch.log(p_trajs_forward).reshape(-1, n_torsion_angles).sum(dim = -1)
        # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
        sigma_b = sigma_schedule[sigma_idx + 1]
        g_b = sigma_b * torch.sqrt( torch.tensor(2 * np.log(sigma_max / sigma_min)))
        std_b = g_b * np.sqrt(eps)
        p_trajs_backward = torus.p_differentiable(perturb.detach(), std_b.to(device))
        logit_pb_local = torch.log(p_trajs_backward).reshape(-1, n_torsion_angles).sum(dim = -1)

        # Get gradient of logit_pf with respect to parameters
        grad_f = torch.autograd.grad((C * logit_pf_local).mean(), model.parameters(), retain_graph=True, allow_unused=True)
        if grad is None:
            grad = list(grad_f)
            for i in range(len(grad)):
                if grad[i] is not None:
                    grad[i] = 2 * grad[i]
        else:
            for i in range(len(grad)):
                if grad[i] is not None and grad_f[i] is not None:
                    grad[i] =  grad[i] + 2 * grad_f[i]
        #Remove logit_pf from the computation graph
        del logit_pf_local, grad_f, score, data.edge_pred, mean, p_trajs_forward
        torch.cuda.empty_cache()
    
    return Batch.to_data_list(data), grad, logit_pf, logit_pb, logrews


def get_loss_diffusion(model, gt_data , sigma_min, sigma_max, device, train, use_wandb = False):      
    '''
    Compute the diffusion loss.
    Args:
    - model (torch.nn.Module): Score model.
    - gt_data (list): List of PyTorch geometric data objects representing conformers.
    - sigma_min (float): Minimum noise variance at timestep 0.
    - sigma_max (float): Maximum noise variance at timestep T.
    - device (torch.device): CUDA or CPU device.
    - train (bool): Whether the model is in training mode.
    - use_wandb (bool): Whether to log to wandb.
    Returns:
    - loss (torch.Tensor): Diffusion loss.
    '''
    assert all( gt_data[i].canonical_smi ==  gt_data[0].canonical_smi for i in range(len(gt_data)))
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
    #if use_wandb : 
        #wandb.log({'gt score L1-norm': torch.abs(score).mean() })
        #wandb.log({'unnormalized diffusion loss': ((score - pred) ** 2 ).mean() })
        #wandb.log({'normalized diffusion loss': loss.item() })
    '''
    # check rmsd to ground truth data
    mols0 = [pyg_to_mol(data0.to('cpu')[i].mol, data0.to('cpu')[i], copy=True) for i in range(len(data0))]
    mols = [pyg_to_mol(data.to('cpu')[i].mol, data.to('cpu')[i], copy=True) for i in range(len(data))]
    rmsds = get_rmsds(mols0, mols)
    print('RMSDs:', np.mean(rmsds), 'sigma', sigma)
    '''
    return loss

def get_rmsds(mols0, mols1):
    '''
    This function compares a set of molecules(e.g. optimized vs non optimized, gt vs generated) and returns the RMSDs between each pair (e.g. non-optimized, optimized).
    '''
    rmsds = []
    for ix in range(len(mols0)):
        mol0, mol1 = mols0[ix], mols1[ix]
        rdMolAlign.AlignMol(mol0, mol1)
        rmsds.append(fast_rmsd(mol0, mol1 , conf1=0, conf2=0))
    return rmsds


def gfn_sgd(model, dataset, optimizer, device,  sigma_min, sigma_max, steps, train, T, batch_size, logrew_clamp, energy_fn, train_mode, use_wandb, ReplayBuffer, p_expl, p_replay, grad_acc = False):
    if train:
        model.train() # set model to training mode
    else:
        try:
            assert grad_acc == False
        except:
            raise ValueError('grad_acc should be False when training is False')
    loss_tot = []
    conformers = []
    logit_pfs = []
    logit_pbs = []
    logrews = []
    total_perturbs = []
    trajs = []

    optimizer.zero_grad()
    

    batch, conditions = sample_batch(dataset, n_smis, n_local_structures) #batch shape: [n_smis, n_local_structures]. Conditions is a dictionary which keys are smiles and values are the local structures ixs in the dataset
    smis = list(conditions.keys())
    
    if train_mode == 'gflownet':

        batch = init_confs(batch, n_confs, randomize_tas = True)  # apply uniform noise to torsion angles. Shape: [n_smis, n_local_structures, n_confs]
        for i, smi in enumerate(smis):
            batch_smi = batch[i]
            batch_smi_flat = rearrange(batch_smi, 'n_local_structures  n_confs -> (n_local_structures  n_confs)') # shape: n_local_structures * n_confs
            traj, logit_pf, logit_pb = sample_forward_trajs(batch_smi_flat , model, train = False if grad_acc else train, sigma_min =  sigma_min, sigma_max = sigma_max,  steps = steps, device = device, p_expl = p_expl, sample_mode = False)
            if train and ReplayBuffer is not None:
                batch_smi_replay, _ = ReplayBuffer.sample(smi, conditions[smi], min(ReplayBuffer.get_len(conditions), int(batch_size*p_replay))) # shape: [n_local_structures, variable_list_size]
                batch_smi_replay_flat = batch_smi_replay.flatten()
                if len(batch_smi_replay_flat.flatten()) > 0:
                    traj_replay = sample_backward_trajs(batch_smi_replay_flat, sigma_min, sigma_max,  steps)
                    logit_pf_replay, logit_pb_replay, _ = get_log_p_f_and_log_pb(traj_replay, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=False if grad_acc else train)
            else:
                traj_replay, logit_pf_replay, logit_pb_replay = None, None, None
            
            traj_concat, logit_pf_concat, logit_pb_concat = concat(traj, traj_replay), torch.cat((logit_pf, logit_pf_replay)), torch.cat((logit_pb, logit_pb_replay))
           
            if grad_acc:
                confs, grad, logit_pf, logit_pb, logrew = vargrad_loss_gradacc(traj_concat, logit_pf_concat, logit_pb_concat, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, logrew_clamp = logrew_clamp)
                logZ = (logit_pb + logrew/T - logit_pf).detach()
                logZ = reduce(logZ, "bs -> ", "mean")
                TB_loss = torch.pow(logZ + logit_pf - logit_pb - logrew/T, 2)
                loss_smile = reduce(TB_loss, "bs -> ", "mean")

                for param, g in zip(model.parameters(), grad):
                    if g is not None:
                        if param.grad is None:
                            param.grad = g / len(dataset)
                        else:
                            param.grad += g / len(dataset)
            else:
                confs, loss_smile, logit_pf, logit_pb, logrew = get_loss(traj_concat , logit_pf_concat, logit_pb_concat, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=train, loss='vargrad', logrew_clamp = logrew_clamp)
        
            if ReplayBuffer is not None:
                ReplayBuffer.update(smi, confs[:batch_size], logrew[:batch_size]) # discard the elements that were already present in the replay buffer
                if use_wandb:
                    N = ReplayBuffer.get_len(smi)
                    rb_logrew_mean_smi =  [ReplayBuffer.content[smi][n][1] for n in range(N)] 
                    rb_logrew_mean_smi = torch.stack(rb_logrew_mean_smi).mean().item()
                    wandb.log({f'RB_logrew_{smi}': rb_logrew_mean_smi }) #TODO: ugly, write this better 
            
    
    elif train_mode =='diffusion':
        loss_smile = get_loss_diffusion(model, gt_data , sigma_min, sigma_max, device, train, use_wandb) 
        traj = None
        confs, logit_pf, logit_pb = gt_data, None, None
        logrew = get_logrew(gt_data, T , energy_fn , clamp = logrew_clamp)
    
    elif train_mode == 'mle': 
        # sample backward trajectories. Samples in gt_data don't need to have the same local structure
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
        logrew = get_logrew(gt_data, T , energy_fn , clamp = logrew_clamp)
    else:
        raise NotImplementedError(f"Training mode {train_mode} not implemented!")
    

    if use_wandb:
        wandb.log({f'logrew mean {smi}': logrew.mean().item()})

    if train_mode == 'mle' or train_mode == 'diffusion':
        # Sample on-policy for computing measures
        samples = perturb_seeds(dataset[i] )  
        traj_on_policy, logit_pf_on_policy, logit_pb_on_policy = sample_forward_trajs(samples, model, train = False, sigma_min =  sigma_min, sigma_max = sigma_max,  steps = steps, device = device, p_expl = 0.0, sample_mode = False)
        confs, _, _, _, logrew = get_loss(traj_on_policy , logit_pf_on_policy, logit_pb_on_policy, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=False, loss='vargrad', logrew_clamp = logrew_clamp)
    
    gt_mols = [pyg_to_mol(conf.mol, conf, copy=True) for conf in dataset[i]]
    gen_mols = [pyg_to_mol(conf.mol, conf, copy=True) for conf in confs]
    rmsds = np.array([get_rmsds([gt_mols[i] for _ in range(len(gen_mols))], gen_mols) for i in range(len(gt_mols))])
    rmsds_precision = np.min(rmsds, axis=0)
    rmsds_recall = np.min(rmsds, axis=1)
    if use_wandb:
        wandb.log({f"RMSD precision {smi}": np.mean(rmsds_precision).item()})
        wandb.log({f"RMSD recall {smi}": np.mean(rmsds_recall).item()})


    loss_tot.append( loss_smile / len(dataset))
    conformers.append(confs)
    logit_pfs.append(logit_pf.detach() if logit_pf is not None else None)
    logit_pbs.append(logit_pb)
    logrews.append(logrew)
    
    if train_mode == 'gflownet' :
        total_perturbs.append(traj[-1].total_perturb)
    trajs.append(traj)
    
    if train:
        if train_mode != 'gflownet' or (  train_mode == 'gflownet' and grad_acc == False): 
            torch.stack(loss_tot).mean().backward()  
        optimizer.step()          
    torch.cuda.empty_cache()
    if train:
        if  use_wandb:
            dict = {'gflownet': 'vargrad loss', 'diffusion': 'diffusion_loss', 'mle': 'mle loss'}
            loss_type = dict[train_mode]
            wandb.log({loss_type: torch.stack(loss_tot).mean().item()})

            wandb.log({f'logrew total': torch.cat(logrews).mean().item()})
    
    

            
    return torch.stack(loss_tot), conformers, logit_pfs, logit_pbs, logrews, total_perturbs, trajs

        



