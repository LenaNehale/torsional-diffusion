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


from rdkit.Chem import rdMolAlign
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


def get_logpT(conformers, model, sigma_min, sigma_max,  steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), ode = True, num_trajs = 10):
    '''
    Computes the log-likelihood of conformers using the reverse ODE (data -> noise)
    Args:
    - conformers: list of pytorch geometric data objects representing conformers
    - model: score model
    - sigma_min, sigma_max: noise variance at timesetps (0,T)
    - steps: number of timesteps
    - device: cuda or cpu
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
        pickle.dump(mols, open("mols_ode_backwards.pkl", "wb"))
        # Get rmsds of noised conformers compared to traj[-1]
        #rmsds = [get_rmsds(mols[i], mols[-1]) for i in range(len(mols))]
        #print('RMSDs(mols[t], mols[0]) for t in [0, T]', [np.mean(r) for r in rmsds])

    else:
        conformers = [ x for x in conformers for _ in range(num_trajs)]
        traj = sample_backward_trajs(conformers, sigma_min, sigma_max,  steps)
        logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=False)
        logit_pf, logit_pb = logit_pf.reshape( -1, num_trajs), logit_pb.reshape( -1, num_trajs)      
        logp = torch.logsumexp(logit_pf - logit_pb, dim = -1)
    return logp, traj
 
    
def get_2dheatmap_array_and_pt(data,model, sigma_min, sigma_max,  steps, device, num_points, ix0, ix1, energy_fn, ode, num_trajs):
    '''
    Get 2Denergy heatmap and logpT heatmap. Both are obtained by computing the energy for different values (linspace) of the 2 torsion angles ix0 and ix1, while fixing the other torsion angles.
    Args:
    - data: pytorch geometric data object representing a conformer
    - model: score model
    - sigma_min, sigma_max: noise variance at timesetps (0,T)
    - steps: number of timesteps
    - device: cuda or cpu
    - num_points: number of points in the linspace
    - ix0, ix1: indices of the torsion angles to vary
    - energy_fn: energy function to use (mmff or dummy)
    Returns:
    - energy_landscape: 2D array of energy values
    - logpTs: 2D array of logpT values
    '''
    torsion_angles_linspace = torch.linspace(0, 2*np.pi, num_points)
    energy_landscape = []
    datas = []
    logpTs = []
    if hasattr(data, 'total_perturb') == False:
        data.total_perturb = torch.zeros( data.mask_rotate.shape[0])
    assert torch.abs(data.total_perturb).max() == 0
    for theta0 in torsion_angles_linspace:
        datas.append([])
        energy_landscape.append([])
        for theta1 in torsion_angles_linspace:
            data0 = copy.deepcopy(data)
            torsion_update = np.zeros(len(data0.mask_rotate))
            torsion_update[ix0], torsion_update[ix1] = theta0, theta1
            new_pos = modify_conformer(data0.pos, data0.edge_index.T[data0.edge_mask], data0.mask_rotate, torsion_update, as_numpy=False) #change the value of the 1st torsion angle
            data0.pos = new_pos
            data0.total_perturb = torch.tensor(torsion_update) % (2 * np.pi)
            #if torch.abs(torch.tensor(torsion_update) % (2 * np.pi) - torch.tensor(torsion_update)).max() > 0 : 
                #breakpoint()
            mol = pyg_to_mol(data0.mol, data0, mmff=False, rmsd=True, copy=True)
            datas[-1].append(copy.deepcopy(data0))
            if energy_fn == "mmff":
                energy_landscape[-1].append(50 * np.log(mmff_energy(mol)).item())
            elif energy_fn == 'dummy':
                energy_landscape[-1].append( - get_dummy_logrew(data0.total_perturb, 1).item())
        logpT, trajs_ode = get_logpT(datas[-1], model.to(device), sigma_min, sigma_max,  steps, device, ode, num_trajs)
        logpT = logpT.tolist()
        logpTs.append(logpT)
    return energy_landscape, logpTs


def sample_forward_trajs(conformers_input, model, train, sigma_min, sigma_max,  steps, device, noise_scale = 1):
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
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data_gpu = model(data_gpu)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
        score = data_gpu.edge_pred.cpu()        
        perturb = g**2 * eps * score + noise_scale * g * np.sqrt(eps) * z
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
    return traj

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
        logp, trajs_ode = get_logpT(traj[-1].to_data_list(), model, sigma_min, sigma_max,  steps, ode = False)
        print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])
    else:
        logp = None
    return logit_pf, logit_pb, logp


def get_dummy_logrew(total_perturb, bs):
    num_torsion_angles = total_perturb.reshape(bs, -1).shape[1]
    mean = torch.ones(num_torsion_angles)
    sigma = 0.05 
    mvn = multivariate_normal(mean = mean, cov = sigma * torch.eye(len(mean)), allow_singular = False, seed = 42)
    probs = mvn.pdf(total_perturb.reshape(bs, -1).cpu().numpy())
    if bs == 1 : 
        probs = [probs]
    logrews = torch.Tensor(np.log(probs))
    return logrews

def get_loss(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn="dummy", T=1.0, train=False, loss='vargrad', logrew_clamp = -1e3):
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
    logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=likelihood, train=train)
    data = traj[-1]
    bs = len(data)
    
    try:
        #assert all(x == data.name[0] for x in data.name)
        assert all( data[i].canonical_smi == data[0].canonical_smi for i in range(len(data)))
    except:
        raise ValueError( "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs" )
    if energy_fn == "mmff":
        logrews = - 50 * torch.log(torch.Tensor([mmff_energy(pyg_to_mol(data.mol[i], data[i])) for i in range(bs)])) / T
    elif energy_fn == 'dummy':        
        total_perturb = traj[-1].total_perturb 
        logrews = get_dummy_logrew(total_perturb, bs)
    else:
        raise  NotImplementedError(f"Energy function {energy_fn} not implemented!")
    # logrews clamping. Mandatory, otherwise we get crazy values for variance
    logrews[logrews<logrew_clamp] = logrew_clamp
    #TB Loss 
    if loss == 'vargrad':
        logZ = (logit_pb + logrews/T - logit_pf).detach()
        logZ = reduce(logZ, "bs -> ", "mean")

    else:
        logZ = None
        raise NotImplementedError(f"Loss {loss} not implemented!")
    TB_loss = torch.pow(logZ + logit_pf - logit_pb - logrews/T, 2)
    TB_loss = reduce(TB_loss, "bs -> ", "mean")
    if train : 
        gradients = torch.autograd.grad(logit_pf.sum(), model.parameters(), retain_graph=True, allow_unused=True)
        #print('Num of Nans in gradients', len([g for g in gradients if torch.isnan(g).any() and g is not None]))
        #if use_wandb:
            #wandb.log({'grads mean': torch.mean([g.mean() for g in gradients if g is not None]).item()})
            #wandb.log({'grads max': torch.max([g.max() for g in gradients if g is not None]).item()})
            #wandb.log({'grads min': torch.min([g.min() for g in gradients if g is not None]).item()})
    return traj[-1].to_data_list(), TB_loss, logit_pf, logit_pb, logrews

    

def vargrad_loss_gradacc(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, pdb=None, energy_fn="dummy", T=1.0, loss='vargrad', logrew_clamp = -1e3):
    logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=likelihood, train=False)
    data = traj[-1]
    bs = len(data)
    try:
        #assert all(x == data.name[0] for x in data.name)
        assert all( data[i].canonical_smi == data[0].canonical_smi for i in range(len(data)))
    except:
        raise ValueError( "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs" )
    # Computing logrews
    if energy_fn == "mmff":
        logrews = - 50 * torch.log(torch.Tensor([mmff_energy(pyg_to_mol(data.mol[i], data[i])) for i in range(bs)])) / T
    elif energy_fn == 'dummy':
        total_perturb = traj[-1].total_perturb 
        logrews = get_dummy_logrew(total_perturb, bs)
    else:
        raise  NotImplementedError(f"WARNING: Energy function {energy_fn} not implemented!")
    # logrews clamping. Mandatory, otherwise we get crazy values for variance
    logrews[logrews<logrew_clamp] = logrew_clamp
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

def get_rmsds(mols0, mols1):
    '''
    This function compares a set of molecules with their optimized versions and returns the RMSDs between each pair (non-optimized, optimized).
    '''
    rmsds = []
    for ix in range(len(mols0)):
        mol0, mol1 = mols0[ix], mols1[ix]
        rdMolAlign.AlignMol(mol0, mol1)
        rmsds.append(fast_rmsd(mol0, mol1 , conf1=0, conf2=0))
    return rmsds

def get_loss_diffusion(model, gt_data_path , sigma_min, sigma_max, smi, device, train, use_wandb = False):
    #Load pickle of dummy ground truth conformers
    if gt_data_path is not None:
        gt_data = pickle.load(open( gt_data_path , 'rb'))
    else:
        raise ValueError("Please provide a path to the ground truth data")
    
    # choose a subset of k elements from dummy_data 
    gt_data_batch = random.sample(gt_data, 16)
    data0 = Batch.from_data_list(gt_data_batch).to(device)
    data = copy.deepcopy(data0).to(device)
    assert data.canonical_smi[0] == smi
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
    # check rmsd to ground truth data
    mols0 = [pyg_to_mol(data0.to('cpu')[i].mol, data0.to('cpu')[i], copy=True) for i in range(len(data0))]
    mols = [pyg_to_mol(data.to('cpu')[i].mol, data.to('cpu')[i], copy=True) for i in range(len(data))]
    rmsds = get_rmsds(mols0, mols)
    print('RMSDs:', np.mean(rmsds), 'sigma', sigma)
    return loss

def get_gt_score(gt_data_path, sigma_min, sigma_max, device, num_points, ix0, ix1, steps = 5):
    gt_data = Batch.from_data_list(pickle.load(open( gt_data_path , 'rb'))).to(device)
    torsion_angles_linspace = torch.linspace(0, 2*np.pi, num_points)
    num_torsion_angles = len(gt_data.mask_rotate[0])
    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)
    scores = torch.zeros(steps, num_points, num_points, 2)
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        #data0.node_sigma = sigma * torch.ones(data0.num_nodes, device=device)
        g = sigma * torch.sqrt( torch.tensor(2 * np.log(sigma_max / sigma_min)))
        eps = 1 / steps
        std = g * np.sqrt(eps)
        for i, theta0 in enumerate(torsion_angles_linspace):
            for j, theta1 in enumerate(torsion_angles_linspace):
                perturb = torch.zeros(num_torsion_angles).to(device)
                perturb[ix0], perturb[ix1] = theta0, theta1  
                for k in range(len(gt_data)):
                    perturbx = perturb - gt_data[k].total_perturb
                    perturbx.requires_grad = True
                    logp = torch.log(torus.p_differentiable(perturbx, std)).sum() 
                    #compute gradlogp with respect to the 1st argument
                    grad = torch.autograd.grad(logp, perturbx, retain_graph=True)[0]
                    scores[sigma_idx, i,j][0]+= grad[ix0].item()
                    scores[sigma_idx, i,j][1]+= grad[ix1].item()
                    del grad
                    torch.cuda.empty_cache()
    return scores
                
class ReplayBufferClass():
    '''
    Replay Buffer that stores trajectories and log rewards. It is sorted such that the trajectories with the highest log rewards are at the beginning of the buffer.
    '''
    def __init__(self, max_size = 1000):
        self.max_size = max_size
        self.datapoints = [] # list of torchgeom.data objects
        self.logrews = torch.Tensor([])
    def __len__(self):
        return len(self.datapoints)
    def update(self, trajs, logrews):
        logrew_before_update =  torch.mean(self.logrews).item()
        #convert each elemnt in trajs to data_list
        trajs_list = [x.to_data_list() for x in trajs]
        trajs_list = list(map(list, zip(*trajs_list)))
        self.datapoints_before = copy.deepcopy(self.datapoints)
        self.logrews_before = copy.deepcopy(self.logrews)
        self.datapoints = self.datapoints + trajs_list 
        self.logrews = torch.cat((self.logrews, logrews ))
        # get indexes of sorted logrews
        sorted_ixs = torch.argsort(self.logrews, descending = True)
        self.datapoints = [self.datapoints[ix] for ix in sorted_ixs]
        self.logrews = self.logrews[sorted_ixs]
        if len(self.datapoints) > self.max_size:
            self.datapoints_ = self.datapoints[:self.max_size]
            self.logrews_ = self.logrews[:self.max_size]
            logrew_after_update =  torch.mean(self.logrews_).item()
            if logrew_after_update < logrew_before_update:
                breakpoint()
            self.datapoints = self.datapoints_
            self.logrews = self.logrews_
        

    def sample(self, n):
        if len(self.datapoints)>=n:
            ixs = np.random.choice(len(self.datapoints), n, replace=False)
            return [self.datapoints[ix]for ix in ixs], self.logrews[ixs]
        else:
            raise ValueError("Not enough elements in the buffer to sample")

def gfn_sgd(model, loader, optimizer, device,  sigma_min, sigma_max, steps, train, T, batch_size, max_batches, smi , logrew_clamp, energy_fn, train_mode, use_wandb, ReplayBuffer, gt_data_path):
    if train:
        model.train() # set model to training mode
    loss_tot = 0
    conformers_noise = []
    conformers = []
    logit_pfs = []
    logit_pbs = []
    logrews = []
    total_perturbs = []
    trajs = []
    max_batches = len(loader) if max_batches is None else max_batches


    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader))):  # Here, loader is used to go through smiles. But in our case, we are going to focus only on one smiles
        assert len(batch) == 1 # one smile at a time
        if batch.canonical_smi[0] == smi:
            break 
    if batch_idx == len(loader):
        raise ValueError(f"Smile {smi} not found in the dataset")
    batch = batch.to(device)
    optimizer.zero_grad()
    loss_batch = 0  # list of per-smile vargrad loss
    if train_mode == 'on_policy':
        data = batch[0] # only consider one smile for now
        samples = [copy.deepcopy(data) for _ in range(batch_size)]
        data = Batch.from_data_list(samples)
        samples = perturb_seeds(samples)  # apply uniform noise to torsion angles
        data = Batch.from_data_list(samples)
        conformers_noise.append(samples)
        noise_scale = 1 # Set to a higher value for more noise (similar to 'epsilon-greedy' for discrete gfns)
        traj = sample_forward_trajs(samples, model, train, sigma_min, sigma_max,  steps, device, noise_scale)
        
        if train and ReplayBuffer is not None:
            if len(ReplayBuffer) > int(batch_size*0.5): 
                traj_replay, logrew_replay = ReplayBuffer.sample(int(batch_size*0.5))
            
        confs, loss_smile, logit_pf, logit_pb, logrew = get_loss(traj , model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=train, loss='vargrad', logrew_clamp = logrew_clamp)
       #num_torsion_angles = len(data.mask_rotate[0])
        #print('total perturb mean', traj[-1].total_perturb.reshape(-1,num_torsion_angles).mean(axis = 0))
        if ReplayBuffer is not None:
            ReplayBuffer.update(traj, logrew)
            if use_wandb:
                wandb.log({'ReplayBuffer mean logrew': torch.mean(ReplayBuffer.logrews).item()})

        total_perturbs.append(traj[-1].total_perturb ) 
    elif train_mode == 'off_policy': 
        data = batch[0]
        num_torsion_angles = len(data.mask_rotate)
        torsion_angles_linspace = torch.linspace(0, 2*np.pi, 20 )
        # Sample randomly a subset of torsion angles in torsion_angles_linspace. Otherwise the batch size is too big and we run out of memory
        thetas = [torsion_angles_linspace for _ in range(num_torsion_angles)]
        if train: 
            thetas = [np.random.choice(torsion_angles_linspace, 2, replace=False) for _ in range(num_torsion_angles)]
        samples = []
        for torsion_update in itertools.product(*thetas):
            data0 = copy.deepcopy(data)
            torsion_update = np.array(torsion_update)
            new_pos = modify_conformer(data0.pos, data0.edge_index.T[data0.edge_mask], data0.mask_rotate, torsion_update, as_numpy=False) 
            data0.pos = new_pos
            data0.total_perturb = torch.Tensor(torsion_update) % (2 * np.pi)
            samples.append(data0)
        traj = sample_backward_trajs(samples, sigma_min, sigma_max,  steps)
        confs, loss_smile, logit_pf, logit_pb, logrew = get_loss(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, energy_fn=energy_fn, T=T, train=train, loss='vargrad', logrew_clamp = logrew_clamp)
        total_perturbs.append(traj[-1].total_perturb  ) 
    
    elif train_mode =='diffusion':
        loss_smile = get_loss_diffusion(model, gt_data_path, sigma_min, sigma_max, smi, device, train, use_wandb = use_wandb)
        traj = None
        confs, logit_pf, logit_pb, logrew = None, None, None, None
    
    elif train_mode == 'mle': 
        dummy_data = pickle.load(open( gt_data_path , 'rb'))
        # choose a subset of k elements from dummy_data 
        dummy_data_batch = random.sample(dummy_data, 16)
        data0 = Batch.from_data_list(dummy_data_batch).to(device)
        data = copy.deepcopy(data0).to(device)
        assert data.canonical_smi[0] == smi 
        # sample backward trajectories
        traj = sample_backward_trajs(dummy_data_batch, sigma_min, sigma_max,  steps)
        traj_bis = sample_backward_trajs(dummy_data_batch, sigma_min, sigma_max,  steps)
        logit_pf, logit_pb, logp = get_log_p_f_and_log_pb(traj, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=train)
        logit_pf_bis, logit_pb_bis, logp_bis = get_log_p_f_and_log_pb(traj_bis, model, device, sigma_min, sigma_max,  steps, likelihood=False, train=train)
        loss_kl = - logit_pf.mean()
        loss_consistency = torch.pow(logit_pf.detach() - logit_pb - logit_pf_bis + logit_pb_bis, 2).mean()
        print('KL loss', loss_kl, 'consistency loss', loss_consistency)
        if use_wandb:
                wandb.log({'KL loss': loss_kl.item()})
                wandb.log({'consistency loss': loss_consistency.item()})
        loss_smile = loss_kl + 0.1 *loss_consistency
        confs = dummy_data
        logrew = None



    else:
        raise NotImplementedError(f"Training mode {train_mode} not implemented!")
    
    conformers.append(confs)
    logit_pfs.append(logit_pf.detach() if logit_pf is not None else None)
    logit_pbs.append(logit_pb)
    logrews.append(logrew)
    trajs.append(traj)
    print('loss smile' , loss_smile)
    loss_smile = loss_smile / len(batch)
    if train:
        loss_smile.backward()  
        optimizer.step()          
    loss_batch += loss_smile.item()
    torch.cuda.empty_cache()
    if train:
        if  use_wandb:
            dict = {'on_policy': 'vargrad loss on-policy', 'off_policy': 'vargrad loss off-policy', 'diffusion': 'diffusion_loss', 'mle': 'mle loss'}
            loss_type = dict[train_mode]
            wandb.log({loss_type: loss_smile})
    loss_tot += loss_batch
    del loss_batch
    torch.cuda.empty_cache()
    if smi is None:
        loss_tot = loss_tot / len(loader)

    return loss_tot, conformers_noise, conformers, logit_pfs, logit_pbs, logrews, total_perturbs, trajs


def log_gfn_metrics(model, train_loader, optimizer, device, sigma_min, sigma_max, steps, batch_size, T, max_batches , smi , num_points , logrew_clamp, energy_fn,  num_trajs, use_wandb, ReplayBuffer, train_mode, gt_data_path, seed):
    assert smi is not None
    #Sanity check : the backward pass samples torsion angles in the uniform distribution
    if gt_data_path is not None:
        dummy_data = pickle.load(open(gt_data_path, 'rb'))
        dummy_data_batch = random.sample(dummy_data, 16)
        back_trajs = sample_backward_trajs(dummy_data_batch, sigma_min, sigma_max,  steps)
        s0 = back_trajs[0].total_perturb.reshape(-1, len(dummy_data_batch[0].mask_rotate))
        plt.figure()
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 2*np.pi)
        plt.scatter(s0[:,0], s0[:,1], c='r', s=5)
        plt.xlabel('Theta0')
        plt.ylabel('Theta1')
        plt.title('s0')
        plt.savefig(f"s0_{energy_fn}_{train_mode}_{seed}.png")
        plt.close()
        if use_wandb:
            wandb.log({"s0":  wandb.Image(f"s0_{energy_fn}_{train_mode}_{seed}.png")})
    
        #Diffusion loss
        train_loss_diffusion = get_loss_diffusion(model, gt_data_path, sigma_min, sigma_max, smi, device, train = False, use_wandb = use_wandb)
        
    #vargrad loss
    train_loss, conformers_train_noise, conformers_train_gen, logit_pfs, logit_pbs, logrews, perturbs, trajs = gfn_sgd(model, train_loader, optimizer, device,  sigma_min, sigma_max, steps, train=False, max_batches=max_batches, batch_size = batch_size, T=T, smi = smi, logrew_clamp = logrew_clamp, energy_fn=energy_fn, train_mode='on_policy', use_wandb = use_wandb, ReplayBuffer = ReplayBuffer, gt_data_path = gt_data_path)
    if use_wandb:
        wandb.log({"vargrad loss": train_loss})
    # Plot evolution of total_perturb in forward pass
    trajs = trajs[0]
    traj_perturbs = torch.stack([x.total_perturb for x in trajs])
    # Plot a figure with a line evolution of the total perturbations. Dimensions of traj_perturbs are (timestep, traj_id)
    plt.figure()
    plt.plot(traj_perturbs.cpu().detach().numpy()[:, :64])
    plt.xlabel('Timestep')
    plt.ylabel('Total Perturb')
    plt.title('Total Perturbations in Forward Pass')
    plt.savefig(f"total_perturb_forward_{energy_fn}_{train_mode}_{seed}.png")
    plt.close()
    if use_wandb:
        wandb.log({"total_perturb_forward": wandb.Image(f"total_perturb_forward_{energy_fn}_{train_mode}_{seed}.png")})

    if gt_data_path is not None:
        #RMSD between generated conformers and ground truth conformers
        gt_mols = [pyg_to_mol(dummy_data_batch[i].mol, dummy_data_batch[i], copy=True) for i in range(len(dummy_data_batch))]
        conformers_train_gen = conformers_train_gen[0]
        gen_mols = [pyg_to_mol(conformers_train_gen[i].mol, conformers_train_gen[i], copy=True) for i in range(len(conformers_train_gen))]
        # dump the generated conformers and the ground truth conformers
        pickle.dump(conformers_train_gen, open(f'conformers_train_gen_{train_mode}.pkl', 'wb'))
        pickle.dump(dummy_data_batch, open(f'dummy_data_batch_{train_mode}.pkl', 'wb'))
        pickle.dump(trajs, open(f'traj_{train_mode}.pkl', 'wb'))
        rmsds = np.array([get_rmsds([gt_mols[i] for _ in range(len(gen_mols))], gen_mols) for i in range(len(gt_mols))])
        rmsds = np.min(rmsds, axis=0)
        if use_wandb:
            wandb.log({"RMSDs gen/ground truth": np.mean(rmsds).item()})
    

    # heatmap of energy/learned logpts
    
    for batch_ix, batch in enumerate(tqdm(train_loader, total=len(train_loader))):  # Here, loader is used to go through smiles. But in our case, we are going to focus only on one smiles
        assert len(batch) == 1
        if batch.canonical_smi[0] == smi:
            break 
    data = batch[0]
    num_torsion_angles = len(data.mask_rotate)
    perturbs = perturbs[0].reshape(-1, num_torsion_angles)
    for ix0, ix1 in itertools.combinations(range(num_torsion_angles), 2):
        _,  logpTs_ode = get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max,  steps, device, num_points=num_points, ix0=ix0, ix1=ix1, energy_fn = energy_fn, ode = True, num_trajs = None)
        energy_landscape, logpTs = get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max,  steps, device, num_points=num_points, ix0=ix0, ix1=ix1, energy_fn = energy_fn, ode = False, num_trajs = num_trajs )
        energy_landscape, logpTs, logpTs_ode = np.array(energy_landscape), np.array(logpTs), np.array(logpTs_ode)
        plt.figure()
        plt.imshow(energy_landscape, extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r', vmin=np.min(energy_landscape), vmax=np.max(energy_landscape))
        plt.colorbar(label='Energy')
        plt.xlabel(f'Theta{ix0}')
        plt.ylabel(f'Theta{ix1}')
        plt.title("Energy Landscape")
        plt.savefig(f"energy_landscape_{energy_fn}_{train_mode}_{seed}.png")
        plt.close()
        
        plt.figure()
        plt.imshow(logpTs , extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
        plt.colorbar(label='logpTs')
        plt.xlabel(f'Theta{ix0}')
        plt.ylabel(f'Theta{ix1}')
        plt.title("logpTs")
        plt.savefig(f"logpTs_{energy_fn}_{train_mode}_{seed}.png")
        plt.close()

        plt.figure()
        plt.imshow(logpTs_ode , extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
        plt.colorbar(label='logpTs_ode')
        plt.xlabel(f'Theta{ix0}')
        plt.ylabel(f'Theta{ix1}')
        plt.title("logpTs_ode")
        plt.savefig(f"logpTs_ode_{energy_fn}_{train_mode}_{seed}.png")
        plt.close()

        #plot samples from gflownet/ground truth/replay buffer
        plt.figure()
        theta0, theta1 = perturbs[:,ix0]%(2*np.pi) , perturbs[:,ix1]%(2*np.pi)
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 2*np.pi)
        plt.scatter(theta0, theta1, c='r', s=30, alpha = 0.25, marker = 'o')
        if gt_data_path is not None:
            # plot the ground truth torsion angles from gt_data_path
            dummy_data = pickle.load(open(gt_data_path, 'rb'))
            gt_total_perturb = Batch.from_data_list(dummy_data).total_perturb.reshape(-1, num_torsion_angles)
            gt_theta0, gt_theta1 = gt_total_perturb[:,ix0]%(2*np.pi) , gt_total_perturb[:,ix1]%(2*np.pi)
            plt.scatter(gt_theta0, gt_theta1, c='b', s=30, alpha=0.25, marker = '^')
        if ReplayBuffer is not None and len(ReplayBuffer) > 0:
            #Plot traj[-1] in the replay buffer
            traj_replay, _ = ReplayBuffer.sample(min(ReplayBuffer.max_size, len(ReplayBuffer)))
            perturbs_replay = torch.stack([x[-1].total_perturb for x in traj_replay])
            theta0replay, theta1replay = perturbs_replay[:,ix0]%(2*np.pi) , perturbs_replay[:,ix1]%(2*np.pi)
            plt.scatter(theta0replay, theta1replay, c='g', s=30, alpha=0.25, marker = 'x')
        plt.xlabel(f'Theta{ix0}')
        plt.ylabel(f'Theta{ix1}')
        plt.title('Samples')
        plt.savefig(f"samples_{energy_fn}_{train_mode}_{seed}.png")
        plt.close()
        

        if use_wandb:
            wandb.log({f"energy_landscape_{ix0,ix1}":  wandb.Image(f"energy_landscape_{energy_fn}_{train_mode}_{seed}.png")})
            wandb.log({f"logpts_{ix0, ix1}": wandb.Image(f"logpTs_{energy_fn}_{train_mode}_{seed}.png")})
            wandb.log({f"logpts_ode_{ix0, ix1}": wandb.Image(f"logpTs_ode_{energy_fn}_{train_mode}_{seed}.png")})
            wandb.log({f"samples_{ix0, ix1}": wandb.Image(f"samples_{energy_fn}_{train_mode}_{seed}.png")})

    #Plot the evolution of the lower bound over logZ
    logZ_hat = (torch.stack(logit_pbs) + torch.stack(logrews) - torch.stack(logit_pfs)) 
    logZ_hat = logZ_hat.mean(dim=1)
    if use_wandb:
        wandb.log({"logZ_hat": logZ_hat[0]})
    return conformers_train_gen
    

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
