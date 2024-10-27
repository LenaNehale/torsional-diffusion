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


import pickle
import wandb
import matplotlib.pyplot as plt
wandb.login()
run = wandb.init(project="gfn_torsional_diff")


"""
    Training procedures for conformer generation using GflowNets.
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def get_logpT(conformers, model, sigma_min, sigma_max,  steps, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    #computes likelihood of terminal samples using the reverse ODE
    batch_size = len(conformers)
    sigma_schedule = 10 ** np.linspace(
        np.log10(sigma_max), np.log10(sigma_min), steps + 1
    )
    eps = 1 / steps
    logp = torch.zeros(batch_size)
    conf_dataset_likelihood = InferenceDataset(copy.deepcopy(conformers))
    loader = DataLoader(conf_dataset_likelihood, batch_size=batch_size, shuffle=False)
    for _, data in enumerate(loader):
        break #there is only ont batch in this loader
    data_gpu = copy.deepcopy(data).to(device)
    for sigma_idx, sigma in enumerate(reversed(sigma_schedule[1:])):
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad():
            data_gpu = model(data_gpu)
        ## apply reverse ODE perturbation
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        score = data_gpu.edge_pred.cpu()
        perturb =  - 0.5 * g ** 2 * eps * score
        conf_dataset_likelihood.apply_torsion_and_update_pos(data, perturb.numpy()) 
        div = divergence(model, data, data_gpu, method='hutch') 
        logp += -0.5 * g ** 2 * eps * div
        data_gpu.pos = data.pos.to(device)
    return logp


def get_2dheatmap_array_and_pt(data,model, sigma_min, sigma_max,  steps, device, num_points=10, ix0=0, ix1=1 ):
    '''
    Get 2Denergy heatmap. This is obtained by computing the energy for different value (linspace) of the 2 torsion angles ix0 and ix1. 
    '''
    torsion_angles_linspace = torch.linspace(0, 2*np.pi, num_points)
    energy_landscape = []
    datas = []
    logpTs = []
    for theta0 in torsion_angles_linspace:
        datas.append([])
        energy_landscape.append([])
        for theta1 in torsion_angles_linspace:
            data0 = copy.deepcopy(data)
            torsion_update = np.zeros(len(data0.mask_rotate))
            torsion_update[ix0], torsion_update[ix1] = theta0, theta1
            new_pos = modify_conformer(data0.pos, data0.edge_index.T[data0.edge_mask], data0.mask_rotate, torsion_update, as_numpy=False) #change the value of the 1st torsion angle
            data0.pos = new_pos
            mol = pyg_to_mol(data0.mol, data0, mmff=False, rmsd=True, copy=True)
            datas[-1].append(copy.deepcopy(data0))
            energy_landscape[-1].append(mmff_energy(mol))
        logpT = get_logpT(datas[-1], model.to(device), sigma_min, sigma_max,  steps, device).tolist()
        logpTs.append(logpT)
    return energy_landscape, logpTs


def sample_and_get_loss(conformers_input,model,device,
    sigma_min, sigma_max,  steps,
    ode=False,
    likelihood=False,
    pdb=None,
    energy_fn="mmff",
    T=1.0,
    add_gradE=False,
    w_grad=0.1,
    train=True,
):
    batch_size = len(conformers_input)
    conformers = copy.deepcopy(conformers_input) # conformers_input are the noised conformers, conformers will be the generated ones
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size, shuffle=False)
    sigma_schedule = 10 ** np.linspace(
        np.log10(sigma_max), np.log10(sigma_min), steps + 1
    )
    eps = 1 / steps

    for batch_idx, data in enumerate(loader):
        bs = data.num_graphs
        print('smi:', [conformers[0].canonical_smi])
        print('energies before sampling', [mmff_energy(pyg_to_mol(data.mol[i], conformers[i])) for i in range(bs) ])  
        n_torsion_angles = len(data.total_perturb[0])
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))
        dlogp = torch.zeros(bs)
        data_gpu = copy.deepcopy(data).to(device)
        break #there is only one batch in this loader
    
    for sigma_idx, sigma in enumerate(sigma_schedule[:-1]):
        data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
        with torch.no_grad() if not train else contextlib.nullcontext():
            data_gpu = model(data_gpu)
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
        score = data_gpu.edge_pred.cpu()
        t = sigma_idx / steps  # t is really 1-t
        if ode:
            perturb = 0.5 * g**2 * eps * score
            if likelihood:
                div = divergence(
                    model, data, data_gpu, method=likelihood
                ) 
                dlogp += -0.5 * g**2 * eps * div
        else:
            perturb = g**2 * eps * score + g * np.sqrt(eps) * z

        if add_gradE:  # TODO this code still doesn't work
            x_t = torch.Tensor(data.total_perturb).flatten()
            x0_hat = x_t + sigma**2 * score
            # apply the perturbation in cartesian coords
            # yet another copy ! Youuhou!
            data_hat = copy.deepcopy(data)
            # data_hat.pos = modify_conformer(data_hat.pos, data_hat.edge_index.T[data.edge_mask], data_hat.mask_rotate, x0_hat)
            conf_dataset_hat = InferenceDataset(copy.deepcopy(conformers))
            conf_dataset_hat.apply_torsion_and_update_pos(
                data_hat, x0_hat.detach().numpy()
            )
            # Compute grad of MMFF energy of conformers at data_hat
            grads = []
            for pos in conf_dataset_hat.data.pos:
                mol = pyg_to_mol(mol, data_hat, copy=False)
                energy, grad = mmff_energy(mol, get_grad=True)
                grads.append(grad)
            grads = torch.stack(grads)
            perturb = perturb + w_grad * grads

        mean, std = g**2 * eps * score, g * np.sqrt(eps)

        # compute the forward and backward (in gflownet language) transitions logprobs
        for i in range(bs):
            start, end = i * n_torsion_angles, (i + 1) * n_torsion_angles
            # in forward, the new mean is obtained using the score (see above)
            p_trajs_forward = torus.p_differentiable(
                (perturb - mean)[start:end], std
            )
            logit_pf[i, sigma_idx] += torch.log(p_trajs_forward).sum()
            # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0. For std, we need to use the next sigma (see https://www.notion.so/logpf-logpb-of-the-ODE-traj-in-diffusion-models-9e63620c419e4516a382d66ba2077e6e)
            sigma_b = sigma_schedule[sigma_idx + 1]
            g_b = sigma_b * torch.sqrt(
                torch.tensor(2 * np.log(sigma_max / sigma_min))
            )
            std_b = g_b * np.sqrt(eps)
            p_trajs_backward = torus.p_differentiable(perturb[start:end], std_b)
            logit_pb[i, sigma_idx] += torch.log(p_trajs_backward).sum()

        conf_dataset.apply_torsion_and_update_pos(data, perturb.detach().numpy())
        data_gpu.pos = data.pos.to(device)

        if pdb:
            for conf_idx in range(bs):
                coords = data.pos[data.ptr[conf_idx] : data.ptr[conf_idx + 1]]
                num_frames = still_frames if sigma_idx == steps - 1 else 1
                pdb.add(
                    coords,
                    part=batch_size * batch_idx + conf_idx,
                    order=sigma_idx + 2,
                    repeat=num_frames,
                )

        for i, d in enumerate(dlogp):
            conformers[data.idx[i]].dlogp = d.item()
        
        #Grad accumulation trick

    logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
    logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
    if likelihood:
        logp = get_logpT(conformers, model, sigma_min, sigma_max,  steps)
        print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])

    # Get VarGrad Loss
    try:
        #assert all(x == data.name[0] for x in data.name)
        assert all(x == data.canonical_smi[0] for x in data.canonical_smi)
    except:
        raise ValueError(
            "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs"
        )
    # Computing logrews
    pos = rearrange(data.pos, "(bs n) d -> bs n d", bs=bs)
    z = rearrange(data.z, "(bs n) -> bs n", bs=bs)
    if energy_fn == "mmff":
        logrews = (-torch.Tensor([mmff_energy(pyg_to_mol(data.mol[i], conformers[i])) for i in range(bs)])/ T)
        print('energies after sampling', [mmff_energy(pyg_to_mol(data.mol[i], conformers[i])) for i in range(bs) ])  
    else:
        raise  NotImplementedError(f"WARNING: Energy function {energy_fn} not implemented!")
    vargrad_quotients = logit_pf - logit_pb - logrews/T
    vargrad_loss = torch.var(vargrad_quotients)   
    return conformers, vargrad_loss


# add replay buffer
# replaybuffer = ReplayBuffer(max_size = 10000)


def gfn_epoch(model, loader, optimizer, device,  sigma_min, sigma_max, steps, train, T, n_trajs = 8, max_batches = None, smi = None):
    if train:
        model.train()
    loss_tot = 0
    conformers_noise = []
    conformers = []
    max_batches = len(loader) if max_batches is None else max_batches


    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader))):  # Here, loader is used to go through smiles. But in our case, we are going to focus only on one smiles
        assert len(batch) == 1
        if batch.canonical_smi[0] == smi:
            break 
        #if batch_idx >= max_batches:
            #break
    if batch_idx == len(loader):
        raise ValueError(f"Smile {smi} not found in the dataset")
    if True:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss_batch = 0  # list of per-smile vargrad loss
        for i in range(len(batch)):
            data = batch[i]
            samples = [copy.deepcopy(data) for _ in range(n_trajs)]
            print('smi:', [samples[0].canonical_smi])
            conf_dataset = InferenceDataset(samples)
            loader = DataLoader(conf_dataset, batch_size = len(samples), shuffle=False)
            data = next(iter(loader))
            print('energies before noising', [mmff_energy(pyg_to_mol(data.mol[i], samples[i].to('cpu'))) for i in range(len(samples)) ]) 
            samples = perturb_seeds(samples)  # apply uniform noise to torsion angles
            conf_dataset = InferenceDataset(samples)
            loader = DataLoader(conf_dataset, batch_size =  len(samples), shuffle=False)
            data = next(iter(loader))
            print('energies after noising', [mmff_energy(pyg_to_mol(data.mol[i], samples[i].to('cpu'))) for i in range(len(samples)) ])
            conformers_noise.append(samples)
            confs, loss_smile = sample_and_get_loss(
                samples, model, device, sigma_min=sigma_min, sigma_max=sigma_max,  steps=steps,train=train, T=T)  # on-policy
            conformers.append(confs)
            loss_smile = loss_smile / len(batch)
            if train:
                loss_smile.backward()
            loss_batch += loss_smile.item()
            #print(f'Available Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.2f} MB')
            #print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            #print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
        if train:
            wandb.log({"loss": loss_batch})
            optimizer.step()
        loss_tot += loss_batch
        del loss_batch
        torch.cuda.empty_cache()

    loss_tot = loss_tot / len(loader)
    return loss_tot, conformers_noise, conformers


def log_gfn_metrics(model, train_loader, val_loader, optimizer, device, sigma_min, sigma_max, steps, n_trajs, T, max_batches = None, smi = None, num_points = 100 ):
    '''
    # Get ground truth val conformers
    data_dir='/home/mila/l/lena-nehale.ezzine/scratch/torsional-diffusion/DRUGS/drugs/'
    split_path='/home/mila/l/lena-nehale.ezzine/scratch/torsional-diffusion/DRUGS/split_boltz_10k.npy'
    mode = 'val'
    dataset='drugs'
    transform = None 
    types = drugs_types
    num_workers = 1
    limit_molecules = 0
    cache = '/home/mila/l/lena-nehale.ezzine/scratch/torsional-diffusion/cache/test_run'
    pickle_dir = '/home/mila/l/lena-nehale.ezzine/scratch/torsional-diffusion/DRUGS/standardized_pickles'
    boltzmann_resampler = None
    val_dataset = ConformerDataset(data_dir, split_path, mode, dataset,
                                   types, transform,
                                   num_workers,
                                   limit_molecules,
                                   cache,
                                   pickle_dir,
                                   boltzmann_resampler)
    '''
    # Log vargrad loss on the replay buffer/training set/val set
    train_loss, conformers_train_noise, conformers_train_gen = gfn_epoch(
        model, train_loader, optimizer, device,  sigma_min, sigma_max, steps, train=False, max_batches=max_batches, n_trajs = n_trajs, T=T, smi = smi
    )
    wandb.log({"train_loss": train_loss})
    conformers_val_gen = []
    '''
    val_loss, conformers_val_noise, conformers_val_gen = gfn_epoch(
        model, val_loader, optimizer, device,  sigma_min, sigma_max, steps, train=False, max_batches=max_batches, n_trajs = n_trajs, T=T, smi = smi
    )
    wandb.log({"val_loss": val_loss})
    '''
    # Log energies/logpT metrics
    energies_train_on_policy = get_energies(conformers_train_gen)
    energies_train_rand = get_energies(conformers_train_noise)
    #energies_val_on_policy = get_energies(conformers_val_gen)
    #energies_val_rand = get_energies(conformers_val_noise)
    
    logpT_train_on_policy = [get_logpT(confs, model, sigma_min, sigma_max,  steps) for confs in conformers_train_gen]
    logpT_train_rand = [get_logpT( confs , model, sigma_min, sigma_max,  steps) for confs in conformers_train_noise]
    #logpT_val_on_policy = [get_logpT(confs, model, sigma_min, sigma_max,  steps) for confs in conformers_val_gen]
    #logpT_val_rand = [get_logpT( confs , model, sigma_min, sigma_max,  steps) for confs in conformers_val_noise]
    # logpT_data = get_logpT(  val_dataset , model, sigma_min, sigma_max,  steps)
    # save in pickle files energies and logpT


    # Wandb log 2 histograms of energies on the same plot
    wandb.log({
    "energies_train_rand": wandb.Histogram(list(energies_train_rand.values())[0], num_bins = 10),
    "energies_train_on_policy": wandb.Histogram(list(energies_train_on_policy.values())[0], num_bins = 10)
})
    '''
    # Wandb log 2 histograms of logpTs on the same plot
    wandb.log({
    "logpTs_train_rand": wandb.plot.histogram(
        value = logpT_train_rand[0].numpy(),
        title="logpTs_train_rand"
    ),
    "logpTs_train_on_policy": wandb.plot.histogram(
        value = logpT_train_on_policy[0].numpy(),
        title="logpTs_train_on_policy"
    )
})
    
    '''
    


    # Plot the heatmap of learned logpts on wandb
    if smi is None:
        smi =next(iter(train_loader)).canonical_smi[0]
    
    for batch_ix, batch in enumerate(tqdm(train_loader, total=len(train_loader))):  # Here, loader is used to go through smiles. But in our case, we are going to focus only on one smiles
        assert len(batch) == 1
        if batch.canonical_smi[0] == smi:
            break 
    data = batch[0]
    
    energy_landscape, logpTs = get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max,  steps, device, num_points=num_points, ix0=0, ix1=1)
    energy_landscape, logpTs = np.array(energy_landscape), np.array(logpTs)
    #image0, image1 =  wandb.Image(np.array(energy_landscape), caption="energy_landscape"), wandb.Image(np.array(logpTs), caption="logpts")
    plt.figure()
    plt.imshow(energy_landscape, extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
    plt.colorbar(label='Energy')
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')
    plt.title("Energy Landscape")
    plt.savefig("energy_landscape.png")
    plt.close()
    
    plt.figure()
    plt.imshow(logpTs , extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
    plt.colorbar(label='logpTs')
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')
    plt.title("logpTs")
    plt.savefig("logpTs.png")
    plt.close()
    
    wandb.log({"energy_landscape":  wandb.Image(np.array(energy_landscape))})
    wandb.log({"logpts": wandb.Image("logpTs.png")})
    return conformers_train_gen, conformers_val_gen
    

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def get_energies(conformers):
    energies = {}
    for confs in conformers:
        for conf in confs:
            mol = pyg_to_mol(conf.mol, conf, rmsd=False)
            energy = mmff_energy(mol)
            if conf.canonical_smi not in energies.keys():
                energies[conf.canonical_smi] = [energy]
            else:
                energies[conf.canonical_smi].append(energy)
    return energies