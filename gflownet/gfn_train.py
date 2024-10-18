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
wandb.login()
run = wandb.init(project="gfn_torsional_diff")


"""
    Training procedures for conformer generation using GflowNets.
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def get_logpT(conformers, model, sigma_min, sigma_max,  steps):
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
        z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
        score = data_gpu.edge_pred.cpu()
        perturb =  - 0.5 * g ** 2 * eps * score
        conf_dataset_likelihood.apply_torsion_and_update_pos(data, perturb.numpy()) 
        div = divergence(model, data, data_gpu, method='hutch') # warning: this function changes data_gpu.pos
        logp += -0.5 * g ** 2 * eps * div
        data_gpu.pos = data.pos.to(device)

    return logp
    


def sample_and_get_loss(
    conformers_input,
    model,
    device,
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
        n_torsion_angles = len(data.total_perturb[0])
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))
        dlogp = torch.zeros(bs)
        data_gpu = copy.deepcopy(data).to(device)
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
                    )  # warning: this function changes data_gpu.pos
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

        logit_pf = reduce(logit_pf, "bs steps-> bs", "sum")
        logit_pb = reduce(logit_pb, "bs steps-> bs", "sum")
        if likelihood:
            logp = get_logpT(conformers, model, sigma_min, sigma_max,  steps)
            print('Correlation(logit_pf - logit_pb,logp )', torch.corrcoef(torch.stack([logit_pf - logit_pb, logp]))[0,1])

    # Get VarGrad Loss
    try:
        # assert all(x == data.name[0] for x in data.name)
        assert all(x == data.canonical_smi[0] for x in data.canonical_smi)
    except:
        raise ValueError(
            "Vargrad loss should be computed for the same molecule only ! Otherwise we have different logZs"
        )
    # Computing logrews
    pos = rearrange(data.pos, "(bs n) d -> bs n d", bs=bs)
    z = rearrange(data.z, "(bs n) -> bs n", bs=bs)
    if energy_fn == "mmff":
        logrews = (
            -torch.Tensor(
                [
                    mmff_energy(
                        pyg_to_mol(data.mol[i], conformers[i], mmff=True, rmsd=False)
                    )
                    for i in range(bs)
                ]
            )
            / T
        )  
    else:
        raise ValueError("Energy function not implemented")
    vargrad_quotients = logit_pf - logit_pb - logrews/T
    vargrad_loss = torch.var(vargrad_quotients)
    return conformers, vargrad_loss


# add replay buffer
# replaybuffer = ReplayBuffer(max_size = 10000)


def gfn_epoch(model, loader, optimizer, device,  sigma_min, sigma_max, steps, train, T, n_trajs = 8, max_batches = None):
    if train:
        model.train()
    loss_tot = 0
    conformers_noise = []
    conformers = []
    max_batches = len(loader) if max_batches is None else max_batches

    for batch_idx, batch in enumerate(
        tqdm(loader, total=len(loader))
    ):  # Here, loader is used to go through smiles
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        optimizer.zero_grad()
        loss_batch = 0  # list of per-smile vargrad loss
        for i in range(len(batch)):
            data = batch[i]
            samples = [copy.deepcopy(data) for _ in range(n_trajs)]
            samples = perturb_seeds(samples)  # apply uniform noise to torsion angles
            conformers_noise.append(samples)
            confs, loss_smile = sample_and_get_loss(
                samples, model, device, sigma_min=sigma_min, sigma_max=sigma_max,  steps=steps,train=train, T=T)  # on-policy
            conformers.append(confs)
            loss_smile = loss_smile / len(batch)
            if train:
                loss_smile.backward()
            loss_batch += loss_smile.item()
            print(f'Available Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.2f} MB')
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
            torch.cuda.empty_cache()
        if train:
            print('loss_batch', loss_batch)
            wandb.log({"loss": loss_batch})
            optimizer.step()
        loss_tot += loss_batch
        del loss_batch
        torch.cuda.empty_cache()

    loss_tot = loss_tot / len(loader)
    return loss_tot, conformers_noise, conformers


def log_gfn_metrics(model, train_loader, val_loader, optimizer, device, sigma_min, sigma_max, steps, n_trajs, T, max_batches = None ):
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
    # Log vargrad loss on the replay buffer/training set/val set
    train_loss, conformers_train_noise, conformers_train_gen = gfn_epoch(
        model, train_loader, optimizer, device,  sigma_min, sigma_max, steps, train=False, max_batches=max_batches, n_trajs = n_trajs, T=T
    )
    wandb.log({"train_loss": train_loss})
    val_loss, conformers_val_noise, conformers_val_gen = gfn_epoch(
        model, val_loader, optimizer, device,  sigma_min, sigma_max, steps, train=False, max_batches=max_batches, n_trajs = n_trajs, T=T
    )
    wandb.log({"val_loss": val_loss})
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
    save_pickle(energies_train_on_policy, 'energies_train_on_policy.pkl')
    save_pickle(energies_train_rand, 'energies_train_rand.pkl')
    save_pickle(logpT_train_on_policy, 'logpT_train_on_policy.pkl')
    save_pickle(logpT_train_rand, 'logpT_train_rand.pkl')
    return conformers_train_gen, conformers_val_gen
    

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def get_energies(conformers):
    energies = {}
    for confs in conformers:
        for conf in confs:
            mol = pyg_to_mol(conf.mol, conf, mmff=True, rmsd=False)
            energy = mmff_energy(mol)
            if conf.canonical_smi not in energies.keys():
                energies[conf.canonical_smi] = [energy]
            else:
                energies[conf.canonical_smi].append(energy)
    return energies