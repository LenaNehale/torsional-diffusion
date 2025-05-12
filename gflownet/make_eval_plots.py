from utils.mols_invariant_feats import *
from utils.dataset import make_dataset_from_smi
from gflownet.gfn_train import gfn_sgd, get_logpT, get_logrew
from diffusion.score_model import TensorProductScoreModel
from utils.dataset import perturb_seeds, pyg_to_mol
from gflownet.gfn_train import get_logrew
from torch_geometric.data import Data, Batch
import pickle
from copy import deepcopy
from utils.torsion import modify_conformer
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import wandb
import math


def load_model(root_path, exp_path, device):
    model = TensorProductScoreModel(in_node_features=74, in_edge_features=4,
                                    sigma_embed_dim=32,
                                    num_conv_layers=4,
                                    max_radius=5.0, radius_embed_dim=50,
                                    scale_by_sigma=True,
                                    use_second_order_repr=False,
                                    residual=True, batch_norm=True)


    model_path = f"{root_path}/model_chkpts"

    model_dir = f"{model_path}/{exp_path}.pt"
    state_dict = torch.load(f'{model_dir}', map_location= torch.device('cuda'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def save_model(model, root_path, exp_path):
    model_path = f"{root_path}/model_chkpts"
    if not os.path.exists(f'{model_path}'):
        os.makedirs(f'{model_path}')
    torch.save(model.state_dict(), f'{model_path}/{exp_path}.pt')


def save_rb(positions_dict, tas_dict, root_path, exp_path, sgd_step):
    replaybuffer_path = f'{root_path}/replay_buffer'
    if not os.path.exists(replaybuffer_path):
        os.makedirs(replaybuffer_path)
    pickle.dump({"rb_positions": positions_dict, "rb_tas": tas_dict}, open(f'{replaybuffer_path}/{exp_path}_{sgd_step}.pkl', 'wb'))


def generate_stuff(model, smis, n_smis_batch, batch_size, diffusion_steps, T, logrew_clamp, energy_fn, device, sigma_min, sigma_max, init_positions_path = None, n_local_structures = None, max_n_local_structures = None, exp_path = None, sgd_step = None, train_mode = 'gflownet', root_path = None): 
 
    assert init_positions_path is not None, 'Please provide a path to the md simulation positions'
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    pos_md = {}
    logrews_md = {}
    mols_md = {}
    
    logrews_gen = {}
    pos_gen = {}
    mols_gen = {}
    tas_gen = {}

    logrews_rand = {}
    pos_rand = {}
    mols_rand = {}
    
    logZs_hat = []
    logZs_md = []


    n_smis = len(smis) 

    for i in range(math.ceil(n_smis / n_smis_batch)):
        smis_subset = smis[n_smis_batch * i : n_smis_batch * (i + 1) ]
        print('smis subset', smis_subset)
        confs_init_smis_subset = make_dataset_from_smi(smis_subset, init_positions_path = init_positions_path, n_local_structures = n_local_structures, max_n_local_structures= max_n_local_structures)

        pos_md.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in confs_init_smis_subset.items()})
        logrews_md.update({ smi: get_logrew(Batch.from_data_list(confs), T , energy_fn = energy_fn, clamp = logrew_clamp).cpu() for smi, confs in confs_init_smis_subset.items()})
        mols_md.update({smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in confs_init_smis_subset.items()})
        train_loss, conformers_gen_subset, logit_pfs, logit_pbs, logrews_gen_subset, perturbs, trajs = gfn_sgd(model, 
                                                                                    confs_init_smis_subset  , 
                                                                                    optimizer, 
                                                                                    device,  
                                                                                    sigma_min = sigma_min, 
                                                                                    sigma_max = sigma_max, 
                                                                                    steps = diffusion_steps, 
                                                                                    train = False, 
                                                                                    batch_size = batch_size,
                                                                                    T=T, 
                                                                                    logrew_clamp = logrew_clamp, 
                                                                                    energy_fn = energy_fn, 
                                                                                    train_mode=train_mode, 
                                                                                    use_wandb = False, 
                                                                                    ReplayBuffer = None, 
                                                                                    p_expl = 0.0, 
                                                                                    p_replay = 0.0, 
                                                                                    grad_acc = False, 
                                                                                    sgd_step = None)


        conformers_gen_subset = {smi: item for smi, item in zip(smis_subset, conformers_gen_subset)}
        
        logrews_gen.update({smi: logrew.cpu() for smi, logrew in zip(smis_subset,logrews_gen_subset ) })
        tas_gen.update({smi: np.stack([conf.total_perturb.cpu().numpy() for conf in confs]) for smi, confs in conformers_gen_subset.items() })
        mols_gen.update({smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_gen_subset.items() })
        pos_gen.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in conformers_gen_subset.items()})

        
        conformers_rand_subset = {smi: perturb_seeds(deepcopy(confs)) for smi, confs in conformers_gen_subset.items()} 
        logrews_rand.update({ smi: get_logrew(Batch.from_data_list(confs), T , energy_fn = energy_fn, clamp = logrew_clamp).cpu() for  smi, confs in conformers_rand_subset.items()})
        pos_rand.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in  conformers_rand_subset.items() })
        mols_rand.update({smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_rand_subset.items() })


        '''
        logZ = torch.logsumexp(torch.stack(logit_pbs) + torch.stack(logrews_gen_subset) - torch.stack(logit_pfs), dim = 1) - np.log(len(logit_pfs[0])).item() #TODO verifier qu'on rajoute la bonne constante?
        logZ = logZ - np.log(sigma_min).item() + np.log(sigma_max).item()
        print('logZ', logZ) 
        logZs_hat.append(logZ)
        print('logZs_hat', logZs_hat)
        '''

    '''
    #logZs_md = [torch.logsumexp(torch.Tensor(logrews_md[smi]) / len(logrews_md[smi]), dim = 0)  for smi in smis]
    logZs_hat = torch.stack(logZs_hat)
    '''

    assert len(logrews_rand) == len(logrews_gen) == len(logrews_md), "the logrews dictionaries don't have the same number of smiles"
    assert len(pos_rand) == len(pos_gen) == len(pos_md), "the positions dictionaries don't have the same number of smiles"
    assert len(mols_rand) == len(mols_gen) == len(mols_md), "the mols dictionaries don't have the same number of smiles"

    
    generated_stuff =  {'logrews': [logrews_rand, logrews_gen, logrews_md], 'positions': [pos_rand, pos_gen, pos_md], 'mols': [mols_rand, mols_gen, mols_md], 'tas': tas_gen,   'logZs': [logZs_md, logZs_hat]} 
    
    if not os.path.exists(f'{root_path}/generated_stuff'):
        os.makedirs(f'{root_path}/generated_stuff')
    if max_n_local_structures == np.inf:
        pickle.dump(generated_stuff, open(f'{root_path}/generated_stuff/{exp_path}_{sgd_step}_all_ls.pkl', 'wb'))
    else:
        pickle.dump(generated_stuff, open(f'{root_path}/generated_stuff/{exp_path}_{sgd_step}.pkl', 'wb'))
    return generated_stuff



def get_logrew_heatmap(data, model, sigma_min, sigma_max,  steps, device, num_points, ix0, ix1, energy_fn, ode, num_trajs, T, logrew_clamp, get_pt = True):
    '''
    Get 2D gt and learned logrews. Both are obtained by computing the energy for different values (linspace) of the 2 torsion angles ix0 and ix1, while fixing the other torsion angles.
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
    - logrew_landscape: 2D array of logreward values
    - logpTs: 2D array of logpT values
    '''
    torsion_angles_linspace = torch.linspace(0, 2*np.pi, num_points)
    logrew_landscape = []
    datas = []
    logpTs = []
    
    #data.total_perturb = torch.zeros( data.mask_rotate.shape[0])
    #assert torch.abs(data.total_perturb).max() == 0
    for theta0 in torsion_angles_linspace:
        datas.append([])
        logrew_landscape.append([])
        for theta1 in torsion_angles_linspace:
            data0 = deepcopy(data)
            torsion_update = np.zeros(len(data0.mask_rotate))
            torsion_update[ix0], torsion_update[ix1] = theta0, theta1
            new_pos = modify_conformer(data0.pos, data0.edge_index.T[data0.edge_mask], data0.mask_rotate, torsion_update, as_numpy=False) #change the value of the 1st torsion angle
            data0.pos = new_pos
            data0.total_perturb = (data.total_perturb + torch.tensor(torsion_update)) % (2 * np.pi)
            #if torch.abs(torch.tensor(torsion_update) % (2 * np.pi) - torch.tensor(torsion_update)).max() > 0 : 
                #breakpoint()
            datas[-1].append(deepcopy(data0))
            logrew_landscape[-1].append(get_logrew(data0, energy_fn = energy_fn, T = T, clamp = logrew_clamp).item())
        if get_pt:
            logpT = get_logpT(datas[-1], model.to(device), sigma_min, sigma_max,  steps, device, ode, num_trajs)
            logpT = logpT.tolist()
            logpTs.append(logpT)
    return np.array(logrew_landscape), np.array(logpTs)


# Visualize the logpTs landscape vs the ground truth energy vs the sampled conformers for a molecule on 2D 

def plot_energy_samples_logpTs(model, smis, generated_stuff, energy_fn, logrew_clamp, init_positions_path, n_local_structures, max_n_local_structures, sigma_min, sigma_max,  steps, device, num_points, num_trajs, T,  plot_energy_landscape, plot_sampled_confs, plot_pt, use_wandb, root_path, exp_path,sgd_step, ode = False, replay_tas = None):
    '''
    
    '''
    if plot_pt == False:
        model = None
    for smi in smis:
        dataset = make_dataset_from_smi([smi], init_positions_path=init_positions_path , n_local_structures = n_local_structures, max_n_local_structures=max_n_local_structures)[smi]
        for ls in range(n_local_structures):
            data = dataset[ls]
            num_torsion_angles = len(data.mask_rotate)
            num_tas_combinations = len(list(itertools.combinations(range(num_torsion_angles), 2)))
            n_columns = int(plot_energy_landscape) + int(plot_sampled_confs) + int(plot_pt)
            fig, ax = plt.subplots(num_tas_combinations,  n_columns, figsize=(4 * n_columns ,  4 * num_tas_combinations))
            ax = np.atleast_2d(ax)
            for ix0, ix1 in itertools.combinations(range(num_torsion_angles), 2):
                row = ix0 //num_torsion_angles + ix1 - 1
                logrew_landscape, logpTs = get_logrew_heatmap(data, model, sigma_min, sigma_max,  steps, device, num_points, ix0 = ix0, ix1 = ix1, energy_fn = energy_fn, ode = ode, num_trajs = num_trajs, T = T  , logrew_clamp = logrew_clamp, get_pt = plot_pt)
                if use_wandb:
                    # log correlation between logpT and logrew
                    corr = np.corrcoef(logpTs.flatten(), logrew_landscape.flatten())[0, 1]
                    wandb.log({f"corr_{smi}_{ix0}_{ix1}": corr}, step = sgd_step)
                    # Log KL divergence between logpT and logrew
                    logrew_normalized = logrew_landscape - np.log(np.exp(logrew_landscape).sum())
                    logpTs_normalized = logpTs - np.log(np.exp(logpTs).sum())
                    reverse_kl =  np.exp(logrew_normalized) * (logrew_normalized - logpTs_normalized)
                    reverse_kl = reverse_kl.sum()
                    forward_kl = np.exp(logpTs_normalized) * (logpTs_normalized - logrew_normalized)
                    forward_kl = forward_kl.sum()   
                    wandb.log({f"KL(PB||PF)_{smi}_{ix0}_{ix1}": reverse_kl }, step = sgd_step)
                    wandb.log({f"KL(PF||PB)_{smi}_{ix0}_{ix1}": forward_kl }, step = sgd_step)
                    jsd = None #todo add it

                
                #print(ix0, ix1)
                # Plot energy landscape
                if plot_energy_landscape:
                    #ax[0].imshow( 100 * np.log(np.array(energy_landscape).transpose()), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
                    print(smi, ix0 //num_torsion_angles + ix1 - 1, 0)
                    ax[ row, 0].imshow(  logrew_landscape.transpose() , extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r', vmin=   np.min(logrew_landscape), vmax=  np.max(logrew_landscape)) #TODO bug to change ! Here i am plotting the tas for all local structures
                    ax[ row, 0].set_title('Logrew Landscape')
                    ax[ row, 0].set_xlabel(f'Torsion Angle {ix0}')
                    ax[row, 0].set_ylabel(f'Torsion Angle {ix1}')
                    fig.colorbar(ax[row, 0].images[0], ax=ax[row, 0], orientation='vertical')
                
                if plot_pt: 
                # Plot logpTs
                    ax[row, 1].imshow(np.array(logpTs).transpose(), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r', vmin=np.min(logpTs), vmax=np.max(logpTs))
                    ax[row, 1].set_title('logpTs Landscape')
                    ax[row, 1].set_xlabel(f'Torsion Angle {ix0}')
                    ax[row, 1].set_ylabel(f'Torsion Angle {ix1}')
                    fig.colorbar(ax[row, 1].images[0], ax=ax[row, 1], orientation='vertical')

                
                
                if plot_sampled_confs:
                    ax[row, 2].scatter(generated_stuff['tas'][smi][:,ix0], generated_stuff['tas'][smi][:,ix1], s = .5, c = 'blue', marker='o', alpha=0.5)
                    if replay_tas is not None and len(replay_tas[smi][0]) > 0:
                        ax[row, 2].scatter(torch.stack(replay_tas[smi][0])[:,ix0].cpu().numpy(), torch.stack(replay_tas[smi][0])[:,ix1].cpu().numpy(), s = .5, c = 'red', marker='^', alpha = 0.5)
                    ax[row, 2].set_title(f'samples GFN (blue) and RB (red)')
                    ax[row, 2].set_xlabel(f'Torsion Angle {ix0}')
                    ax[row, 2].set_ylabel(f'Torsion Angle {ix1}') 
                
                        
                
            plt.tight_layout()
            
            if exp_path is not None:
                if not os.path.exists(f'{root_path}/gfn_samples'):
                    os.makedirs(f'{root_path}/gfn_samples')
                plt.savefig(f"{root_path}/gfn_samples/{exp_path}_{smi}_ls_{ls}_timestep_{sgd_step}.png")
            plt.title('Smi ' + smi + ' - Local structure ' + str(ls))
            plt.show()
            plt.close(fig)
            #if use_wandb:
                #wandb.log({f"energy_samples_logpTs_{smi}": wandb.Image(plt)})
            
        

        
def get_correlations(confs, model, T, sigma_min, sigma_max,  steps, device, num_trajs, energy_fn, logrew_clamp, root_path, exp_path, n_subplots = 5): 
    '''
    Args:
    - confs: dictionary where keys are smiles and values are lists of pytorch geometric conformers
    - model: trained model
    - sigma_min, sigma_max: float, minimum and maximum values of the diffusion step size
    - steps: int, number of diffusion steps
    - device: torch.device
    - num_trajs: int, number of backward trajectories used toapproixmate logpT
    - n_subplots: int, number of subplots per row
    Returns:
    - corrs: dictionary where keys are smiles and values are the correlation coefficient between logpT and logrews
    '''
    logpTs = {}
    logrews = {}
    corrs = {}
    n_smis = len(confs.keys())
    fig, axes = plt.subplots(max(n_smis // n_subplots, 2), n_subplots, figsize=(10, 5))
    for smi_idx, smi in enumerate(confs.keys()):
        logpTs[smi] = get_logpT(confs[smi], model, sigma_min, sigma_max,  steps, device, ode=False, num_trajs = num_trajs).cpu().detach().numpy()
        logrews[smi] = get_logrew(confs[smi], T, energy_fn = energy_fn, clamp = logrew_clamp).cpu().numpy()
        corrs[smi] = np.corrcoef(logpTs[smi], logrews[smi])[0, 1]
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].scatter(logpTs[smi], logrews[smi])
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].set_title(smi)
    fig.suptitle('logpT vs logrews') 
    plt.tight_layout()
    path = f"{root_path}/correlations_plots"
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    plt.savefig(f"{path}/{exp_path}.png")
    plt.close(fig)
    return corrs

def plot_energies_hist_one_smi(generated_stuff, smi, label, ax=None, n_bins = 100, md_key='md', gfn_key='gfn', md_rand_key='md_rand', rew_temp = 0.001987204118 * 298.15):
    logrews_rand, logrews_gen, logrews_md = generated_stuff['logrews']
    data_dict = {md_key: - logrews_md[smi] * rew_temp  , gfn_key:  - logrews_gen[smi] * rew_temp,  md_rand_key:  - logrews_rand[smi] * rew_temp} # convert logrews to energies
    assert len(data_dict[md_key]) == len(data_dict[gfn_key]) == len(data_dict[md_rand_key]), "the logrews dictionaries don't have the same number of conformers for random, generated and md"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    a,b = data_dict[md_key].min(), data_dict[md_key].max()
    range_min = a - 2 * (b - a)
    range_max = b + 2 * (b - a)
    n_bins = 100
    bins=np.linspace(range_min, range_max, n_bins)
    ax.hist(data_dict[md_key], bins=bins, alpha=0.5, color='g', label = 'md', density=True)
    if gfn_key in data_dict.keys() and data_dict[gfn_key] is not None:
        ax.hist(data_dict[gfn_key], bins=bins, alpha=0.5, color='b', label = 'md + gflownet tas', density=True)
    ax.hist(data_dict[md_rand_key], bins=bins, alpha=0.5, color='y', label = 'md + uniform tas', density=True)
    ax.grid(True)
    ax.set_title(f"{smi}, {label}")
    ax.legend(loc='upper right')


def plot_energies_hist(av_smiles, av_smiles_labels, generated_stuff, rew_temp, root_path, exp_path, sgd_step, save=True):
    n_smis = len(av_smiles)
    n_subplots = 5 # number of subplots per row
    n_rows = max(n_smis // n_subplots + int(n_smis % n_subplots > 0), 1)
    fig, axes = plt.subplots(n_rows, n_subplots, figsize=(5*n_subplots, 5*n_rows))
    for idx, (sm, lb) in enumerate(zip(av_smiles, av_smiles_labels)):     
        if len(axes.shape) == 1:
            ax = axes[idx]
        else:
            ax = axes[idx // n_subplots , idx % n_subplots]
        plot_energies_hist_one_smi(generated_stuff, sm, lb, ax, rew_temp)
    plt.tight_layout()
    if save:
        energyhist_path = f'{root_path}/energy_hist'
        if not os.path.exists(energyhist_path):
            os.makedirs(energyhist_path)
        plt.savefig(f'{energyhist_path}/{exp_path}_{sgd_step}', format='png')
        # Save figure with a higher resolution
        plt.savefig(f'{energyhist_path}/{exp_path}_{sgd_step}', format='png', dpi=100)
    plt.show()



def make_localstructures_histograms(generated_stuff, root_path, exp_path, label):
    '''
    Plots histograms of bond lengths, bond angles and torsion angles for generated, ground truth and random molecules, for the same set of smis.
    Args:
    '''
    
    mols_rand, mols_gen, mols_md = generated_stuff['mols']
    smis = list(mols_rand.keys())
    assert mols_gen.keys() == mols_rand.keys() == mols_md.keys()
    fig, axs = plt.subplots(max(len(smis),2), 3, figsize=(18, 6))
    for i, smi in enumerate(smis):
        # Plot bond lengths histogram
        bond_lengths_rand = torch.stack([get_bond_lengths(mol) for mol in mols_rand[smi]]).flatten()
        bond_lengths_gen = torch.stack([get_bond_lengths(mol) for mol in mols_gen[smi]]).flatten()
        bond_lengths_md = torch.stack([get_bond_lengths(mol) for mol in mols_md[smi]]).flatten()
        axs[i, 0].hist(bond_lengths_rand, bins=100, color='blue', alpha=0.5, label = 'rand', density=True)
        axs[i, 0].hist(bond_lengths_gen, bins=100, color='red', alpha=0.5, label = 'gen', density=True)
        axs[i, 0].hist(bond_lengths_md, bins=100, color='green', alpha=0.5, label = 'md', density=True)
        axs[i, 0].set_title('Bond Lengths')
        axs[i, 0].set_ylabel('Frequency')
        axs[i, 0].legend()
        
        
        # Plot bond angles histogram
        bond_angles_rand = torch.stack([get_bond_angles(mol) for mol in mols_rand[smi]]).flatten()
        bond_angles_gen = torch.stack([get_bond_angles(mol) for mol in mols_gen[smi]]).flatten()
        bond_angles_md = torch.stack([get_bond_angles(mol) for mol in mols_md[smi]]).flatten()
        axs[i, 1].hist(bond_angles_rand, bins=100, color='blue', alpha=0.5, label = 'rand', density=True)
        axs[i, 1].hist(bond_angles_gen, bins=100, color='red', alpha=0.5, label = 'gen', density=True)
        axs[i, 1].hist(bond_angles_md, bins=100, color='green', alpha=0.5, label = 'md', density=True)
        axs[i, 1].set_title('Bond Angles')
        axs[i, 1].set_ylabel('Frequency')
        axs[i, 1].legend()
        
        # Plot torsion angles histogram
        torsion_angles_rand = torch.stack([get_torsion_angles(mol) for mol in mols_rand[smi]]).flatten()
        torsion_angles_gen = torch.stack([get_torsion_angles(mol) for mol in mols_gen[smi]]).flatten()
        torsion_angles_md = torch.stack([get_torsion_angles(mol) for mol in mols_md[smi]]).flatten()
        axs[i, 2].hist(torsion_angles_rand, bins=100, color='blue', alpha=0.5, label = 'rand', density=True)
        axs[i, 2].hist(torsion_angles_gen, bins=100, color='red', alpha=0.5, label = 'gen', density=True)
        axs[i, 2].hist(torsion_angles_md, bins=100, color='green', alpha=0.5, label = 'md', density=True)
        axs[i, 2].set_title('Torsion Angles')
        axs[i, 2].set_ylabel('Frequency')
        axs[i, 2].legend()
    
    plt.tight_layout()
    path = f"{root_path}/localstructures_hist"
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    plt.savefig(f"{path}/{exp_path}_{label}.png")
    plt.close(fig)





from sklearn.neighbors import KernelDensity

# Fit KDE plot 
def augment_samples(samples: np.array, exclude_original: bool = False) -> np.array:
        """
        Augments a batch of samples by applying the periodic boundary conditions from
        [0, 2pi) to [-2pi, 4pi) for all dimensions.
        """
        samples_aug = []
        for offsets in itertools.product(
            [-2 * np.pi, 0.0, 2 * np.pi], repeat=samples.shape[-1]
        ):
            if exclude_original and all([offset == 0.0 for offset in offsets]):
                continue
            samples_aug.append(
                np.stack(
                    [samples[:, dim] + offset for dim, offset in enumerate(offsets)],
                    axis=-1,
                )
            )
        samples_aug = np.concatenate(samples_aug, axis=0)
        return samples_aug


def fit_kde(
    #samples: TensorType["batch_size", "state_proxy_dim"],
    samples: np.array,
    kernel: str = "gaussian",
    bandwidth: float = 0.1,
):
    r"""
    Fits a Kernel Density Estimator on a batch of samples.

    The samples are previously augmented in order to account for the periodic
    aspect of the sample space.

    Parameters
    ----------
    samples : tensor
        A batch of samples in proxy format.
    kernel : str
        An identifier of the kernel to use for the density estimation. It must be a
        valid kernel for the scikit-learn method
        :py:meth:`sklearn.neighbors.KernelDensity`.
    bandwidth : float
        The bandwidth of the kernel.
    """
    samples_aug = augment_samples(samples)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples_aug)
    return kde



# show KDE plot 

def plot_kde(
    kde,
    alpha=0.5,
    low=-np.pi * 0.5,
    high=2.5 * np.pi,
    dpi=150,
    colorbar=True,
):
    x = np.linspace(0, 2 * np.pi, 101)
    y = np.linspace(0, 2 * np.pi, 101)
    xx, yy = np.meshgrid(x, y)
    X = np.stack([xx, yy], axis=-1)
    Z = np.exp(kde.score_samples(X.reshape(-1, 2))).reshape(xx.shape)
    # Init figure
    fig, ax = plt.subplots()
    fig.set_dpi(dpi)
    # Plot KDE
    h = ax.contourf(xx, yy, Z, alpha=alpha)
    ax.axis("scaled")
    if colorbar:
        fig.colorbar(h, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, -0.3, r"$0$", fontsize=15)
    ax.text(-0.28, 0, r"$0$", fontsize=15)
    ax.text(2 * np.pi - 0.4, -0.3, r"$2\pi$", fontsize=15)
    ax.text(-0.45, 2 * np.pi - 0.3, r"$2\pi$", fontsize=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Set tight layout
    plt.tight_layout()
    return fig



def plot_kde_2d(generated_stuff, root_path, exp_path, sgd_step): 
     tas_dict = generated_stuff['tas']
     smis = list(tas_dict.keys())
     for smi in smis:
        print('smi', smi)
        tas = tas_dict[smi]
        kde = fit_kde(tas, kernel='gaussian', bandwidth=0.1)
        fig = plot_kde(kde, alpha=0.5, low=-np.pi * 0.5, high=2.5 * np.pi, dpi=150, colorbar=True)
        # save figure in exp_path
        kde_path = f'{root_path}/kde_plots'
        if not os.path.exists(kde_path):
            os.makedirs(kde_path)
        fig.savefig(f'{kde_path}/{exp_path}_{smi}_{sgd_step}.png')
        plt.show()
        plt.close(fig)