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







def load_model(exp_path, device):
    model = TensorProductScoreModel(in_node_features=74, in_edge_features=4,
                                    sigma_embed_dim=32,
                                    num_conv_layers=4,
                                    max_radius=5.0, radius_embed_dim=50,
                                    scale_by_sigma=True,
                                    use_second_order_repr=False,
                                    residual=True, batch_norm=True)


    model_path = "/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFN/model_chkpts"

    model_dir = f"{model_path}/{exp_path}.pt"
    state_dict = torch.load(f'{model_dir}', map_location= torch.device('cuda'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def generate_stuff(model, smis, n_smis_batch, batch_size, diffusion_steps, T, logrew_clamp, energy_fn, device, sigma_min, sigma_max, init_positions_path = None, n_local_structures = 1, train_mode = 'gflownet'): 
 
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    logrews_gen_all = {}
    pos_gen = {}
    mols_gen = {}
    num_tas = {}
    tas_all = {}

    logrews_rand = {}
    pos_rand = {}
    mols_rand = {}
    
    logZs_hat = []
    logZs_md = []


    n_smis = len(smis)

    for i in range(n_smis // n_smis_batch):
        smis_subset = smis[n_smis_batch * i : n_smis_batch * (i + 1) ]
        print('smis subset', smis_subset)
        confs_rdkit_from_smi = make_dataset_from_smi(smis_subset, init_positions_path = init_positions_path, n_local_structures = n_local_structures)
        train_loss, conformers_gen_subset, logit_pfs, logit_pbs, logrews_gen_subset, perturbs, trajs = gfn_sgd(model, 
                                                                                    confs_rdkit_from_smi  , 
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
                                                                                    grad_acc = False)


        conformers_gen_subset = {smi: item for smi, item in zip(smis_subset, conformers_gen_subset)}
        logrews_gen_all.update({smi: logrew.cpu() for smi, logrew in zip(smis_subset,logrews_gen_subset ) })

        tas_all.update({smi: np.stack([conf.total_perturb.cpu().numpy() for conf in confs]) for smi, confs in conformers_gen_subset.items() })
        
        mols_gen.update({smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_gen_subset.items() })
        pos_gen.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in conformers_gen_subset.items()})

        
        conformers_rand_subset = {smi: perturb_seeds(deepcopy(confs)) for smi, confs in conformers_gen_subset.items()} 
        logrews_rand.update({ smi: get_logrew(Batch.from_data_list(confs), T , energy_fn = energy_fn, clamp = logrew_clamp).cpu() for  smi, confs in conformers_rand_subset.items()})
        pos_rand.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in  conformers_rand_subset.items() })
        mols_rand.update({smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_rand_subset.items() })

            
        num_tas.update({smi: len(confs[0].mask_rotate)  for smi, confs in  conformers_rand_subset.items()   })

        '''
        logZ = torch.logsumexp(torch.stack(logit_pbs) + torch.stack(logrews_gen_subset) - torch.stack(logit_pfs), dim = 1) - np.log(len(logit_pfs[0])).item() #TODO verifier qu'on rajoute la bonne constante?
        logZ = logZ - np.log(sigma_min).item() + np.log(sigma_max).item()
        print('logZ', logZ) 
        logZs_hat.append(logZ)
        print('logZs_hat', logZs_hat)
        '''
        

    pos_md, logrews_md, mols_md = pickle.load(open("data/pos_md.pkl", 'rb')), pickle.load(open("data/logrews_md.pkl", 'rb')), pickle.load(open("data/mols_md.pkl", 'rb'))
    pos_md = {smi:pos_md[smi] for smi in smis}
    logrews_md = {smi:logrews_md[smi] for smi in smis}
    mols_md = {smi:mols_md[smi] for smi in smis}
    '''
    #logZs_md = [torch.logsumexp(torch.Tensor(logrews_md[smi]) / len(logrews_md[smi]), dim = 0)  for smi in smis]
    logZs_hat = torch.stack(logZs_hat)
    '''
    
    for smi in smis:
        try:
            print('###################')
            print(f'logrew medians for {smi}: rand {logrews_rand[smi].median()} gen {logrews_gen_all[smi].median()} md {logrews_md[smi].median()} ' )
            print(f'logrew means for {smi}: rand {logrews_rand[smi].mean()} gen {logrews_gen_all[smi].mean()} md {logrews_md[smi].mean()} ')
        except:
            pass

    assert len(logrews_rand) == len(logrews_gen_all) == len(logrews_md)
    assert len(pos_rand) == len(pos_gen) == len(pos_md)
    assert len(mols_rand) == len(mols_gen) == len(mols_md)
    
    return {'logrews': [logrews_rand, logrews_gen_all, logrews_md], 'positions': [pos_rand, pos_gen, pos_md], 'mols': [mols_rand, mols_gen, mols_md], 'tas': tas_all,   'logZs': [logZs_md, logZs_hat]} 



def get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max,  steps, device, num_points, ix0, ix1, energy_fn, ode, num_trajs, T, get_pt = True):
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
    
    data.total_perturb = torch.zeros( data.mask_rotate.shape[0])
    assert torch.abs(data.total_perturb).max() == 0
    for theta0 in torsion_angles_linspace:
        datas.append([])
        energy_landscape.append([])
        for theta1 in torsion_angles_linspace:
            data0 = deepcopy(data)
            torsion_update = np.zeros(len(data0.mask_rotate))
            torsion_update[ix0], torsion_update[ix1] = theta0, theta1
            new_pos = modify_conformer(data0.pos, data0.edge_index.T[data0.edge_mask], data0.mask_rotate, torsion_update, as_numpy=False) #change the value of the 1st torsion angle
            data0.pos = new_pos
            data0.total_perturb = torch.tensor(torsion_update) % (2 * np.pi)
            #if torch.abs(torch.tensor(torsion_update) % (2 * np.pi) - torch.tensor(torsion_update)).max() > 0 : 
                #breakpoint()
            datas[-1].append(deepcopy(data0))
            energy_landscape[-1].append( - get_logrew(data0, energy_fn = energy_fn, T = T).item())
        if get_pt:
            logpT = get_logpT(datas[-1], model.to(device), sigma_min, sigma_max,  steps, device, ode, num_trajs)
            logpT = logpT.tolist()
            logpTs.append(logpT)
    return energy_landscape, logpTs



def get_correlations(confs, model, T, sigma_min, sigma_max,  steps, device, num_trajs, energy_fn, logrew_clamp, exp_path, n_subplots = 5): 
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
    plt.savefig(f"correlations_{exp_path}.png")
    plt.close(fig)
    return corrs

def make_logrew_histograms(generated_stuff, exp_path, label, range = 2):
    '''
    Plots histograms of logrews for generated, ground truth and random conformers, for the same set of smis.
    Args:
        - logrews_gen: dictionary where keys are smiles and values are tensors of logrews for generated conformers
        - logrews_gt: dictionary where keys are smiles and values are tensors of logrews for ground truth conformers
        - logrews_random: dictionary where keys are smiles and values are tensors of logrews for random conformers
    Returns:
        A figure of shape (n_smis // 5, 5) where each subplot coresponds to a different smile and shows the histograms of logrews for generated, ground truth and random conformers.

    '''
    
    
    
    logrews_random, logrews_gen, logrews_gt = generated_stuff['logrews']    
    
    
    
    assert logrews_gen.keys() == logrews_random.keys() == logrews_gt.keys()
    smis = logrews_gen.keys()
    n_smis = len(smis)
    n_subplots = 5 # number of subplots per row
    fig, axes = plt.subplots(max(n_smis // n_subplots, 2), n_subplots, figsize=(10, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    for smi_idx, smi in enumerate(smis):

        a , b = torch.Tensor(logrews_gt[smi]).min().item() ,  torch.Tensor(logrews_gt[smi]).max().item()
        range_min = a - range * (b - a)
        range_max = b + range * (b - a)
        n_bins = 100
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(  logrews_random[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='r', label = 'random', density=True)
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(logrews_gen[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='b', label = 'generated', density=True)
        axes[ smi_idx //n_subplots , smi_idx % n_subplots ].hist(logrews_gt[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='g', label = 'ground truth', density=True)
        axes[smi_idx // n_subplots, smi_idx % n_subplots].set_title(smi)
        
        
    fig.suptitle('logrews distribution') 
    
    fig.legend(['random', 'generated','ground truth'], loc='upper right')
    plt.tight_layout() 
    plt.savefig(f"logrewshist_{exp_path}_{label}.png")
    plt.close(fig)



def make_localstructures_histograms(generated_stuff, exp_path, label):
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
    plt.savefig(f"localstructures_{exp_path}_{label}.png")
    plt.close(fig)





'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smis_train', type=str, default='CC(C)CC1NC(=S)N(Cc2ccccc2)C1=O', help='train SMILES')
parser.add_argument('--smis_test', type=str, default='CC(C)CC1NC(=S)N(Cc2ccccc2)C1=O', help='test SMILES')
parser.add_argument('--p_expl', type=float, default=0.2, help='p_expl')
parser.add_argument('--p_replay', type=float, default=0.2, help='p_replay')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--diffusion_steps', type=int, default=20, help='diffusion_steps')
args = parser.parse_args()
if __name__ == '__main__':

    # load params
    ## general model params
    device = torch.device('cuda')
    sigma_min = 0.01 * np.pi
    sigma_max = np.pi
    ## energy params
    energy_fn = 'mmff'
    seed = 0
    k_b = 0.001987204118 
    room_temp = 298.15
    T = k_b * room_temp
    logrew_clamp = -1e5

    
    
    exp_path = f"gflownet_{args.energy_fn}_{args.seed}_limit_train_mols_{len(args.smis_train)}_dataset_freesolv_p_replay_{args.p_replay}_p_expl_{args.p_expl}_diffusion_steps_{args.diffusion_steps}"
    if len(args.smis_train) == 1 : 
        exp_path += f"_smi_{args.smis[0]}"
    model = load_model(exp_path)
    # generate stuff on the train data and test data
    for smis, label in zip([args.smis_train, args.smis_test], ['train', 'test']):
        generated_stuff = generate_stuff(model, smis, args.n_smis_batch, args.batch_size, args.diffusion_steps, args.T, args.logrew_clamp, args.energy_fn)

        logrews_rand, logrews_gen, logrews_md = generated_stuff['logrews']
        pos_rand, pos_gen, pos_md = generated_stuff['positions']
        mols_rand, mols_gen, mols_md = generated_stuff['mols']
        num_tas = generated_stuff['num_tas']

        make_logrew_histograms(logrews_rand, logrews_gen, logrews_md, exp_path, label)
        make_localstructures_hist(mols_rand, mols_gen, mols_md, exp_path, label)
        
    
'''  
