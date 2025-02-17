from utils.eval_plots import *
from utils.dataset import make_dataset_from_smi
from gflownet.gfn_train import gfn_sgd
import argparse
from diffusion.score_model import TensorProductScoreModel
from diffusion.sampling import perturb_seeds, pyg_to_mol
from gflownet.gfn_train import get_logrew
from torch_geometric.data import Data, Batch
import pickle
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--smis_train', type=str, default='CC(C)CC1NC(=S)N(Cc2ccccc2)C1=O', help='train SMILES')
parser.add_argument('--smis_test', type=str, default='CC(C)CC1NC(=S)N(Cc2ccccc2)C1=O', help='test SMILES')
parser.add_argument('--p_expl', type=float, default=0.2, help='p_expl')
parser.add_argument('--p_replay', type=float, default=0.2, help='p_replay')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--diffusion_steps', type=int, default=20, help='diffusion_steps')
args = parser.parse_args()



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

def load_model(exp_path):
    model = TensorProductScoreModel(in_node_features=74, in_edge_features=4,
                                    sigma_embed_dim=32,
                                    num_conv_layers=4,
                                    max_radius=5.0, radius_embed_dim=50,
                                    scale_by_sigma=True,
                                    use_second_order_repr=False,
                                    residual=True, batch_norm=True)


    model_path = "/home/mila/l/lena-nehale.ezzine/scratch/torsionalGFNmodel_chkpts"

    model_dir = f"{model_path}/{exp_path}.pt"
    state_dict = torch.load(f'{model_dir}', map_location= torch.device('cuda'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def generate_stuff(model, smis, n_smis_batch = 10): 
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    logrews_gen = {}
    conformers_gen = {}
    pos_gen = {}
    num_tas = {}

    logrews_rand = {}
    conformers_rand = {}
    pos_rand = {}

    logZs_hat = []



    n_smis = len(smis)

    for i in range(n_smis // n_smis_batch):
        smis_subset = smis[n_smis_batch * i : n_smis_batch * (i + 1) ]
        print('smis subset', smis_subset)
        confs_rdkit_from_smi = make_dataset_from_smi(smis_subset)
        train_loss, confs, logit_pfs, logit_pbs, logrews, perturbs, trajs = gfn_sgd(model, 
                                                                                    confs_rdkit_from_smi  , 
                                                                                    optimizer, 
                                                                                    device,  
                                                                                    sigma_min = sigma_min, 
                                                                                    sigma_max = sigma_max, 
                                                                                    steps = args.diffusion_steps, 
                                                                                    train=False, 
                                                                                    batch_size = args.batch_size,
                                                                                    T=T, 
                                                                                    logrew_clamp = logrew_clamp, 
                                                                                    energy_fn = energy_fn, 
                                                                                    train_mode='gflownet', 
                                                                                    use_wandb = False, 
                                                                                    ReplayBuffer = None, 
                                                                                    p_expl = args.p_expl, 
                                                                                    p_replay = args.p_replay, 
                                                                                    grad_acc = False)


        logrews_gen.update({smi: logrew_gen.cpu() for smi, logrew_gen in zip(smis, logrews) })
        conformers_gen.update({smi: conf for smi, conf in zip(smis, confs) })
        pos_gen.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in conformers_gen.items()})
        mols_gen = {smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_gen.items()}
        num_tas.update({smi: len(conf[0].mask_rotate)  for smi, conf in conformers_gen.items()})


        conformers_rand.update({smi: perturb_seeds(deepcopy(conf)) for smi, conf in conformers_gen.items()} ) 
        logrews_rand.update({ smi: get_logrew(Batch.from_data_list(conf), T , energy_fn = energy_fn, clamp = logrew_clamp).cpu() for  smi, conf in conformers_rand.items()})
        pos_rand.update({smi: np.array([conf.pos.cpu().numpy() for conf in confs ]) for smi, confs in conformers_rand.items()})
        mols_rand = {smi: [pyg_to_mol(conf.mol, conf, mmff=False, rmsd=False, copy=True) for conf in confs] for smi, confs in conformers_rand.items()}

        logZ = torch.logsumexp(logit_pbs + logrews - logit_pfs, dim = 1) - np.log(len(logit_pfs[0])).item() #TODO verifier qu'on rajoute la bonne constante?
        logZ = logZ - np.log(sigma_min).item() + np.log(sigma_max).item()
        logZs_hat.append(logZ)



    for smi in args.smis:
        print(f'logrew medians for {smi}: rand {logrews_rand[smi].median()} gen {logrews_gen[smi].median()} md {logrews_md[smi].median()} ' )


    pos_md, logrews_md, mols_md = pickle.load(open("data/mdtrajs_dict_reordered.pkl", 'rb')), pickle.load(open("data/logrews_md.pkl", 'rb')), pickle.load(open("data/mols_md", 'rb'))


    logZs_md = [torch.logsumexp(logrews_md[smi] / len( logrews_md[smi] ))  for smi in logrews_md.keys()]


    
    
    return {'logrews': [logrews_rand, logrews_gen, logrews_md], 'positions': [pos_rand, pos_gen, pos_md], 'mols': [mols_rand, mols_gen, mols_md], 'num_tas': num_tas, 'logZs_hat': logZs_hat, 'logZs_md': logZs_md} 





    
if __name__ == '__main__':


    
    if len(args.smis_train) == 53:
        exp_path = f"gflownet_{energy_fn}_{seed}_limit_train_mols_{len(args.smis_train)}_dataset_freesolv_p_replay_{args.p_replay}_p_expl_{args.p_expl}_diffusion_steps_{args.diffusion_steps}"
    elif len(args.smis_test) == 80:
        exp_path = f"gflownet_{energy_fn}_{seed}_limit_train_mols_{len(args.smis_train)}_dataset_freesolv_p_replay_{args.p_replay}_p_expl_{args.p_expl}"
    if len(args.smis_train) == 1 : 
        exp_path += f"_smi_{args.smis[0]}"
    model = load_model(exp_path)
    # generate stuff on the train data and test data
    for smis, label in zip([args.smis_train, args.smis_test], ['train', 'test']):
        generated_stuff = generate_stuff(model, smis )
        
        
        # save stuff
        pickle.dump(generated_stuff['logrews'], open(f'"/home/mila/l/lena-nehale.ezzine/scratch/TorsionalGFNgenerated_logrews/{exp_path}_{label}.pkl', 'wb')) # in case hist is not generated well
        pickle.dump(generated_stuff['positions'], open(f'"/home/mila/l/lena-nehale.ezzine/scratch/TorsionalGFNgenerated_positions/{exp_path}_{label}.pkl', 'wb')) # for tica


        logrews_rand, logrews_gen, logrews_md = generated_stuff['logrews']
        pos_rand, pos_gen, pos_md = generated_stuff['positions']
        mols_rand, mols_gen, mols_md = generated_stuff['mols']
        num_tas = generated_stuff['num_tas']

        make_logrew_histograms(logrews_rand, logrews_gen, logrews_md, exp_path, label)
        make_localstructures_hist(mols_rand, mols_gen, mols_md, exp_path, label)
        