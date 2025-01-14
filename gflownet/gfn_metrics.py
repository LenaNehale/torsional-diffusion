from gflownet.gfn_train import * 
from torch.utils.data import Subset




 
    
def get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max,  steps, device, num_points, ix0, ix1, energy_fn, ode, num_trajs, T = 1.0, get_pt = True):
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
            datas[-1].append(copy.deepcopy(data0))
            energy_landscape[-1].append( - get_logrew(data0, energy_fn = energy_fn, T = T).item())
        if get_pt:
            logpT = get_logpT(datas[-1], model.to(device), sigma_min, sigma_max,  steps, device, ode, num_trajs)
            logpT = logpT.tolist()
            logpTs.append(logpT)
    return energy_landscape, logpTs


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




def log_gfn_metrics(model, dataset, optimizer, device, sigma_min, sigma_max, steps, batch_size, T  , num_points , logrew_clamp, energy_fn,  num_trajs, use_wandb, ReplayBuffer, train_mode, gt_data_path, seed):
    # Save the current model in a folder model_chkpts
    if not os.path.exists('model_chkpts'):
        os.makedirs('model_chkpts')
    torch.save(model.state_dict(), f'model_chkpts/model_{energy_fn}_{train_mode}_{seed}.pt')

    #vargrad loss
    train_loss, conformers_train_gen, logit_pfs, logit_pbs, logrews, perturbs, trajs = gfn_sgd(model, dataset, optimizer, device,  sigma_min, sigma_max, steps, train=False, batch_size = batch_size, T=T, logrew_clamp = logrew_clamp, energy_fn=energy_fn, train_mode='gflownet', use_wandb = use_wandb, ReplayBuffer = ReplayBuffer, p_expl = 0, p_replay = 0, grad_acc = False)
    if use_wandb:
        wandb.log({"vargrad loss": torch.mean(train_loss)})

    # Create a folder img
    if not os.path.exists('img'):
        os.makedirs('img')


    for l, gt_batch in enumerate(dataset):

        num_torsion_angles = len(gt_batch.mask_rotate)
        smi_ix = l
        smi = gt_batch.canonical_smi
        
        if gt_data_path is not None:
            dummy_data = pickle.load(open(gt_data_path, 'rb'))[l]
            dummy_data_batch = random.sample(dummy_data, 16)
            #Diffusion loss
            train_loss_diffusion = get_loss_diffusion(model, gt_data_path, sigma_min, sigma_max, device, train = False, use_wandb = use_wandb) #TODO modify to get the loss for a list of smiles
            #RMSD between generated conformers and ground truth conformers
            gt_mols = [pyg_to_mol(dummy_data_batch[i].mol, dummy_data_batch[i], copy=True) for i in range(len(dummy_data_batch))]
            gen_mols = [pyg_to_mol(conformers_train_gen[l][i].mol, conformers_train_gen[l][i], copy=True) for i in range(len(conformers_train_gen[l]))]
            rmsds = np.array([get_rmsds([gt_mols[i] for _ in range(len(gen_mols))], gen_mols) for i in range(len(gt_mols))]) #TODO do per-smile RMSD 
            rmsds = np.min(rmsds, axis=0)
            if use_wandb:
                wandb.log({f"RMSDs gen/ground truth_smi_{smi_ix}_n_{num_torsion_angles}": np.mean(rmsds).item()})




        
        # heatmap of energy/learned logpts 
        data = gt_batch
        energies_off_policy = []
        logpTs_off_policy = []
        #for ix0, ix1 in itertools.combinations(range(num_torsion_angles), 2):

        fig, axes = plt.subplots(5, 3, figsize=(15, 20))

        for idx, (ix0, ix1) in enumerate(itertools.combinations(range(3), 2)):
            energy_landscape, logpTs = get_2dheatmap_array_and_pt(data, model, sigma_min, sigma_max, steps, device, num_points=num_points, ix0=ix0, ix1=ix1, energy_fn=energy_fn, ode=False, num_trajs=num_trajs)
            energy_landscape, logpTs = np.array(energy_landscape), np.array(logpTs)

            row, col = 0, idx
            ax = axes[row, col]
            im = ax.imshow(energy_landscape.transpose() / T , extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r', vmin=np.min(energy_landscape) / T, vmax=np.max(energy_landscape) / T)
            fig.colorbar(im, ax=ax, label=f'Energy, T = {T}')
            ax.set_xlabel(f'Theta{ix0}')
            ax.set_ylabel(f'Theta{ix1}')
            ax.set_title("Energy Landscape")

            row = 1
            ax = axes[row, col]
            im = ax.imshow(logpTs.transpose(), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
            fig.colorbar(im, ax=ax, label='logpTs')
            ax.set_xlabel(f'Theta{ix0}')
            ax.set_ylabel(f'Theta{ix1}')
            ax.set_title("logpTs")

            row = 2
            ax = axes[row, col]
            im = ax.imshow(np.exp(logpTs).transpose(), extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis_r')
            fig.colorbar(im, ax=ax, label='pTs')
            ax.set_xlabel(f'Theta{ix0}')
            ax.set_ylabel(f'Theta{ix1}')
            ax.set_title("pTs")

            energies_off_policy.append(energy_landscape)
            logpTs_off_policy.append(logpTs)
            
            row = 3
            ax = axes[row, col]
            theta0, theta1 = perturbs[l].reshape(-1, num_torsion_angles)[:, ix0] % (2 * np.pi), perturbs[l].reshape(-1, num_torsion_angles)[:, ix1] % (2 * np.pi)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)
            ax.scatter(theta0.to('cpu'), theta1.to('cpu'), c='r', s=30, alpha=0.25, marker='o')
            ax.set_xlabel(f'Theta{ix0}')
            ax.set_ylabel(f'Theta{ix1}')
            ax.set_title("Samples")

            if gt_data_path is not None: 
                # plot the ground truth torsion angles from gt_data_path
                dummy_data = pickle.load(open(gt_data_path, 'rb'))[l]
                gt_total_perturb = Batch.from_data_list(dummy_data).total_perturb.reshape(-1, num_torsion_angles)
                gt_theta0, gt_theta1 = gt_total_perturb[:,ix0]%(2*np.pi) , gt_total_perturb[:,ix1]%(2*np.pi)
                ax.scatter(gt_theta0, gt_theta1, c='b', s=30, alpha=0.25, marker = '^')
            if ReplayBuffer is not None and len(ReplayBuffer) > 0:  
                #Plot traj[-1] in the replay buffer
                traj_replay, _ = ReplayBuffer.sample(l, min(ReplayBuffer.max_size, len(ReplayBuffer)), smi = smi)
                perturbs_replay = traj_replay[-1].total_perturb.reshape(-1, num_torsion_angles)
                theta0replay, theta1replay = perturbs_replay[:,ix0]%(2*np.pi) , perturbs_replay[:,ix1]%(2*np.pi)
                ax.scatter(theta0replay, theta1replay, c='g', s=30, alpha=0.25, marker = 'x')


            row = 4 
            ax = axes[row, col]
            if col == 0:
                rdDepictor.Compute2DCoords(data.mol)
                img = Draw.MolToImage(data.mol)
                ax.imshow(img)
                ax.axis('off')
            elif col == 1:
                # Scatter plot of logrews and logpTs (on-policy + off-policy, uniformly sampled on the grid )
                k = 60 # base value 60
                logrews_on_policy = logrews[l].cpu().detach().numpy()
                logpTs_on_policy = get_logpT(Batch.to_data_list(trajs[l][-1])[:k], model.to(device), sigma_min, sigma_max,  steps, device, ode=False, num_trajs = 8)
                correlation = np.corrcoef(logrews_on_policy[:k], logpTs_on_policy.cpu())[0, 1].item()
                if use_wandb:
                    wandb.log({f"correlation_logrew_logpTs_on_policy_smi_{smi_ix}_n_{num_torsion_angles}": correlation})    

                logrews_off_policy = - np.stack(energies_off_policy).flatten()
                if use_wandb:
                    wandb.log({f"logrews_off_policy_smi_{smi_ix}_n_{num_torsion_angles}": logrews_off_policy.mean(), f"logrews_on_policy_smi_{smi_ix}_n_{num_torsion_angles}": logrews_on_policy.mean()})

                logrews_all = logrews_on_policy[:k] / T
                logpTs = np.array(logpTs_on_policy.cpu())
                #logrews_all = np.concatenate((logrews_on_policy[:k], logrews_off_policy)) / T
                #logpTs = np.concatenate((np.array(logpTs_on_policy), np.stack(logpTs_off_policy).flatten()))

                ax.scatter(logrews_all, logpTs, c='b', s=15)
                ax.set_xlabel('logrews')
                ax.set_ylabel('logpTs')
                ax.set_title('Scatter plot of logrews and logpTs')

            elif col == 2:
                # Plot evolution of total_perturb in forward pass
                traj_perturbs = torch.stack([x.total_perturb for x in trajs[l]])
                # Plot a figure with a line evolution of the total perturbations. Dimensions of traj_perturbs are (timestep, traj_id)
                ax.plot(traj_perturbs.cpu().detach().numpy()[:, :64])
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Total Perturb')
                ax.set_title('Total Perturbations in Forward Pass')
    
    
        plt.tight_layout()
        plt.savefig(f"img/combined_plot_{energy_fn}_{train_mode}_{seed}_smi_{smi_ix}_n_{num_torsion_angles}.png")
        plt.close()
        if use_wandb:
            wandb.log({f"combined_plot_smi_{smi_ix}_n_{num_torsion_angles}": wandb.Image(f"img/combined_plot_{energy_fn}_{train_mode}_{seed}_smi_{smi_ix}_n_{num_torsion_angles}.png")})

        # logZ
        logZ = torch.logsumexp(logit_pbs[l] + logrews[l] - logit_pfs[l], dim = 0) - np.log(len(logit_pfs[l])).item()
        logZ = logZ - np.log(sigma_min).item() + np.log(sigma_max).item()
        if use_wandb:
            wandb.log({f"logZ_hat_smi_{smi_ix}_n_{num_torsion_angles}": logZ})




def log_gfn_metrics_cond(model, dataset, optimizer, device, sigma_min, sigma_max, steps, n_smis_batch, batch_size, T , logrew_clamp, energy_fn,  num_trajs, use_wandb, ReplayBuffer, train_mode, seed):

    train_losses = []
    correlations_on_policy = []
    avg_logrews_on_policy = []
    avg_logrews_gt = []
    avg_logrews_rand = []
    logZs = []
    logZs_mcmc = []


    for i in tqdm(range(len(dataset) // n_smis_batch)): 
        subset_indices = [i * n_smis_batch + j for j in range(n_smis_batch)] 
        subset = Subset(dataset, subset_indices)
        #train loss
        train_loss, conformers_train_gen, logit_pfs, logit_pbs, logrews, perturbs, trajs = gfn_sgd(model, subset, optimizer, device,  sigma_min, sigma_max, steps, train=False, batch_size = batch_size, T=T, logrew_clamp = logrew_clamp, energy_fn=energy_fn, train_mode='gflownet', use_wandb = use_wandb, ReplayBuffer = ReplayBuffer, p_expl = 0, p_replay = 0, grad_acc = False)
        train_losses.append(train_loss)
        #Corr(logpT, logrew). 
        logpT = [get_logpT(x, model, sigma_min, sigma_max,  steps, device, ode=False, num_trajs = num_trajs) for x in conformers_train_gen ]
        logpT_one_traj = logit_pfs - logit_pbs 
        correlation = []
        for  j in range(n_smis_batch):
            correlation.append(np.corrcoef(logpT[j].cpu().detach().numpy(), logrews[j].cpu().detach().numpy())[0, 1])
        correlations_on_policy.append(correlation)
        # avg logrews on policy
        avg_logrews_on_policy.append(reduce(logrews, 'n_smis bs -> n_smis', 'mean' ))
        # avg logrews rand
        conformers_train_rand = [perturb_seeds(x) for x in conformers_train_gen]
        logrews_rand = torch.stack([get_logrew(Batch.from_data_list(x), energy_fn = energy_fn, T = T, clamp = logrew_clamp) for x in conformers_train_rand])
        avg_logrews_rand.append(reduce(logrews_rand, 'n_smis bs -> n_smis', 'mean' ))
        # logrews gt 
        logrews_gt = get_logrew(Batch.from_data_list([x for x in subset]), energy_fn = energy_fn, T = T, clamp = logrew_clamp) 
        avg_logrews_gt.append(logrews_gt)

        #logZs 
        logZ = torch.logsumexp(logit_pbs + logrews - logit_pfs, dim = 1) - np.log(len(logit_pfs[0])).item() #TODO verifier qu'on rajoute la bonne constante?
        logZ = logZ - np.log(sigma_min).item() + np.log(sigma_max).item()
        logZs.append(logZ)
    
        #logZs_mcmc

    train_losses, correlations_on_policy, avg_logrews_on_policy, avg_logrews_rand, avg_logrews_gt, logZs = torch.stack(train_losses).flatten(), torch.Tensor(correlations_on_policy).flatten(), torch.stack(avg_logrews_on_policy).flatten(), torch.stack(avg_logrews_rand).flatten(), torch.stack(avg_logrews_gt).flatten(), torch.stack(logZs).flatten()


    fig, ax = plt.subplots(2)
    x = np.arange(len(dataset))
    ax[0].bar(x, correlations_on_policy, color ='r', label ='correlations_on_policy') 
    #ax[0].set_xticks(x)
    ax[0].set_xlabel('SMILES')
    ax[0].set_ylabel('correlation')
    #ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].bar(x, train_losses.cpu(), color ='b', label ='train_losses') 
    #ax[1].set_xticks(x)
    ax[1].set_xlabel('SMILES')
    ax[1].set_ylabel('train_loss')
    ax[1].set_yscale('log')
    ax[1].legend()

    plt.savefig(f"img/barplot_{energy_fn}_{train_mode}_{seed}.png")
    plt.close()

    fig, ax = plt.subplots(2,2)

    ax[0,0].scatter(logZs.cpu(), train_losses.cpu(), c='b', s=15, marker = 'o')
    ax[0,0].set_xlabel('logZs')
    #ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel('train_losses')
    ax[0,0].set_yscale('log')


    ax[0,1].scatter(avg_logrews_rand, avg_logrews_on_policy.cpu(), c='y', s=15, marker = '^')
    ax[0,1].set_xlabel('avg_logrews_rand')
    #ax[0,1].set_xscale('log')
    ax[0,1].set_ylabel('avg_logrews_on_policy')
    #ax[0,1].set_yscale('log')

    ax[1,0].scatter(train_losses.cpu(), correlations_on_policy, c='r', s=15, marker = 'x')
    ax[1,0].set_xlabel('train_losses')
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel('correlations_on_policy')

    ax[1,1].scatter(avg_logrews_gt, avg_logrews_on_policy.cpu(), c='g', s=15, marker = '^')
    ax[1,1].set_xlabel('avg_logrews_gt')
    #ax[1,1].set_xscale('log')
    ax[1,1].set_ylabel('avg_logrews_on_policy')
    #ax[1,1].set_yscale('log')


    plt.tight_layout()
    plt.savefig(f"img/scatterplot_{energy_fn}_{train_mode}_{seed}.png")
    plt.close()
    
    

    
    if use_wandb:
        wandb.log({f"barplot_{energy_fn}_{train_mode}_{seed}": wandb.Image(f"img/barplot_{energy_fn}_{train_mode}_{seed}.png")})
        wandb.log({f"scatterplot_{energy_fn}_{train_mode}_{seed}": wandb.Image(f"img/scatterplot_{energy_fn}_{train_mode}_{seed}.png")})
    
    
    #return train_losses, correlations_on_policy, avg_logrews_on_policy, avg_logrews_rand, avg_logrews_gt, logZs

        




    
    

    