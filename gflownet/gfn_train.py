from diffusion.sampling import *



def sample_traj(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None, pg_weight_log_0=None, pg_repulsive_weight_log_0=None,
           pg_weight_log_1=None, pg_repulsive_weight_log_1=None, pg_kernel_size_log_0=None,
           pg_kernel_size_log_1=None, pg_langevin_weight_log_0=None, pg_langevin_weight_log_1=None,
           pg_invariant=False, mol=None):

    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps


    if pg_weight_log_0 is not None and pg_weight_log_1 is not None:
        edge_index, edge_mask = conformers[0].edge_index, conformers[0].edge_mask
        edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

        for p in edge_index.T:
            edge_list[p[0]].append(p[1])

        rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

        dihedral = []
        for a, b in rot_bonds:
            c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
            d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
            dihedral.append((c.item(), a.item(), b.item(), d.item()))
        dihedral_numpy = np.asarray(dihedral)
        dihedral = torch.tensor(dihedral)

        if pg_invariant:
            try:
                with time_limit(10):
                    mol = molecule.Molecule.from_rdkit(mol)

                    aprops = mol.atomicnums
                    am = mol.adjacency_matrix

                    # Convert molecules to graphs
                    G = graph.graph_from_adjacency_matrix(am, aprops)

                    # Get all the possible graph isomorphisms
                    isomorphisms = graph.match_graphs(G, G)
                    isomorphisms = [iso[0] for iso in isomorphisms]
                    isomorphisms = np.asarray(isomorphisms)

                    # filter out those having an effect on the dihedrals
                    dih_iso = isomorphisms[:, dihedral_numpy]
                    dih_iso = np.unique(dih_iso, axis=0)

                    if len(dih_iso) > 32:
                        print("reduce isomorphisms from", len(dih_iso), "to", 32)
                        dih_iso = dih_iso[np.random.choice(len(dih_iso), replace=False, size=32)]
                    else:
                        print("isomorphisms", len(dih_iso))
                    dih_iso = torch.from_numpy(dih_iso).to(device)

            except TimeoutException as e:
                print("Timeout generating with non invariant kernel")
                pg_invariant = False
    
    for batch_idx, data in enumerate(loader):
        bs = data.num_graphs
        n_torsion_angles = len(data.total_perturb[0]) 
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))
        trajs = torch.zeros(len(sigma_schedule) + 1, bs, n_torsion_angles)

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()

            t = sigma_idx / steps   # t is really 1-t
            pg_weight = 10**(pg_weight_log_0 * t + pg_weight_log_1 * (1 - t)) if pg_weight_log_0 is not None and pg_weight_log_1 is not None else 0.0
            pg_repulsive_weight = 10**(pg_repulsive_weight_log_0 * t + pg_repulsive_weight_log_1 * (1 - t)) if pg_repulsive_weight_log_0 is not None and pg_repulsive_weight_log_1 is not None else 1.0

            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            if pg_weight > 0:
                n = data.num_graphs
                if pg_invariant:
                    S, D, _ = dih_iso.shape
                    dih_iso_cat = dih_iso.reshape(-1, 4)
                    tau = get_torsion_angles(dih_iso_cat, data_gpu.pos, n)
                    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
                    tau_diff = torch.fmod(tau_diff + 3 * np.pi, 2 * np.pi) - np.pi
                    tau_diff = tau_diff.reshape(n, n, S, D)
                    tau_matrix = torch.sum(tau_diff ** 2, dim=-1, keepdim=True)
                    tau_matrix, indices = torch.min(tau_matrix, dim=2)
                    tau_diff = torch.gather(tau_diff, 2, indices.unsqueeze(-1).repeat(1, 1, 1, D)).squeeze(2)
                else:
                    tau = get_torsion_angles(dihedral, data_gpu.pos, n)
                    tau_diff = tau.unsqueeze(1) - tau.unsqueeze(0)
                    tau_diff = torch.fmod(tau_diff+3*np.pi, 2*np.pi)-np.pi
                    assert torch.all(tau_diff < np.pi + 0.1) and torch.all(tau_diff > -np.pi - 0.1), tau_diff
                    tau_matrix = torch.sum(tau_diff**2, dim=-1, keepdim=True)

                kernel_size = 10 ** (pg_kernel_size_log_0 * t + pg_kernel_size_log_1 * (1 - t)) if pg_kernel_size_log_0 is not None and pg_kernel_size_log_1 is not None else 1.0
                langevin_weight = 10 ** (pg_langevin_weight_log_0 * t + pg_langevin_weight_log_1 * (1 - t)) if pg_langevin_weight_log_0 is not None and pg_langevin_weight_log_1 is not None else 1.0

                k = torch.exp(-1 / kernel_size * tau_matrix)
                repulsive = torch.sum(2/kernel_size*tau_diff*k, dim=1).cpu().reshape(-1) / n

                perturb = (0.5 * g ** 2 * eps * score) + langevin_weight * (0.5 * g ** 2 * eps * score + g * np.sqrt(eps) * z)
                perturb += pg_weight * (g ** 2 * eps * (score + pg_repulsive_weight * repulsive))
            
                mean, std = 0.5 * g ** 2 * eps * score + langevin_weight * (0.5 * g ** 2 * eps * score)  ,  langevin_weight * g * np.sqrt(eps)+torch.eye(data_gpu.edge_pred.shape[0])
            else:
                mean, std = g ** 2 * eps * score, g * np.sqrt(eps)*torch.eye(data_gpu.edge_pred.shape[0])
            
            # compute the forward and backward (in gflownet language) transitions logprobs
            for i in range(bs):
                start, end = i*n_torsion_angles, (i+1)*n_torsion_angles
                # in forward, the new mean is obtained using the score (see above)
                p_trajs_forward = torus.p((perturb - mean)[start:end].cpu().numpy(), g.cpu().numpy() ) 
                logit_pf[i,sigma_idx ] = torch.log(torch.tensor(p_trajs_forward)).sum() 
                # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0
                p_trajs_backward = torus.p( - perturb[start:end].cpu().numpy() , g.cpu().numpy() ) 
                logit_pb[i,sigma_idx ] = torch.log(torch.tensor(p_trajs_backward)).sum() 

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)
            trajs[sigma_idx +1 ] = torch.Tensor(data.total_perturb) - trajs[sigma_idx]

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()
        
        
        logit_pf = reduce(logit_pf, 'bs steps-> bs', 'sum' )
        logit_pb = reduce(logit_pb, 'bs steps-> bs', 'sum' )
        breakpoint()
    return conformers, trajs, logit_pf, logit_pb

def get_logpf():
    pass

def get_logpb():
    pass 


def get_likelihood(x0):
    pass