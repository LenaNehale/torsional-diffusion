from diffusion.sampling import *



def sample_traj(conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,
           ode=False, likelihood=None, pdb=None):
    
    #In this function, we remove the diversity terms that they were using in sampling.py

    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    
    for batch_idx, data in enumerate(loader):
        bs = data.num_graphs
        n_torsion_angles = len(data.total_perturb[0]) 
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))
        trajs = torch.zeros(len(sigma_schedule) + 1, bs, n_torsion_angles) # trajectories of states, not actions

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()
    
            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            
            mean, std = g ** 2 * eps * score, g * np.sqrt(eps)
            
            # compute the forward and backward (in gflownet language) transitions logprobs
            for i in range(bs):
                start, end = i*n_torsion_angles, (i+1)*n_torsion_angles
                # in forward, the new mean is obtained using the score (see above)
                p_trajs_forward = torus.p((perturb - mean)[start:end].cpu().numpy(), std.cpu().numpy() ) 
                logit_pf[i,sigma_idx ] = torch.log(torch.tensor(p_trajs_forward)).sum() 
                # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0
                p_trajs_backward = torus.p( - perturb[start:end].cpu().numpy() , std.cpu().numpy() ) 
                logit_pb[i,sigma_idx ] = torch.log(torch.tensor(p_trajs_backward)).sum() 

            conf_dataset.apply_torsion_and_update_pos(data, perturb.numpy())
            data_gpu.pos = data.pos.to(device)
            trajs[sigma_idx +1 ] = torch.Tensor(data.total_perturb) # if trajectories of actions, set trajs[sigma_idx +1 ] = torch.Tensor(data.total_perturb)- trajs[sigma_idx]

            if pdb:
                for conf_idx in range(data.num_graphs):
                    coords = data.pos[data.ptr[conf_idx]:data.ptr[conf_idx + 1]]
                    num_frames = still_frames if sigma_idx == steps - 1 else 1
                    pdb.add(coords, part=batch_size * batch_idx + conf_idx, order=sigma_idx + 2, repeat=num_frames)

            for i, d in enumerate(dlogp):
                conformers[data.idx[i]].dlogp = d.item()
        
        
        logit_pf = reduce(logit_pf, 'bs steps-> bs', 'sum' )
        logit_pb = reduce(logit_pb, 'bs steps-> bs', 'sum' )
    return conformers, trajs, logit_pf, logit_pb

def get_logpf_logpb(trajs, conformers, model, sigma_max=np.pi, sigma_min=0.01 * np.pi, steps=20, batch_size=32,):
        
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)

    sigma_schedule = 10 ** np.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    
    for batch_idx, data in enumerate(loader):
        bs = data.num_graphs
        n_torsion_angles = len(data.total_perturb[0]) 
        logit_pf = torch.zeros(bs, len(sigma_schedule))
        logit_pb = torch.zeros(bs, len(sigma_schedule))

        dlogp = torch.zeros(data.num_graphs)
        data_gpu = copy.deepcopy(data).to(device)
        for sigma_idx, sigma in enumerate(sigma_schedule):
            data_gpu.node_sigma = sigma * torch.ones(data.num_nodes, device=device)
            with torch.no_grad():
                data_gpu = model(data_gpu)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
            z = torch.normal(mean=0, std=1, size=data_gpu.edge_pred.shape)
            score = data_gpu.edge_pred.cpu()
    
            if ode:
                perturb = 0.5 * g ** 2 * eps * score
                if likelihood:
                    div = divergence(model, data, data_gpu, method=likelihood)
                    dlogp += -0.5 * g ** 2 * eps * div
            else:
                perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            
            mean, std = g ** 2 * eps * score, g * np.sqrt(eps)
            
            # compute the forward and backward (in gflownet language) transitions logprobs
            for i in range(bs):
                start, end = i*n_torsion_angles, (i+1)*n_torsion_angles
                # in forward, the new mean is obtained using the score (see above)
                p_trajs_forward = torus.p((perturb - mean)[start:end].cpu().numpy(), std.cpu().numpy() ) 
                logit_pf[i,sigma_idx ] = torch.log(torch.tensor(p_trajs_forward)).sum() 
                # in backward, since we are in variance-exploding, f(t)=0. So the mean of the backward kernels is 0
                p_trajs_backward = torus.p( - perturb[start:end].cpu().numpy() , std.cpu().numpy() ) 
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
    return conformers, trajs, logit_pf, logit_pb
    

def get_logpb():
    pass 


def get_likelihood(x0):
    pass