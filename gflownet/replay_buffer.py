import torch
import  heapq
from torch_geometric.data import Data, Batch
import numpy as np

class ReplayBufferClass(): 
    '''
    Replay Buffer that stores terminal states and log rewards. It is sorted such that the states with the highest log rewards are at the beginning of the buffer.
    Args:
        smis_dataset (list): list of SMILES strings
        max_size (int): maximum size of the buffer
        max_n_local_structures (int): maximum number of local structures per SMILES
        content (dict): dictionary of dictionaries. The outer dictionary is indexed by SMILES strings, and the inner dictionary is indexed by local structure IDs. The values are lists of tuples (final_state, logrew).
    '''
    def __init__(self, smis_dataset, max_size = 1000, max_n_local_structures = 2000):
        self.smis_dataset = smis_dataset
        self.smi2smi_ix = {smi: i for i, smi in enumerate(smis_dataset)}
        self.smi_ix2smi = {i: smi for i, smi in enumerate(smis_dataset)}
        self.max_size = max_size
        self.max_n_local_structures = max_n_local_structures
        self.content = {smi: {i:[] for i in range(self.max_n_local_structures)} for smi in self.smis_dataset} # dictionary of dictionaries. # 
    
    def get_len(self, smi, local_structure_id):
        '''
        Returns the number of elements in the buffer for a given SMILES string and local structure ID.
        '''
        return len(self.content[smi][local_structure_id])
    
    def update(self, smi, local_struc_id, batch_final_states, batch_logrews):
        '''
        Add the batch final states and log rewards to the buffer. The buffer is sorted such that the states with the highest log rewards are at the beginning of the buffer.
        '''
        # sort batch elements by logrew
        local_struc_ids= [s.local_structure_id for s in batch_final_states]
        assert len(set(local_struc_ids)) == 1 , f"All elements in the batch should have the same local_structure_id. Got {local_struc_ids} local_structure_ids."
        assert local_struc_id == local_struc_ids[0], f"local_structure_id should be {local_struc_id}. Got {local_struc_ids[0]} instead."
        smis = [s.canonical_smi for s in batch_final_states]
        assert len(set(smis)) == 1 , f"All elements in the batch should have the same SMILES string. Got {smis} SMILES strings."
        assert smi == smis[0], f"SMILES string should be {smi}. Got {smis[0]} instead."
        ixs = torch.argsort(batch_logrews, descending = True)
        batch_content = [[batch_final_states[ix] , batch_logrews[ix]] for ix in ixs]
        # Use heapq merge to merge the batch content with the buffer content
        self.content[smi][local_struc_id] = list(heapq.merge(self.content[smi][local_struc_id], batch_content, key = lambda x: x[1], reverse = True))
        self.content[smi][local_struc_id] = self.content[smi][local_struc_id][:self.max_size]
         
        
    def sample(self, smi, local_structure_id, n):
        '''
        Sample n elements from the buffer for a given SMILES string and local structure ID. The elements are sampled without replacement.
        '''
        m = self.get_len(smi, local_structure_id)
        if m>=n:
            d = self.content[smi][local_structure_id]
            ixs = np.random.choice(m, n, replace=False)
            final_states, logrews =  [d[ix][0] for ix in ixs], [d[ix][1] for ix in ixs]
            return final_states, torch.Tensor(logrews) 
        else:
            raise ValueError(f"{n} samples requested, but only {m} samples available in the buffer for {smi, local_structure_id}.")
        
    
    def get_logrews(self, smi, local_structure_id):
        '''
        Returns the log rewards for a given SMILES string and local structure ID.
        '''
        m = self.get_len(smi, local_structure_id)
        if m>0:
            d = self.content[smi][local_structure_id]
            logrews = [d[ix][1] for ix in range(m)]
            return torch.Tensor(logrews)
        else:
            raise ValueError(f"No samples available in the buffer for {smi, local_structure_id}.")


    def get_positions_and_tas(self, smis):
        '''
        Returns 2 dictionaries of positions and torsion angles for a list of SMILES strings.
        Args:
            smis (list): list of SMILES strings
        Returns:
            positions: a dictionary of dictionaries. The outer dictionary is indexed by SMILES strings, and the inner dictionary is indexed by local structure IDs. The values are lists of positions.
            tas: a dictionary of dictionaries. The outer dictionary is indexed by SMILES strings, and the inner dictionary is indexed by local structure IDs. The values are lists of torsion angles.
        '''
        positions = {smi: {} for smi in smis}
        tas = {smi: {} for smi in smis}
        for smi in smis:
            local_struc_ids = self.content[smi].keys()
            for local_struc_id in local_struc_ids:
                positions[smi][local_struc_id] = []
                tas[smi][local_struc_id] = []
                d = self.content[smi][local_struc_id]
                for (conf, logrew) in d:
                    positions[smi][local_struc_id].append(conf.pos)
                    tas[smi][local_struc_id].append(conf.total_perturb)
        return positions, tas
    

def concat(traj1, traj2):
    '''
    Concatenates 2 lists of trajectories in one.
    '''
    if traj1 is None:
        return traj2
    elif traj2 is None:
        return traj1
    else:
        traj1 = [x.to_data_list() for x in traj1]
        traj1 = list(map(list, zip(*traj1))) 
        traj2 = [x.to_data_list() for x in traj2]
        traj2 = list(map(list, zip(*traj2)))
        traj = traj1 + traj2
        traj = list(map(list, zip(*traj)))
        traj = [Batch.from_data_list(x) for x in traj]
        return traj
