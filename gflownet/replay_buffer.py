import torch
import  heapq
from torch_geometric.data import Data, Batch
import numpy as np

class ReplayBufferClass():
    '''
    Replay Buffer that stores trajectories and log rewards. It is sorted such that the trajectories with the highest log rewards are at the beginning of the buffer.
    '''
    def __init__(self, smis_dataset, max_size = 1000):
        self.smis_dataset = smis_dataset
        self.smi2smi_ix = {smi: i for i, smi in enumerate(smis_dataset)}
        self.smi_ix2smi = {i: smi for i, smi in enumerate(smis_dataset)}
        self.max_size = max_size
        self.content = {smi: [] for smi in self.smis_dataset}  # shape of self.content[smi_ix]: (max_size, 2). First element is final state, second element is log reward 
    def get_len(self, smi):
        return len(self.content[smi])
    def update(self, smi, batch_final_states, batch_logrews):
        # sort batch elements by logrew
        ixs = torch.argsort(batch_logrews, descending = True)
        batch_content = [[batch_final_states[ix] , batch_logrews[ix]] for ix in ixs]
        # Use heapq merge to merge the batch content with the buffer content
        self.content[smi] = list(heapq.merge(self.content[smi], batch_content, key = lambda x: x[1], reverse = True))
        self.content[smi] = self.content[smi][:self.max_size]
        
        
    def sample(self,smi, n):
        if len(self.content[smi])>=n:
            ixs = np.random.choice(len(self.content[smi]), n, replace=False)
            final_states, logrews =  [self.content[smi][ix][0] for ix in ixs], [self.content[smi][ix][1] for ix in ixs]
            return final_states, torch.Tensor(logrews) 
        else:
            raise ValueError(f"{n} samples requested, but only {len(self.buffer_trajs)} samples available in the buffer")
        
    def get_positions(self, smis):
        positions = {smi: [] for smi in smis}
        for smi in smis:
            for i in range(self.get_len(smi)):
                conf = self.content[smi][i][0]
                positions[smi].append(conf.pos)
        return positions

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
