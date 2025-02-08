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
        self.content = {smi_ix: [] for smi_ix in self.smi_ix2smi.keys()}  # shape of self.content[smi_ix]: (size, 2)
    def get_len(self, smi):
        smi_ix = self.smi2smi_ix[smi]
        return len(self.content[smi_ix])
    def update(self, smi, batch_trajs, batch_logrews):
        smi_ix = self.smi2smi_ix[smi]
        #transpose batch_trajs
        batch_trajs = [x.to_data_list() for x in batch_trajs]
        batch_trajs = list(map(list, zip(*batch_trajs)))
        # sort batch elements by logrew
        ixs = torch.argsort(batch_logrews, descending = True)
        batch_content = [[batch_trajs[ix] , batch_logrews[ix]] for ix in ixs]
        # Use heapq merge to merge the batch content with the buffer content
        self.content[smi_ix] = list(heapq.merge(self.content[smi_ix], batch_content, key = lambda x: x[1], reverse = True))
        self.content[smi_ix] = self.content[smi_ix][:self.max_size]
        
        
    def sample(self,smi, n):
        smi_ix = self.smi2smi_ix[smi]
        if len(self.content[smi_ix])>=n:
            ixs = np.random.choice(len(self.content[smi_ix]), n, replace=False)
            trajs, logrews =  [self.content[smi_ix][ix][0] for ix in ixs], [self.content[smi_ix][ix][1] for ix in ixs]
            trajs = list(map(list, zip(*trajs)))
            trajs = [Batch.from_data_list(x) for x in trajs]
            return trajs, torch.Tensor(logrews)
        else:
            raise ValueError(f"{n} samples requested, but only {len(self.buffer_trajs)} samples available in the buffer")

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
