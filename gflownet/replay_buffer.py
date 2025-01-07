



class ReplayBufferClass():
    '''
    Replay Buffer that stores trajectories and log rewards. It is sorted such that the trajectories with the highest log rewards are at the beginning of the buffer.
    '''
    def __init__(self, max_size = 1000):
        self.max_size = max_size
        self.buffer_trajs = [] # list of torchgeom.data objects
        self.buffer_logrews = torch.Tensor([])
    def __len__(self):
        return len(self.buffer_trajs)
    def update(self, batch_trajs, batch_logrews):
        #print(batch_trajs, batch_logrews)
        #transose batch_trajs
        batch_trajs = [x.to_data_list() for x in batch_trajs]
        batch_trajs = list(map(list, zip(*batch_trajs)))
        # sort batch elements by logrew
        ixs = torch.argsort(batch_logrews, descending = True)
        batch_trajs = [batch_trajs[ix] for ix in ixs]
        batch_logrews = batch_logrews[ixs]
        #batch_logrews, batch_trajs = zip(*sorted(zip(batch_logrews, batch_trajs), reverse = True))
        batch_trajs = list(batch_trajs)
        batch_logrews = torch.Tensor(batch_logrews)
        # discard all elements in batch which logrew is smaller than the smallest logrew in the buffer
        if len(self.buffer_logrews) > 0:
            min_logrew = self.buffer_logrews[-1]
            ixs = torch.where(batch_logrews >= min_logrew)[0]
            batch_trajs = [batch_trajs[ix] for ix in ixs]
            batch_logrews = batch_logrews[ixs]
        # Insert the batch elements in the buffer
        self.buffer_trajs = self.buffer_trajs + batch_trajs
        self.buffer_logrews = torch.cat((self.buffer_logrews, batch_logrews))
        # get indexes of sorted logrews
        sorted_ixs = torch.argsort(self.buffer_logrews, descending = True)
        self.buffer_trajs = [self.buffer_trajs[ix] for ix in sorted_ixs]
        self.buffer_logrews = self.buffer_logrews[sorted_ixs]
        if len(self.buffer_trajs) > self.max_size:
            self.buffer_trajs = self.buffer_trajs[:self.max_size]
            self.buffer_logrews = self.buffer_logrews[:self.max_size]
        
        assert len(self.buffer_trajs) == len(self.buffer_logrews)
        
    def sample(self, n):
        if len(self.buffer_trajs)>=n:
            ixs = np.random.choice(len(self.buffer_trajs), n, replace=False)
            trajs =  [self.buffer_trajs[ix]for ix in ixs]
            trajs = list(map(list, zip(*trajs)))
            trajs = [Batch.from_data_list(x) for x in trajs]
            return trajs, self.buffer_logrews[ixs]
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
