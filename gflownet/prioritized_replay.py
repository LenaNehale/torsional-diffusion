import torch

from gflownet.base_replay_buffer import ReplayBuffer
from .utils import extend_trajectories
from rdkit.Chem import rdMolAlign


def distance(src_mols: list, dst_mols: list):
    """Compute the squared distance between two tensors.
    """
    rmsds = []
    for mol0 in src_mols:
        rmsds.append([])
        for mol1 in dst_mols:
            rdMolAlign.AlignMol(mol0, mol1)
            rmsds[-1].append(fast_rmsd(mol0, mol1 , conf1=0, conf2=0))
    return rmsds


class PrioritizedReplay(ReplayBuffer):
    def __init__(
        self,
        cutoff_distance: float,
        capacity: int = 1000,
        is_conditional: bool = False,
        **kwargs,
    ):
        super().__init__(capacity, is_conditional, **kwargs)
        self.cutoff_distance = cutoff_distance

    def add(
        self,
        input: torch.Tensor,
        trajectories: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        final_state: torch.Tensor,
        logreward: torch.Tensor,
    ):
        """Add samples to the replay buffer.
        Assumes all arguments to be torch tensors and that the replay buffer is sorted in ascending order.
        Args:
            input (torch.Tensor[n_samples, max_len, input_dim]): Input to the model.
            trajectories (torch.Tensor[n_samples, traj_len, max_len, state_dim]): Trajectories.
            actions (torch.Tensor[n_samples, traj_len]): Actions.
            dones (torch.Tensor[n_samples, traj_len]): Whether trajectory is over.
            final_state (torch.Tensor[n_samples, max_len, state_dim]): Final state.
            logreward (torch.Tensor[n_samples]): Log reward.
        """
        assert all(
            isinstance(arg, torch.Tensor)
            for arg in [input, trajectories, actions, dones, final_state, logreward]
        ), "All elements must be torch tensors!"

        input = input.cpu()
        trajectories = trajectories.cpu()
        actions = actions.cpu()
        dones = dones.cpu()
        final_state = final_state.cpu()
        logreward = logreward.cpu()

        # This is the first batch.
        if len(self) == 0:
            self.storage = {
                "input": input,
                "trajectories": trajectories,
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "logreward": logreward,
            }
            # Sort elements by logreward
            ix = torch.argsort(logreward)
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ix]

        # Adding a batch and the buffer isn't full yet.
        elif len(self) < self.capacity:
            new_trajectories, new_actions, new_dones = extend_trajectories(
                self.storage["trajectories"],
                trajectories,
                self.storage["actions"],
                actions,
                self.storage["dones"],
                dones,
            )
            self.storage["input"] = torch.cat([self.storage["input"], input], dim=0)
            self.storage["trajectories"] = new_trajectories
            self.storage["actions"] = new_actions
            self.storage["dones"] = new_dones
            self.storage["final_state"] = torch.cat(
                [self.storage["final_state"], final_state], dim=0
            )
            self.storage["logreward"] = torch.cat(
                [self.storage["logreward"], logreward], dim=0
            )
            # Sort elements by logreward
            ix = torch.argsort(self.storage["logreward"])
            for key in self.storage.keys():
                self.storage[key] = self.storage[key][ix]
                # Ensures that the buffer is the correct size.
                self.storage[key] = self.storage[key][-self.capacity :]

        # Our buffer is full and we will prioritize diverse, high reward additions.
        else:
            dict_curr_batch = {
                "input": input,
                "trajectories": trajectories,
                "actions": actions,
                "dones": dones,
                "final_state": final_state,
                "logreward": logreward,
            }

            def _apply_idx(idx, d):
                for k, v in d.items():
                    d[k] = v[idx, ...]

            # Sort elements by logreward.
            idx_sorted = torch.argsort(dict_curr_batch["logreward"], descending=True)
            _apply_idx(idx_sorted, dict_curr_batch)

            # Filter all batch logrewards lower than the smallest logreward in buffer.
            idx_min_lr = dict_curr_batch["logreward"] >= self.storage["logreward"].min()
            _apply_idx(idx_min_lr, dict_curr_batch)

            # Compute all pairwise distances between the batch and the buffer.
            curr_dim = dict_curr_batch["final_state"].shape[0]
            buffer_dim = self.storage["final_state"].shape[0]
            if curr_dim > 0:
                # Distances should incorporate conditioning vector.
                if self.is_conditional:
                    batch = torch.cat(
                        [dict_curr_batch["input"], dict_curr_batch["final_state"]],
                        dim=-1,
                    )
                    buffer = torch.cat(
                        [self.storage["input"], self.storage["final_state"]],
                        dim=-1,
                    )
                else:
                    batch = dict_curr_batch["final_state"].float()
                    buffer = self.storage["final_state"].float()

                batch_lr = dict_curr_batch["logreward"].float()
                buffer_lr = self.storage["logreward"].float()

                # Filter the batch for diverse final_states with high reward.
                batch_batch_dist = torch.cdist(
                    batch.view(curr_dim, -1).unsqueeze(0),
                    batch.view(curr_dim, -1).unsqueeze(0),
                    p=1.0,
                ).squeeze(0)

                r, w = torch.triu_indices(*batch_batch_dist.shape)  # Remove upper diag.
                batch_batch_dist[r, w] = torch.finfo(batch_batch_dist.dtype).max
                batch_batch_dist = batch_batch_dist.min(-1)[0]

                # We include examples from the batch assuming they are unique within the
                # batch, and also either diverse relative to examples in the buffer, or
                # if they are similar, they must have a higher reward than their
                # comparable example currently in the buffer.
                batch_buffer_dist = torch.cdist(
                    batch.view(curr_dim, -1).unsqueeze(0),
                    buffer.view(buffer_dim, -1).unsqueeze(0),
                    p=1.0,
                ).squeeze(0)

                idx_batch = batch_batch_dist > self.cutoff_distance

                # Case where the added examples are far from the buffer (diverse).
                idx_batch_far_from_buffer = (
                    batch_buffer_dist.min(-1)[0] > self.cutoff_distance
                )

                # Case where the added examples are close to the buffer but have higher
                # reward than their comparable.
                idx_batch_lr_is_higher = batch_lr.unsqueeze(-1) > buffer_lr
                idx_batch_close_to_buffer = batch_buffer_dist < self.cutoff_distance
                idx_batch_beats_buffer = (
                    idx_batch_close_to_buffer & idx_batch_lr_is_higher
                ).any(-1)

                # We add unique elements from the batch which are either diverse or
                # give a higher reward.
                idx_to_add = idx_batch & (
                    idx_batch_far_from_buffer | idx_batch_beats_buffer
                )
                # Make sure that the new indices with high rewards are NOT duplicates
                idx_to_add = idx_to_add & (batch_buffer_dist.min(-1)[0] > 0)
                _apply_idx(idx_to_add, dict_curr_batch)

            # Concatenate everything, sort, and remove leftovers.
            for k, v in self.storage.items():
                if k not in ["trajectories", "actions", "dones"]:
                    self.storage[k] = torch.cat(
                        (self.storage[k], dict_curr_batch[k]), dim=0
                    )
            new_trajectories, new_actions, new_dones = extend_trajectories(
                self.storage["trajectories"],
                dict_curr_batch["trajectories"],
                self.storage["actions"],
                dict_curr_batch["actions"],
                self.storage["dones"],
                dict_curr_batch["dones"],
            )
            self.storage["trajectories"] = new_trajectories
            self.storage["actions"] = new_actions
            self.storage["dones"] = new_dones

            idx_sorted = torch.argsort(self.storage["logreward"], descending=False)
            _apply_idx(idx_sorted, self.storage)

            for k, v in self.storage.items():
                self.storage[k] = self.storage[k][-self.capacity :]  # Keep largest.

    def sample(self, num_samples: int):
        """Sample from the replay buffer according to the logreward and without replacement.
        Args:
            num_samples (int): Number of samples to draw.
        Returns:
            dict: Dictionary containing the samples.
        """
        probs = torch.softmax(self.storage["logreward"], dim=-1)
        if probs.shape[-1] < num_samples:
            raise ValueError(
                f"Number of samples to draw is larger than the buffer size. Decrease gfn.num_samples to less than {probs.shape[-1]}"
            )
        ixs = torch.multinomial(probs, num_samples, replacement=False)
        samples = {}

        for key in self.storage.keys():
            samples[key] = self.storage[key][ixs]
        return samples