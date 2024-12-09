from abc import ABC, abstractmethod

import torch


class ReplayBuffer(ABC):
    def __init__(self, capacity: int = 1000, is_conditional: bool = True, **kwargs):
        self.capacity = capacity
        self.is_conditional = is_conditional
        self.storage = {
            "smi": [],  
            "terminal_states": [],
            "positions": torch.Tensor(),
            "torsion_updates": torch.Tensor(),
            "logreward": torch.Tensor(),
        }

    def __len__(self):
        return len(self.storage["positions"])

    def clear(self):
        """Clear the replay buffer."""
        self.storage = {
            "smi": [],
            "terminal_states": [],
            "positions": torch.Tensor(),
            "torsion_updates": torch.Tensor(),
            "logreward": torch.Tensor(),
        }

    @abstractmethod
    def sample(self, num_samples: int):
        NotImplementedError

    @abstractmethod
    def add(
        self,
        smi: list, 
        terminal_states: list,
        positions: torch.Tensor,
        torsion_updates: torch.Tensor,
        logreward: torch.Tensor,
    ):
        """Add samples to the replay buffer. Assumes all arguments to be torch tensors.
        Args:
            smi (list): List of SMILES strings.
            terminal_states (list): List of terminal states.
            positions (torch.Tensor): Tensor of positions.
            torsion_updates (torch.Tensor): Tensor of torsion updates.
            logreward (torch.Tensor): Tensor of log rewards.
        """
        NotImplementedError