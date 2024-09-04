import torchani
import torch
import numpy as np

#TODO add the GFN2XTB oracle

class Energy(torch.nn.Module):
    def __init__(
        self,
        oracle="torchani",
        T=1.0,
    ):

        self.T = T
        self.oracle = oracle
        super().__init__()
        if self.oracle == "torchani":
            self.model = torchani.models.ANI2x(periodic_table_index=True)
        else:
            raise NotImplementedError("Only torchani is supported for now")

    def logrew(self, atom_ids, pos) -> torch.Tensor:
        logrews = -self.model((atom_ids, pos)).energies / self.T
        return logrews

    def check_requires_grad(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                raise ValueError(f"Parameter {name} does not require grad!")
        return True

    def score(self, atom_ids, pos) -> torch.Tensor:
        energies = self(atom_ids, pos)
        # self.check_requires_grad()
        grads = torch.autograd.grad(torch.mean(energies), pos, retain_graph=True)[0]
        torch.cuda.empty_cache()
        return grads
