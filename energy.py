import copy
import os.path
import sys

import mdtraj
import numpy as np
import torch
from etflow.commons.featurization import MoleculeFeaturizer
from loguru import logger
from numpy.typing import ArrayLike
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openmm import LangevinIntegrator, MonteCarloBarostat, Platform, VerletIntegrator, app, unit
from openmm.unit import Quantity, angstrom, femtoseconds
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data

from gfn_free_energy.eval import angle_dist, bond_length_dist, compound_pairwise_dist, improper_dihedral_dist


def log_mean_exp(input: ArrayLike):
    input_max = np.amax(input, keepdims=True)
    out = np.log(np.mean(np.exp(input - input_max)))
    out += input_max
    return out


class Energy:
    def get_energy(self, pos):
        raise NotImplementedError

    def compute_many(self, confs):
        return torch.tensor([self.get_energy(i) for i in confs]).float()

    def featurize(self):
        # Default featurization will just rely on the smiles string
        feat = MoleculeFeaturizer()
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)
        return feat.get_data_from_smiles(self.smiles)


def print_min_rdkit_energy(smiles, nrg):
    # Minimize the molecule with rdkit to see what's the minimum energy
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol.RemoveAllConformers()
    Chem.EmbedMolecule(mol)
    Chem.MMFFOptimizeMolecule(mol)
    minconf = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    logger.info("min E =", nrg.get_energy(minconf))


class RDKitEnergy(Energy):
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.AddHs(Chem.MolFromSmiles(smiles, sanitize=False))
        print_min_rdkit_energy(smiles, self)

    def get_energy(self, pos):
        mol = self.mol
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i])
        mol.AddConformer(conf)
        ff = Chem.MMFFGetMoleculeForceField(mol, Chem.MMFFGetMoleculeProperties(mol))
        return ff.CalcEnergy()


class OpenFFEnergy(Energy):
    def __init__(self, smiles, temperature=298, mol=None):
        self.smiles = smiles
        self.mol = Molecule.from_smiles(smiles) if mol is None else mol
        self.forcefield = ForceField("openff-2.2.0.offxml")
        self.interchange = Interchange.from_smirnoff(self.forcefield, [self.mol])
        self.integrator = VerletIntegrator(1 * femtoseconds)
        self.simulation = self.interchange.to_openmm_simulation(self.integrator)
        self.kT = 8.314 * temperature / 1000  # J/K⋅mol ⋅ K / 1000 = kJ/mol
        print_min_rdkit_energy(smiles, self)

    def get_energy(self, pos):
        pos = Quantity(pos, angstrom)
        self.simulation.context.setPositions(pos)
        # unit=kilojoule/mole
        E = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
        # (kJ/mol) / (kJ/mol) = 1
        return E / self.kT  # unitless


class openMMEnergy(Energy):
    def __init__(
        self,
        mm_traj,
        pdb_file,
        temperature,
        db_path=None,
        sdf_path=None,
        meta_path=None,
        with_pbcs=False,
        forcefields=None,
        small_mol_ff=None,
        nonbonded_cutoff=1.0,
        print_stats=True,
        platform_backend="CUDA",
        timestep=1.0,
    ):
        if forcefields is None:
            forcefields = [
                "amber/ff14SB.xml",
                "amber/tip3p_standard.xml",
                "amber/tip3p_HFE_multivalent.xml",
                "amber/phosaa10.xml",
            ]
        if small_mol_ff is None:
            small_mol_ff = "openff-2.1.1"

        self.platform_backend = platform_backend
        self.nonbonded_cutoff = nonbonded_cutoff * unit.nanometer
        kb = 0.0019872041  # in kcal / mol K
        self.kbt = kb * temperature
        self.temperature = temperature
        self.forcefield = app.ForceField(*forcefields)
        if db_path is not None:
            template_generator = SMIRNOFFTemplateGenerator(cache=db_path, forcefield=small_mol_ff)
            self.forcefield.registerTemplateGenerator(template_generator.generator)

        self.mm_traj = mm_traj
        self.init_positions = mm_traj.openmm_positions(0)
        self.with_pbcs = with_pbcs
        self.timestep = timestep

        # This is possibly crude but seems to work from some visual inspection. May be worth thinking about a better way
        # to do this that's compatible with the ETFlow featurization (or rewrite our own featurized).
        # TODO: also load from .sdf file to guarantee that the atom indices are correct. Revisit once we move to water.
        # self.rdmol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
        self.rdmol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        rdDetermineBonds.DetermineConnectivity(self.rdmol)
        rdDetermineBonds.DetermineBondOrders(self.rdmol, charge=0)

        self.pdb = app.PDBFile(pdb_file)

        mol_list = [Molecule.from_file(sdf_path)] if sdf_path is not None else []
        self.openff_top = Topology.from_pdb(pdb_file, unique_molecules=mol_list)
        self.openmm_top = self.openff_top.to_openmm()
        self.init_simulation()

        if print_stats:
            self.sim.minimizeEnergy()
            min_energy = self.sim.context.getState(getEnergy=True).getPotentialEnergy()
            print(f"Minimum energy: " f"{min_energy.value_in_unit(unit.kilocalorie_per_mole)} kcal/mol")
            positions = [mm_traj.openmm_positions(i) for i in range(len(mm_traj))]
            energies = self.compute_many(positions)
            print(f"Free energy (MD): {self.free_energy(energies)}")

        self.meta_path = meta_path

    def get_statistics(self, cfg=None, traj=None, v=1):
        _is_self_traj = traj is None
        if _is_self_traj:
            path = self.meta_path + f"_stats_v{v}.pt"
            if os.path.exists(path):
                self.statistics = torch.load(path, weights_only=False)
                self.statistics["free_energy"] = self.free_energy(self.statistics["energies"])
                return self.statistics
            traj = self.mm_traj
            # Compute the test set statistics
            if cfg.md_train_set_ratio != 1.0:
                traj = copy.copy(traj)
                traj.xyz = traj.xyz[int(len(traj) * cfg.md_train_set_ratio) :]
        statistics = {
            "angle_dist": angle_dist(traj, self.openff_top),
            "bond_length_dist": bond_length_dist(traj),
            "compound_pairwise_dist": compound_pairwise_dist(traj),
            "improper_dihedral_dist": improper_dihedral_dist(traj, self.openff_top),
            "energies": self.compute_many(traj.xyz).numpy(),
        }
        statistics["free_energy"] = self.free_energy(statistics["energies"])
        if _is_self_traj:
            self.statistics = statistics
            torch.save(statistics, path)
        return statistics

    def init_simulation(self):
        platform = Platform.getPlatformByName(self.platform_backend)
        integrator = LangevinIntegrator(self.temperature * unit.kelvin, 1.0, self.timestep * unit.femtosecond)

        if self.with_pbcs:
            box_length = self.mm_traj.openmm_boxes(0)
            nonbonded_method = app.PME
            self.openmm_top.setPeriodicBoxVectors(box_length)
        else:
            nonbonded_method = app.NoCutoff

        system = self.forcefield.createSystem(
            self.openmm_top,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=self.nonbonded_cutoff,
            rigidWater=False,
            removeCMMotion=False,
        )
        self.sim = app.Simulation(self.openmm_top, system, integrator, platform)
        self.sim.context.setPositions(self.init_positions)
        self.mdtraj_top = mdtraj.Topology.from_openmm(self.sim.topology)

    def update_context(self, positions, box_lengths=None):
        """Updates the atomic postions.
        If boxvectors are provided, adjusts box size."""
        if box_lengths is not None:
            self.sim.context.setPeriodicBoxVectors(*box_lengths)
        assert positions.dtype in [np.float64, np.float32], f"openMM requires numpy f32/f64, got {positions.dtype}"
        self.sim.context.setPositions(positions)

    def get_energy(self, pos=None):
        """Get the energy of the current system state. Noprmed by kbT."""
        if pos is not None:
            self.update_context(pos)
        u = self.sim.context.getState(getEnergy=True).getPotentialEnergy()
        beta_u = u.value_in_unit(unit.kilocalorie_per_mole) / self.kbt
        return beta_u  # unitless exponent Boltzmann dist. of NVT ensemble

    def get_forces(self, pos=None):
        """Get the forces of the current system state.
        The units are in 1/A as we norm them with kbT."""
        if pos is not None:
            self.update_context(pos)
        f = self.sim.context.getState(getForces=True).getForces(asNumpy=True)
        f_normed = f.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom) / self.kbt
        return f_normed

    def compute_many(self, positions: list):
        """Returns beta*U for all provided"""
        energies = []
        for pos in positions:
            self.update_context(pos)
            energies.append(self.get_energy())
        return torch.tensor(energies).float()

    def free_energy(self, energy_dist):
        """Returns free energy estimate in the NVT ensemble of the system using
        the the normalized energy distribution (beta*U)."""
        energy_dist = np.asarray(energy_dist)
        partition = log_mean_exp(-energy_dist)
        return float(-partition / self.kbt)

    def positions_to_mdtraj(self, positions: list[list[float]]) -> mdtraj.Trajectory:
        """Transforms a list of positions to a mdtraj trajectory for
        postprocessing. Assumes that the positions are in nanometers.
        """
        return mdtraj.Trajectory(np.array(positions), self.mdtraj_top)

    def featurize(self):
        feat = MoleculeFeaturizer()
        node_attr = feat.get_atom_features_from_mol(self.rdmol, True)
        chiral_index, chiral_nbr_index, chiral_tag = feat.get_chiral_centers_from_mol(self.rdmol)
        edge_index, edge_attr = feat.get_edge_index_from_mol(self.rdmol, False)
        atomic_numbers = feat.get_atomic_numbers_from_mol(self.rdmol)

        return Data(
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )

    def perform_MD(self, traj_path, prod_time: float, equil_time: float, save_every: int = 100):
        """Perform MD simulation to verify FF is setup correctly. Production and equil time in ns. Returns mdtraj object
        of generated trajectory.
        """
        prod_steps = int(prod_time * 1e6)  # assumes 1fs time step
        equil_steps = int(equil_time * 1e6)
        self.update_context(self.init_positions)
        if self.with_pbcs:
            self.sim.system.addForce(MonteCarloBarostat(1.0 * unit.bar, self.temperature * unit.kelvin, 1.0))

        self.sim.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                save_every * 10,
                step=True,
                potentialEnergy=True,
                volume=True,
                temperature=True,
                speed=True,
                separator="\t",
            ),
        )
        self.sim.step(equil_steps)
        self.sim.reporters.append(app.DCDReporter(traj_path, save_every))
        self.sim.step(prod_steps)

        return mdtraj.load(traj_path, top=self.mdtraj_top)

    def __getstate__(self):
        # return everything except the simulation object
        state = self.__dict__.copy()
        del state["sim"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # restore the simulation object
        self.init_simulation()
