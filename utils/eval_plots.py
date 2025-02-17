import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdMolAlign
from numpy.typing import ArrayLike, NDArray


def make_logrew_histograms(logrews_random, logrews_gen, logrews_gt, exp_path, label, range = 2):
    '''
    Plots histograms of logrews for generated, ground truth and random conformers, for the same set of smis.
    Args:
        - logrews_gen: dictionary where keys are smiles and values are tensors of logrews for generated conformers
        - logrews_gt: dictionary where keys are smiles and values are tensors of logrews for ground truth conformers
        - logrews_random: dictionary where keys are smiles and values are tensors of logrews for random conformers
    Returns:
        A figure of shape (n_smis // 5, 5) where each subplot coresponds to a different smile and shows the histograms of logrews for generated, ground truth and random conformers.

    '''
    
    assert logrews_gen.keys() == logrews_random.keys() == logrews_gt.keys()
    smis = logrews_gen.keys()
    n_smis = len(smis)
    n_subplots = 5 # number of subplots per row
    fig, axes = plt.subplots(max(n_smis // n_subplots, 1), n_subplots, figsize=(10, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    for smi_idx, smi in enumerate(smis):

        a , b = logrews_gt[smi].min().item() ,  logrews_gt[smi].max().item()
        range_min = a - range * (b - a)
        range_max = b + range * (b - a)
        n_bins = 100
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(  logrews_random[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='r', label = 'random', density=True)
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(logrews_gen[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='b', label = 'generated', density=True)
        axes[ smi_idx //n_subplots , smi_idx % n_subplots ].hist(logrews_gt[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='g', label = 'ground truth', density=True)
        axes[smi_idx // n_subplots, smi_idx % n_subplots].set_title(smi)
        
        
    fig.suptitle('logrews distribution') 
    fig.legend(['random', 'generated','ground truth'], loc='upper right')
    plt.tight_layout() 
    plt.savefig(f"{exp_path}_{label}.png")
    plt.close(fig)



def get_bond_lengths(mol):
    '''
    params: mol: rdkit molecule object
    returns a tensor with the bond lengths of the molecule
    '''
    assert mol.GetNumConformers() == 1    
    bond_lengths = []
    for bond in mol.GetBonds():
        bond_lengths.append(rdMolTransforms.GetBondLength(mol.GetConformer(),bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx() ) )
    return torch.Tensor(bond_lengths)

def get_bond_angles(mol):
    '''
    params: mol: rdkit molecule object
    returns a tensor with the bond angles of the molecule
    '''
    assert mol.GetNumConformers() == 1
    bond_angles = []
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        for neighbor in atom2.GetNeighbors():
            if neighbor.GetIdx() != atom1.GetIdx():
                angle = rdMolTransforms.GetAngleRad(mol.GetConformer(),   
                                                    atom1.GetIdx(), 
                                                    atom2.GetIdx(), 
                                                    neighbor.GetIdx())
                bond_angles.append(angle)
    return torch.Tensor(bond_angles)

def get_torsion_angles(mol):
    '''
    params: mol: rdkit molecule object
    returns a tensor with the torsion angles of the molecule
    '''
    assert mol.GetNumConformers() == 1
    torsion_angles = []
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        # Get neighboring atoms
        for neighbor1 in atom2.GetNeighbors():
            if neighbor1.GetIdx() != atom1.GetIdx():
                # Calculate the angle
                for neighbor2 in neighbor1.GetNeighbors():
                    if neighbor2.GetIdx() != atom2.GetIdx() and neighbor2.GetIdx() != atom1.GetIdx():
                        torsion = rdMolTransforms.GetDihedralRad(mol.GetConformer(), 
                                                                 atom1.GetIdx(), 
                                                                 atom2.GetIdx(), 
                                                                 neighbor1.GetIdx(), 
                                                                 neighbor2.GetIdx())
                        torsion_angles.append(torsion)
    return torch.Tensor(torsion_angles)

def get_means_and_stds_internal_coords(mols):
    ''''
    args:
        - mols: list of rdkit molecule objects
    returns:
        - bond_lengths_mean: tensor with the mean bond lengths
        - bond_angles_mean: tensor with the mean bond angles
        - torsion_angles_mean: tensor with the mean torsion angles
        - bond_lengths_std: tensor with the std of bond lengths
        - bond_angles_std: tensor with the std of bond angles
        - torsion_angles_std: tensor with the std of torsion angles
    
    '''
    all_bond_lengths = [get_bond_lengths(mol) for mol in mols]
    all_bond_angles = [get_bond_angles(mol) for mol in mols]
    all_torsion_angles = [get_torsion_angles(mol) for mol in mols]
    bond_lengths_mean, bond_lengths_std =  torch.stack(all_bond_lengths).mean(dim=0) , torch.stack(all_bond_lengths).std(dim=0)
    bond_angles_mean  , bond_angles_std =   torch.stack(all_bond_angles).mean(dim=0) , torch.stack(all_bond_angles).std(dim=0)
    torsion_angles_mean , torsion_angles_std =  torch.stack(all_torsion_angles).mean(dim=0)  , torch.stack(all_torsion_angles).std(dim=0)
    return bond_lengths_mean, bond_angles_mean, torsion_angles_mean, bond_lengths_std, bond_angles_std, torsion_angles_std


def make_localstructures_hist(mols_rand, mols_gen, mols_md, exp_path, label):
    '''
    Plots histograms of bond lengths, bond angles and torsion angles for generated, ground truth and random molecules, for the same set of smis.
    Args:
    '''

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot bond lengths histogram

    bond_lengths_rand = get_bond_lengths(mols_rand)
    bond_lengths_gen = get_bond_lengths(mols_gen)
    bond_lengths_md = get_bond_lengths(mols_md)
    axs[0].hist(bond_lengths_rand, bins=100, color='blue', alpha=0.5, label = 'rand')
    axs[0].hist(bond_lengths_gen, bins=100, color='red', alpha=0.5, label = 'gen')
    axs[0].hist(bond_lengths_md, bins=100, color='green', alpha=0.5, label = 'md')
    axs[0].set_title('Bond Lengths')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    
    
    # Plot bond angles histogram
    bond_angles_rand = get_bond_angles(mols_rand)
    bond_angles_gen = get_bond_angles(mols_gen)
    bond_angles_md = get_bond_angles(mols_md)
    axs[1].hist(bond_angles_rand, bins=100, color='blue', alpha=0.5, label = 'rand')
    axs[1].hist(bond_angles_gen, bins=100, color='red', alpha=0.5, label = 'gen')
    axs[1].hist(bond_angles_md, bins=100, color='green', alpha=0.5, label = 'md')
    axs[1].set_title('Bond Angles')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    
    # Plot torsion angles histogram
    torsion_angles_rand = get_torsion_angles(mols_rand)
    torsion_angles_gen = get_torsion_angles(mols_gen)
    torsion_angles_md = get_torsion_angles(mols_md)
    axs[2].hist(torsion_angles_rand, bins=100, color='blue', alpha=0.5, label = 'rand')
    axs[2].hist(torsion_angles_gen, bins=100, color='red', alpha=0.5, label = 'gen')
    axs[2].hist(torsion_angles_md, bins=100, color='green', alpha=0.5, label = 'md')
    axs[2].set_title('Torsion Angles')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{exp_path}_{label}.png")
    plt.close(fig)










'''
#### Eval plots from Emmanuel's code 

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
from openff.toolkit import Topology


def bond_length_dist(traj: mdtraj.Trajectory, periodic=False) -> ArrayLike:
    """Distribution of distances between all bonded atoms across all
    trajectories.
    """
    bond_pairs = []
    for bond in traj.topology.bonds:
        bond_pairs.append([bond.atom1.index, bond.atom2.index])
    bond_distances = mdtraj.compute_distances(traj, bond_pairs, periodic=periodic)
    return np.reshape(bond_distances, -1)


def compound_pairwise_dist(traj: mdtraj.Trajectory, periodic=False) -> ArrayLike:
    """Distribution of distances between all atom pairs across all
    trajectories. This gives a global view of the molecule trajectory.
    """
    all_atoms = np.arange(traj.topology.n_atoms)
    all_pairs = traj.topology.select_pairs(all_atoms, all_atoms)
    distances = mdtraj.compute_distances(traj, all_pairs, periodic=periodic)
    return np.reshape(distances, -1)


def angle_dist(traj: mdtraj.Trajectory, top: Topology, periodic=False):
    angles = []
    for angle in top.angles:
        angles.append([angle[0].molecule_atom_index, angle[1].molecule_atom_index, angle[2].molecule_atom_index])
    angle_vals = mdtraj.compute_angles(traj, angles, periodic=periodic)
    return np.reshape(angle_vals, -1)


def improper_dihedral_dist(traj: mdtraj.Trajectory, top: Topology, periodic=False):
    impropers = []
    for improper in top.impropers:
        impropers.append(
            [
                improper[0].molecule_atom_index,
                improper[1].molecule_atom_index,
                improper[2].molecule_atom_index,
                improper[3].molecule_atom_index,
            ]
        )
    if len(impropers) == 0:
        return np.array([])
    dihedral_vals = mdtraj.compute_dihedrals(traj, impropers, periodic=periodic)
    return np.reshape(dihedral_vals, -1)


def alanine_dihedrals(traj: mdtraj.Trajectory):
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    # forward compatibility to solvated alanine:
    traj.restrict_atoms(traj.topology.select("protein"))
    angles = mdtraj.geometry.compute_dihedrals(traj, [phi_indices, psi_indices])
    return angles * 180 / np.pi


def _dihedral_map():
    mymap = np.array(
        [
            [0.9, 0.9, 0.9],
            [0.85, 0.85, 0.85],
            [0.8, 0.8, 0.8],
            [0.75, 0.75, 0.75],
            [0.7, 0.7, 0.7],
            [0.65, 0.65, 0.65],
            [0.6, 0.6, 0.6],
            [0.55, 0.55, 0.55],
            [0.5, 0.5, 0.5],
            [0.45, 0.45, 0.45],
            [0.4, 0.4, 0.4],
            [0.35, 0.35, 0.35],
            [0.3, 0.3, 0.3],
            [0.25, 0.25, 0.25],
            [0.2, 0.2, 0.2],
            [0.15, 0.15, 0.15],
            [0.1, 0.1, 0.1],
            [0.05, 0.05, 0.05],
            [0, 0, 0],
        ]
    )
    newcmp = clr.ListedColormap(mymap)
    return newcmp


def _annotate_alanine_histrogram(axis=None):
    if axis is None:
        target = plt
        target.xlabel(r"$\phi$ in $\mathrm{deg}$")
        target.ylabel(r"$\psi$ in $\mathrm{deg}$")
        target.xlim([-180, 180])
        target.ylim([-180, 180])
    else:
        target = axis
        target.set_xlabel(r"$\phi$ in $\mathrm{deg}$")
        target.set_ylabel(r"$\psi$ in $\mathrm{deg}$")
        target.set_xlim([-180, 180])
        target.set_ylim([-180, 180])

    target.text(-155, 90, "$C5$", fontsize=18)
    target.text(-70, 90, "$C7eq$", fontsize=18)
    target.text(145, 90, "$C5$", fontsize=18)
    target.text(-155, -150, "$C5$", fontsize=18)
    target.text(-70, -150, "$C7eq$", fontsize=18)
    target.text(145, -150, "$C5$", fontsize=18)
    target.text(-170, -90, r'$\alpha_R$"', fontsize=18)
    target.text(140, -90, r'$\alpha_R$"', fontsize=18)
    target.text(-70, -90, r"$\alpha_R$", fontsize=18)
    target.text(70, 0, r"$\alpha_L$", fontsize=18)
    target.plot([-180, 13], [74, 74], "k", linewidth=0.5)
    target.plot([128, 180], [74, 74], "k", linewidth=0.5)
    target.plot([13, 13], [-180, 180], "k", linewidth=0.5)
    target.plot([128, 128], [-180, 180], "k", linewidth=0.5)
    target.plot([-180, 13], [-125, -125], "k", linewidth=0.5)
    target.plot([128, 180], [-125, -125], "k", linewidth=0.5)
    target.plot([-134, -134], [-125, 74], "k", linewidth=0.5)
    target.plot([-110, -110], [-180, -125], "k", linewidth=0.5)
    target.plot([-110, -110], [74, 180], "k", linewidth=0.5)

    if axis is not None:
        return axis


def plot_histogram_density(angles: np.ndarray, ax=None):
    """Plot 2D histogram for alanine from the dihedral angles."""
    newcmp = _dihedral_map()
    h, x_edges, y_edges = np.histogram2d(angles[:, 0], angles[:, 1], bins=60, density=True)

    h_masked = np.where(h == 0, np.nan, h)
    x, y = np.meshgrid(x_edges, y_edges)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        print("Creating new figure")

    ax.pcolormesh(x, y, h_masked.T, cmap=newcmp, vmax=0.000225)  # vmin=1, vmax=5.25
    # axs.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    # axs.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    mesh = ax.pcolormesh(x, y, h_masked.T, cmap=newcmp, vmax=0.000225)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore
    cbar.formatter.set_useMathText(True)  # type: ignore
    cbar.set_label("Density")
    ax = _annotate_alanine_histrogram(axis=ax)
    # plt.savefig(f'output/postprocessing/{folder}histogram_density_{saveas}.png')
    if ax is None:
        print("Showing figure")
        fig.show()
    else:
        return ax

'''
def kl_div_histogram(px: NDArray, qx: NDArray, n: int = 32) -> float:
    """Compute the Kullback-Leibler divergence between two histograms with smoothing for non-overlaps."""
    if len(px) == 0 or len(qx) == 0:
        if len(px) == 0 and len(qx) == 0:
            return 0.0  # Both are empty
        raise ValueError("Both histograms should be non-empty or both should be empty")
    xmin, xmax = np.min(qx), np.max(qx)  # We only consider q because it is the target, values outside are not relevant
    xmin, xmax = np.nextafter(xmin, xmin - 1), np.nextafter(xmax, xmax + 1)  # make sure we exclude the edges
    bins = np.linspace(xmin, xmax, n + 1)
    ph = np.bincount(np.digitize(px, bins, right=True), minlength=n + 2) + 1  # + 1 to avoid / 0
    qh = np.bincount(np.digitize(qx, bins, right=True), minlength=n + 2) + 1  # + 2 for the under/overflow bins
    p = ph / ph.sum()
    q = qh / qh.sum()
    kl = p * (np.log(p) - np.log(q))
    return kl.sum()
