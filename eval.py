import matplotlib.colors as clr
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
from numpy.typing import ArrayLike, NDArray
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
