import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem




### From Sasha's code

def remove_duplicate_tas(tas_list):
    """
    Remove duplicate torsion angles from a list of torsion angle tuples.

    Args
    ----
    tas_list : list of tuples
        A list of torsion angle tuples, each containing four values:
        (atom1, atom2, atom3, atom4).

    Returns
    -------
    list of tuples: A list of unique torsion angle tuples, where duplicate angles have been removed.
    """
    tas = np.array(tas_list)
    clean_tas = []
    considered = []
    for row in tas:
        begin = row[1]
        end = row[2]
        if not (begin, end) in considered and not (end, begin) in considered:
            if begin > end:
                begin, end = end, begin
            duplicates = tas[np.logical_and(tas[:, 1] == begin, tas[:, 2] == end)]
            duplicates_reversed = tas[
                np.logical_and(tas[:, 2] == begin, tas[:, 1] == end)
            ]
            duplicates_reversed = np.flip(duplicates_reversed, axis=1)
            duplicates = np.concatenate([duplicates, duplicates_reversed], axis=0)
            assert duplicates.shape[-1] == 4
            duplicates = duplicates[
                np.where(duplicates[:, 0] == duplicates[:, 0].min())[0]
            ]
            clean_tas.append(duplicates[np.argmin(duplicates[:, 3])].tolist())
            considered.append((begin, end))
    return clean_tas

def is_hydrogen_ta(mol, ta):
    """
    Simple check whether the given torsion angle is 'hydrogen torsion angle', i.e.
    it effectively influences only positions of some hydrogens in the molecule
    """

    def is_connected_to_all_hydrogens(mol, atom_id, except_id):
        atom = mol.GetAtomWithIdx(atom_id)
        neigh_numbers = []
        for n in atom.GetNeighbors():
            if n.GetIdx() != except_id:
                neigh_numbers.append(n.GetAtomicNum())
        neigh_numbers = np.array(neigh_numbers)
        return np.all(neigh_numbers == 1)

    first = is_connected_to_all_hydrogens(mol, ta[1], ta[2])
    second = is_connected_to_all_hydrogens(mol, ta[2], ta[1])
    return first or second

def get_rotatable_ta_list(mol, rotate_hydrogen_tas=True):
    """
    Find unique rotatable torsion angles of a molecule. Torsion angle is given by a tuple of adjacent atoms'
    indices (atom1, atom2, atom3, atom4), where:
    - atom2 < atom3,
    - atom1 and atom4 are minimal among neighbours of atom2 and atom3 correspondingly.

    Torsion angle is considered rotatable if:
    - the bond (atom2, atom3) is a single bond,
    - none of atom2 and atom3 are adjacent to a triple bond (as the bonds near the triple bonds must be fixed),
    - atom2 and atom3 are not in the same ring.

    Args
    ----
    mol : RDKit Mol object
        A molecule for which torsion angles need to be detected.

    rotate_hydrogen_tas : bool
        If True, the torsion angles with hydrogen atoms will be considered rotatable. Default is True.

    Returns
    -------
    list of tuples: A list of unique torsion angle tuples corresponding to rotatable bonds in the molecule.
    """
    # TODO: implement a way to drop hydrogen torsion angles
    torsion_pattern = "[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]"
    substructures = Chem.MolFromSmarts(torsion_pattern)
    torsion_angles = remove_duplicate_tas(list(mol.GetSubstructMatches(substructures)))
    if not rotate_hydrogen_tas:
        torsion_angles = [ta for ta in torsion_angles if not is_hydrogen_ta(mol, ta)]
    return torsion_angles



def get_transformation_mask_correct(mol, pyg_data):  


    rotatable_torsion_angles = get_rotatable_ta_list(mol)
    if len(rotatable_torsion_angles) == 0:
        print('Mol has no rotatable bonds!')
        return None, None
    rotatable_edges = np.array(rotatable_torsion_angles)[:, 1:3]
    rotatable_edges = np.concatenate((rotatable_edges, rotatable_edges[:,::-1] )) # Take edges in both directions
    edges = pyg_data.edge_index.T.numpy()
    rotatable_edges_ixs = []
    for e0 in rotatable_edges:
        for i, e1 in enumerate(edges):
            if np.all(e0 == e1):
                rotatable_edges_ixs.append(i)
        

    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    for i in range(0, edges.shape[0]):
        if i in rotatable_edges_ixs:
            G2 = G.to_undirected()
            G2.remove_edge(*edges[i])
            assert not nx.is_connected(G2)
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if edges[i, 0] in l:
                to_rotate.append(l) # only add the atoms if it is the minimal connected component to rotate
            else:
                to_rotate.append([])
        else:   
            to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def get_transformation_mask(mol, pyg_data):  
    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2): 
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate 


def get_distance_matrix(pyg_data, mask_edges, mask_rotate):
    G = to_networkx(pyg_data, to_undirected=False)
    N = G.number_of_nodes()
    edge_distances = []
    for i, e in enumerate(pyg_data.edge_index.T.numpy()[mask_edges]):
        v = e[1]
        d = nx.shortest_path_length(G, source=v)
        d = np.asarray([d[j] for j in range(N)])
        d = d - 1 + mask_rotate[i]
        edge_distances.append(d)

    edge_distances = np.asarray(edge_distances)
    return edge_distances





def modify_conformer(pos, edge_index, mask_rotate, torsion_updates_input, as_numpy=False):
    '''
    Modifies the conformer's 3D coordinates based on the torsion updates.
    Args:
        pos: (N, 3) tensor of 3D coordinates
        edge_index: (E, 2) tensor of edge indices
        mask_rotate: (E, N) tensor of boolean masks indicating which atoms are rotated by each edge
        torsion_updates_input: (E,) tensor of torsion updates
        as_numpy: whether to return the output as a numpy array
    Returns:
        pos: (N, 3) tensor of modified 3D coordinates
    '''
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    if type(torsion_updates_input) != np.ndarray: 
        torsion_updates = copy.deepcopy(torsion_updates_input.detach()).cpu().numpy()
    else:
        torsion_updates = torsion_updates_input
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = int(e[0]), int(e[1])

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u] 
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer(data.pos, 
            data.edge_index.T[data.edge_mask], 
            data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0 
    idx_edges = 0 
    torsion_update_list = [] 
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new


def bdot(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)


def get_torsion_angles(dihedral, batch_pos, batch_size):
    batch_pos = batch_pos.reshape(batch_size, -1, 3)

    c, a, b, d = dihedral[:, 0], dihedral[:, 1], dihedral[:, 2], dihedral[:, 3]
    c_project_ab = batch_pos[:,a] + bdot(batch_pos[:,c] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    d_project_ab = batch_pos[:,a] + bdot(batch_pos[:,d] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) / bdot(batch_pos[:,b] - batch_pos[:,a], batch_pos[:,b] - batch_pos[:,a]) * (batch_pos[:,b] - batch_pos[:,a])
    dshifted = batch_pos[:,d] - d_project_ab + c_project_ab
    cos = bdot(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab) / (
                torch.norm(dshifted - c_project_ab, dim=-1, keepdim=True) * torch.norm(batch_pos[:,c] - c_project_ab, dim=-1,
                                                                                       keepdim=True))
    cos = torch.clamp(cos, -1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cos)
    sign = torch.sign(bdot(torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab), batch_pos[:,b] - batch_pos[:,a]))
    torsion_angles = (angle * sign).squeeze(-1)
    return torsion_angles
