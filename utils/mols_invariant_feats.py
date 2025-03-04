import torch
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdMolAlign
from numpy.typing import ArrayLike, NDArray





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
