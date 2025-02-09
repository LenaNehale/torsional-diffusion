import matplotlib.pyplot as plt
import numpy as np
import torch




def make_logrew_histograms(logrews_random, logrews_gen, logrews_gt):
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
    fig, axes = plt.subplots(n_smis // n_subplots, n_subplots, figsize=(10, 5))

    for smi_idx, smi in enumerate(smis):

        a , b = logrews_gt[smi].min().item() ,  logrews_gt[smi].max().item()
        range_min = a - 10 * (b - a)
        range_max = b + 10 * (b - a)
        n_bins = 100
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(  logrews_random[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='r', label = 'random', density=True)
        axes[ smi_idx // n_subplots , smi_idx % n_subplots ].hist(logrews_gen[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='b', label = 'generated', density=True)
        axes[ smi_idx //5 , smi_idx % 5 ].hist(logrews_gt[smi], np.linspace(range_min, range_max, n_bins) , alpha=0.5, color='g', label = 'ground truth', density=True)
        
        
    fig.suptitle('logrews distribution') 
    fig.legend(['random', 'generated','ground truth'], loc='upper right')
    plt.tight_layout()
    plt.show()



