"""
Author: Juhani Kivimäki (juhani.kivimaki.at.helsinki.fi)
Disclaimer: https://github.com/JuhaniK/AC_trials/blob/main/Disclaimer

This file contains helper functions for other scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Global variables
RANDOM_SEED = 13
BINS = 20
rng = np.random.default_rng(seed=RANDOM_SEED)


def setstyle():
    """This is used to control the plotting settings"""
    plt.style.use("seaborn-v0_8")

    plt.rc("figure", figsize=(10, 10))
    plt.rc("image", cmap='coolwarm')

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12) 
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)


def calculate_ese(true_labels, pos_confidences, num_bins=BINS, scheme='equal'):
    """Calculates the calibration error in either in the form of ECE or AdaECE
    Parameters:
        true_labels (NumPy Array): The (binary) labels for all the datapoints used in the calculations
        pos_confidences (NumPy Array): The confidences for the positive class for all the datapoints used in the calculations
        num_bins (int): The number of bins used 
        scheme (srt): Either 'equal' for equiwidth binning (ECE) or 'dynamic' for adaptive binning (AdaECE)
    Returns:
        ece (float): The calibration error     
    """
    # Perform binning
    if scheme == 'equal':
        bins = np.linspace(0.0, 1.0, num_bins + 1)
    elif scheme == 'dynamic':
        borders = np.linspace(0.0, 1.0, num_bins + 1)
        bins = np.array([np.quantile(pos_confidences, q) for q in borders])
        if np.isnan(bins).any():
            print("Revert to equiwidth binning")
            bins = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        raise NameError(f"Binning scheme '{scheme}' is not recognized.")
    bin_indices = np.digitize(pos_confidences, bins, right=True)
    
    # Bin indices should range from 1 to num_bins
    zero_mask = bin_indices[bin_indices == 0]
    zero_amount = zero_mask.sum()
    if zero_amount > 0:
        print(f"{zero_amount} zero indices found. Replace with ones.")
        bin_indices[zero_mask] = 1
        
    # Calculate statistics for each bin
    bin_fraction_of_positives = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(bin_indices == b + 1)[0]
        if len(selected) > 0:
            bin_fraction_of_positives[b] = np.mean(true_labels[selected] == 1)
            bin_confidences[b] = np.mean(pos_confidences[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_fraction_of_positives - bin_confidences)

    # Calculate statistics over all bins
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def calculate_correlations_all(values):
    """Used to calculate the Pearson correlation coefficient between calibration error and a set of estimates"""
    correlations = np.zeros(4)
    aces = values[:, 0].astype(float)
    for i in range(1,5):
        raw_errors = values[:, i]

        # convert the errors back from string format to floats (if needed)
        if "$" in raw_errors[0]:
            raw_errors = np.array([s.lstrip("$").split("\\")[0] for s in raw_errors])
        errors = raw_errors.astype(float)
        
        correlations[i-1] = pearsonr(aces, errors)[0]
    return correlations
