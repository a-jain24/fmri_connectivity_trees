# load libraries
import csv
import matplotlib.pyplot as plt
import multiprocessing as mp 
import numpy as np
import scipy.io as sio
import scipy.stats as stat
import scipy
import zipfile
from tvb.simulator.plot.tools import *
from tvb.simulator.lab import *
from tvb.contrib.scripts.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
import torch
import networkx as nx
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import os
import json

data_dir = os.path.abspath("TVB_input")
zip_suffix = "_TVB" 


# load pre-defined connectivity
def load_connectivity(input_name):
    zip_file_name = input_name + zip_suffix + ".zip"
    dir_name = input_name + zip_suffix
    zip_path = os.path.join(data_dir,  input_name, zip_file_name)
    dir_path = os.path.join(data_dir, input_name, dir_name)
    # Load the connectivity data
    conn = connectivity.Connectivity.from_file(zip_path)
    # Configure, to compute derived data, such as number_of_nodes and delays
    conn.configure()
    
    # Check weight matrix from .zip is corresponding to structural connectivity matrix from matlab file. 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
    weight_txt = np.loadtxt(fname = dir_path + "/weights.txt")
    # Load the structural connectivity matrix from matlab file
    SC_path = data_dir + "/" + input_name + "/SCthrAn.mat"
    x = sio.loadmat(SC_path)
    assert np.allclose(x['SCthrAn'], weight_txt), "Weight matrix in weights.txt should be the same as SCthrAn.mat"
    return conn

# load simulation data
def load_data(input_folder, monitors):
    
    data = {}
    for i in range(len(monitors)):
        data[monitors[i]] = np.load(f"{input_folder}/{monitors[i]}.npz")
        
    return data

# load empirical functional connectivity
def load_empirical_functional_connectivity(input_name):

    # get empirical functional connectivity from FC.mat
    # in this demo the functional connectivity parameter name is FC_cc_DK68, adjust to your variable name
    fc_file_name = os.path.join(data_dir, input_name, "FC.mat")
    fc_cc_name = "FC_cc_DK68"
    em_fc_matrix = sio.loadmat(fc_file_name)[fc_cc_name]

    # indexes of all the values above the diagonal.
    uidx = np.triu_indices(68, 1)

    # Fisher-Z transform the correlations, important for standardization
    em_fc_z = arctanh(em_fc_matrix)
    # get the upper triangle since it is symmetric along diagonal
    em_fc = em_fc_z[uidx]

    return em_fc, em_fc_matrix


# select data for a specific monitor
def select_monitor(data, monitor_name):
    return data[monitor_name]

# choose specific state variables from simulator output
def choose_state_variables(y, var_indices=[0], avg=False):
    """
    Choose specific state variables from the simulator output.

    Parameters
    ----------
    y : np.ndarray
        The output data from the simulator.
    var_indices : list of int
        The indices of the state variables to select.

    Returns
    -------
    np.ndarray
        The selected state variables.
    """
    y_state = y[:, var_indices[0]:(var_indices[0]+1), :, 0]
    if len(var_indices) > 1:
        for index in var_indices[1:]:
            y_state = np.concatenate((y_state, y[:, index:(index+1), :, 0]), axis=1)
    if avg:
        y_state = np.mean(y_state, axis=1, keepdims=True)
    return y_state



# density thresholding function
def density_threshold_matrix(C, N, use_abs=True):
        
        # convert C to numpy if torch tensor
        if torch.is_tensor(C):
            C = C.cpu().numpy()
        
        C = C.copy()

        # get density thresholded
    
        np.fill_diagonal(C, 0.0)

        if use_abs:
            W = np.abs(C)
        else:
            W = C

        # Upper triangle indices
        iu = np.triu_indices_from(W, k=1)

        weights = W[iu]

        if N > len(weights):
            raise ValueError("N exceeds number of possible edges")

        # Find cutoff for top N edges
        idx = np.argpartition(weights, -N)[-N:]
        mask = np.zeros_like(weights, dtype=bool)
        mask[idx] = True

        # Build thresholded matrix
        C_thr = np.zeros_like(C)
        C_thr[iu[0][mask], iu[1][mask]] = C[iu[0][mask], iu[1][mask]]
        C_thr = C_thr + C_thr.T

        return C_thr

## MI CODE

def discretize_time_series(timeseries, num_bins=10):
    """
    Discretize the time series data into specified number of bins using matrix operations.
    """
    timeseries = timeseries.contiguous()  # Ensure the tensor is contiguous for efficient operations
    bin_edges = torch.linspace(timeseries.min(), timeseries.max(), steps=num_bins + 1, device=timeseries.device)[1:-1]
    discretized = torch.bucketize(timeseries, bin_edges)
    return discretized


def joint_prob(a, b, discretized_timeseries, num_bins=10):
    """
    Compute the joint probability matrix for two ROIs a and b using matrix operations.
    """
    a = discretized_timeseries[a, :]
    b = discretized_timeseries[b, :]
    joint_indices = num_bins * a + b
    joint_prob_mat = torch.bincount(joint_indices, minlength=num_bins * num_bins).reshape(num_bins, num_bins).float()
    joint_prob_mat /= joint_prob_mat.sum()  # Normalize to get probabilities
    return joint_prob_mat


def get_joint_probs(discretized_timeseries, num_bins=10):
    """
    Compute the joint probability matrices for all pairs of ROIs using matrix operations.
    """
    num_rois = discretized_timeseries.shape[0]
    joint_probs = torch.zeros((num_rois, num_rois, num_bins, num_bins), dtype=torch.float32, device=discretized_timeseries.device)
    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                joint_probs[i, j] = joint_prob(i, j, discretized_timeseries, num_bins)
    return joint_probs


def marginal_prods(a, b, joint_probs, num_bins=10):
    """
    Compute the product of marginals for two ROIs a and b using matrix operations.
    Element (i, j) is P(a=i) * P(b=j).
    """
    joint_prob_mat = joint_probs[a, b]
    marginal_a = joint_prob_mat.sum(axis=1)  # P(a=i)
    marginal_b = joint_prob_mat.sum(axis=0)  # P(b=j)
    return torch.outer(marginal_a, marginal_b)  # Element (i, j) = P(a=i) * P(b=j)


def get_product_of_marginals(joint_probs, num_bins=10):
    """
    Compute the product of marginals for all pairs of ROIs using matrix operations.
    """
    num_rois = joint_probs.shape[0]
    product_marginals = torch.zeros((num_rois, num_rois, num_bins, num_bins), dtype=torch.float32, device=joint_probs.device)
    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                product_marginals[i, j] = marginal_prods(i, j, joint_probs, num_bins)
    return product_marginals


def mutual_information(a, b, joint_probs, product_marginals, num_bins=10):
    """
    Compute the mutual information between two ROIs a and b.
    """
    joint_prob_mat = joint_probs[a, b]
    product_marginal_mat = product_marginals[a, b]

    # Avoid division by zero
    joint_prob_mat[joint_prob_mat == 0] = 1e-10
    product_marginal_mat[product_marginal_mat == 0] = 1e-10

    # Compute mutual information
    mi = torch.sum(joint_prob_mat * torch.log(joint_prob_mat / product_marginal_mat))

    return mi


def get_mutual_information(joint_probs, product_marginals, num_bins=10):
    """
    Compute the mutual information matrix for all pairs of ROIs using matrix operations.
    """
    num_rois = joint_probs.shape[0]
    mi_matrix = torch.zeros((num_rois, num_rois), dtype=torch.float32, device=joint_probs.device)
    # Compute mutual information for each pair of ROIs
    for i in range(num_rois):
        for j in range(num_rois):
            if i != j:
                mi_matrix[i, j] = mutual_information(i, j, joint_probs, product_marginals, num_bins)
    return mi_matrix


def pairwise_roi_mutual_information(timeseries, num_bins=100):
    """
    Compute the pairwise mutual information matrix for the given time series data.
    This function discretizes the time series data, computes joint probabilities,
    computes the product of marginals, and then calculates the mutual information matrix.
    """
    discretized_timeseries = discretize_time_series(timeseries, num_bins=num_bins)
    joint_probs = get_joint_probs(discretized_timeseries, num_bins=num_bins)
    product_marginals = get_product_of_marginals(joint_probs, num_bins=num_bins)
    mi_matrix = get_mutual_information(joint_probs, product_marginals, num_bins=num_bins)
    return mi_matrix, joint_probs, product_marginals


# main functional connectivity calculation function
def calculate_mutual_information(data, state_vars, monitors, alpha=0.05, fdr=True, n_perm=1000, use_abs=True):
    """
    data: dictionary of monitor outputs
    returns: thresholded significance matrix, timeshifted significance matrix, 
    p-values matrix, thresholded connectivity matrix
    """
    print("Calculating functional connectivity...")
    print(y.shape)

    mis = {}
    mi_densities = {}
    
    for monitor in monitors:

        print(f"Processing monitor: {monitor}")
        y = select_monitor(data, monitor)['data']
        print(y.shape)
        y = choose_state_variables(y, state_vars)
        print(y.shape)

        T, S, N = y.shape

        mis[monitor] = {}
        mi_densities[monitor] = {}


        for i in range(S):

            
            # get data for this specific state variable
            this_y = y[:, i, :]
            this_y_tensor = torch.tensor(this_y.T, dtype=torch.float32)
            
            mi_matrix, _, _ = pairwise_roi_mutual_information(this_y_tensor, num_bins=100)
            mi_matrix = mi_matrix.cpu().numpy()
            
            # update dicts
            mis[monitor][i] = mi_matrix
            mi_densities[monitor][i] = density_threshold_matrix(mi_matrix, N-1, use_abs=use_abs)
    

    return mis, mi_densities


def save_data(output_folder, state_vars, monitors, mis, mi_densities, state_avg=False):

    if state_avg:
        suffix = 'avg'
        for var in state_vars:
            suffix += f"_{var}"
    
    for monitor in monitors:
        for i in range(len(state_vars)):
            var_index = state_vars[i] if not state_avg else suffix
            np.savez_compressed(f"{output_folder}/{monitor}/mi_var_{var_index}.npz", mi=mis[monitor][i])
            np.savez_compressed(f"{output_folder}/{monitor}/mi_density_var_{var_index}.npz", mi_density=mi_densities[monitor][i])


def main():
    
    oscillator_state_vars = [
        0, #V
        1  #W
    ]

    wong_wang_state_vars = [
        0,  # Excitatory synaptic gating variable
        1,  # Inhibitory synaptic gating variable
        2,  # Excitatory firing rate
        3,  # Inhibitory firing rate
        4,  # Excitatory intermediate rate variable
        5,  # Inhibitory intermediate rate variable
        6,  # Total excitatory input current
        7,   # Total inhibitory input current
    ]

    state_avg = False # whether to compute metrics for averaged state variables

    model = "wong_wang"  # or "2d_oscillator"

    monitors = [
        'tavg',
        'bold'
    ]

    if model == "wong_wang":
        state_vars = wong_wang_state_vars
    elif model == "2d_oscillator":
        state_vars = oscillator_state_vars

    input_names = [
        # "CON02T1",
        # "CON02T2",
        # "CON03T1",
        # "CON03T2",
        # "PAT01T1",
        # "PAT01T2",
        "PAT02T1",
        # "PAT02T2",
        ]

    marker = 'try3'


    for input_name in input_names: 
        input_folder = f"/Users/ajjain/Downloads/Code/tvb-educase-braintumor/TVB_input/{input_name}/{model}/{marker}"
        data = load_data(input_folder, monitors)
        output_folder = f"/Users/ajjain/Downloads/Code/tvb-educase-braintumor/results/MI/{input_name}/{model}/{marker}"
        mis, mi_densities = calculate_mutual_information(data, state_vars, monitors, alpha=0.05, fdr=True, n_perm=1000, use_abs=True, state_avg=state_avg)
        save_data(output_folder, state_vars, monitors, mis, mi_densities, state_avg=state_avg)


if __name__ == "__main__":
    main()