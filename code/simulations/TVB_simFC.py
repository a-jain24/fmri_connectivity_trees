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

# calculate functional connectivity and threshold based on significance
def fc_significance_threshold(X, alpha=0.05, fdr=True):
    """
    X: (time, regions)
    returns: thresholded FC matrix
    """
    T, N = X.shape

    # 1. Pearson correlation
    C = np.corrcoef(X, rowvar=False)

    # Numerical safety
    C = np.clip(C, -0.999999, 0.999999)

    # 2. Fisher z-transform
    Z = np.arctanh(C)

    # 3. Z-statistic for H0: r = 0
    Z_stat = Z * np.sqrt(T - 3)

    # Two-sided p-values
    pvals = 2 * (1 - norm.cdf(np.abs(Z_stat)))

    # Ignore diagonal
    np.fill_diagonal(pvals, 1.0)

    # 4. Multiple comparison correction
    if fdr:
        mask = np.triu(np.ones_like(pvals, dtype=bool), k=1)
        rejected, pvals_corr = fdrcorrection(pvals[mask], alpha=alpha)
        sig = np.zeros_like(C, dtype=bool)
        sig[mask] = rejected
        sig = sig | sig.T
    else:
        sig = pvals < alpha

    # Thresholded FC
    FC_thr = C * sig
    return FC_thr, C, pvals


# calculate functional connectivity and threshold based on time-shifted null model
def fc_timeshift_null(X, n_perm=1000, alpha=0.05):
    T, N = X.shape
    C_emp = np.corrcoef(X, rowvar=False)

    null = np.zeros((n_perm, N, N))

    for p in range(n_perm):
        Xs = np.zeros_like(X)
        for i in range(N):
            shift = np.random.randint(T)
            Xs[:, i] = np.roll(X[:, i], shift)
        null[p] = np.corrcoef(Xs, rowvar=False)

    # Two-sided p-values
    pvals = np.mean(np.abs(null) >= np.abs(C_emp), axis=0)
    np.fill_diagonal(pvals, 1.0)

    FC_thr = C_emp * (pvals < alpha)
    return FC_thr, C_emp, pvals

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

# main functional connectivity calculation function
def calculate_functional_connectivity(data, state_vars, monitors, alpha=0.05, fdr=True, n_perm=1000, use_abs=True):
    """
    data: dictionary of monitor outputs
    returns: thresholded significance matrix, timeshifted significance matrix, 
    p-values matrix, thresholded connectivity matrix
    """
    print("Calculating functional connectivity...")
    print(y.shape)

    FC_thrs = {}
    FC_times = {}
    FC_densities = {}
    p_vals = {}
    
    for monitor in monitors:
        print(f"Processing monitor: {monitor}")
        y = select_monitor(data, monitor)['data']
        print(y.shape)
        y = choose_state_variables(y, state_vars)
        print(y.shape)

        T, S, N = y.shape

        FC_thrs[monitor] = {}
        FC_times[monitor] = {}
        FC_densities[monitor] = {}
        p_vals[monitor] = {}


        for i in range(S):
            
            # get data for this specific state variable
            this_y = y[:, i, :]
            
            FC_thr, C, pvals = fc_significance_threshold(this_y, alpha=alpha, fdr=fdr)
            FC_time, _, _ = fc_timeshift_null(this_y, n_perm=n_perm, alpha=alpha)


            # update dicts
            FC_thrs[monitor][i] = FC_thr
            p_vals[monitor][i] = pvals
            FC_times[monitor][i] = FC_time
            FC_densities[monitor][i] = density_threshold_matrix(C, N-1, use_abs=use_abs)
    

    return FC_thrs, FC_times, FC_densities, pvals


def save_data(output_folder, state_vars, monitors, FC_thrs, FC_times, FC_densities, pvals, state_avg=False):

    if state_avg:
        suffix = 'avg'
        for var in state_vars:
            suffix += f"_{var}"
    
    for monitor in monitors:
        for i in range(len(state_vars)):
            var_index = state_vars[i] if not state_avg else suffix
            np.savez_compressed(f"{output_folder}/{monitor}/FC_thr_var_{var_index}.npz", FC_thrs=FC_thrs[monitor][i])
            np.savez_compressed(f"{output_folder}/{monitor}/FC_time_var_{var_index}.npz", FC_time=FC_times[monitor][i])
            np.savez_compressed(f"{output_folder}/{monitor}/FC_density_var_{var_index}.npz", FC_density=FC_densities[monitor][i])
            np.savez_compressed(f"{output_folder}/{monitor}/pvals_var_{var_index}.npz", pvals=pvals[monitor][i])


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
        output_folder = f"/Users/ajjain/Downloads/Code/tvb-educase-braintumor/results/FC/{input_name}/{model}/{marker}"
        FC_thrs, FC_times, FC_densities, pvals = calculate_functional_connectivity(data, state_vars, monitors, alpha=0.05, fdr=True, n_perm=1000, use_abs=True, state_avg=state_avg)
        save_data(output_folder, state_vars, monitors, FC_thrs, FC_times, FC_densities, pvals, state_avg=state_avg)


if __name__ == "__main__":
    main()