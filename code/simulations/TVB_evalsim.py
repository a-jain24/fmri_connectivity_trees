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


def calculate_confusion_matrix_elements(true_matrix, predicted_matrix):

    # Flatten the matrices to 1D arrays
    true_flat = np.abs(true_matrix.flatten())
    # print(np.sum(true_flat > 0))
    predicted_flat = np.abs(predicted_matrix.flatten())
    
    # Calculate TP, FP, TN, FN
    TP = np.sum((true_flat > 0) & (predicted_flat > 0))
    FP = np.sum((true_flat == 0) & (predicted_flat > 0))
    TN = np.sum((true_flat == 0) & (predicted_flat == 0))
    FN = np.sum((true_flat > 0) & (predicted_flat == 0))
    
    return TP, FP, TN, FN

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


def evaluate_metrics(simulated_matrices, original_weights, density=True):

    print("Evaluating metrics...")
    
    metrics = {}
    if density:
        original_weights = density_threshold_matrix(original_weights, N=original_weights.shape[0]-1, use_abs=True)
    weights_flat = np.abs(original_weights.flatten())

    for measure in simulated_matrices.keys():

        sim_mat = np.abs(simulated_matrices[measure])
        sim_flat = sim_mat.flatten()

        # Flatten upper triangle
        iu = np.triu_indices_from(sim_mat, k=1)
        sim_vals = sim_mat[iu]
        orig_vals = original_weights[iu]

        # Pearson correlation
        corr, _ = pearsonr(weights_flat, sim_flat)

        # Mean Squared Error
        mse = np.mean((sim_vals - orig_vals) ** 2)

        # confusion matrix elements
        TP, FP, TN, FN = calculate_confusion_matrix_elements(original_weights, sim_mat)

        metrics[measure] = {
            'pearson_correlation': float(corr),
            'mean_squared_error': float(mse),
            'confusion_matrix': {
                'TP': int(TP),
                'FP': int(FP),
                'TN': int(TN),
                'FN': int(FN)
            } 
        } 

    return metrics

def save_metrics(file_path, metrics):

    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_matrices(input_folder, state_vars, monitors, measures=["FC_thrs"], state_avg=False):
    
    matrices = {}

    if state_avg:
        suffix = 'avg'
        for var in state_vars:
            suffix += f"_{var}"

    for monitor in monitors:
        matrices[monitor] = {}

        for measure in measures:
            matrices[monitor][measure] = {}

            for var in state_vars:
                var_index = var if not state_avg else suffix
                file_path = f"{input_folder}/{monitor}/{measure}_var_{var_index}.npz"
                if os.path.exists(file_path):
                    matrix = np.load(file_path, allow_pickle=True)[measure].item()
                    matrices[monitor][measure][var] = matrix
                else:
                    print(f"Matrix file not found: {file_path}")

    return matrices

def run_eval(matrices, monitors, measures, state_vars, original_weights, state_avg=False, density=True):

    metrics = {}
    
    for monitor in monitors:
        metrics[monitor] = {}

        for measure in measures:
            metrics[monitor][measure] = {}

            for var in state_vars:

                var_index = var if not state_avg else 'avg'
                sim_matrix = matrices[monitor][measure][var_index]
                eval_metrics = evaluate_metrics({measure: sim_matrix}, original_weights, density=density)
                metrics[monitor][measure][var] = eval_metrics[measure]

        

def main():
    
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
    
    model = "wong_wang" # or "2d_oscillator"
    marker = "try1_1_27_26"

    monitors = [
        'Region_TemporalAverage', 
        'BOLD'
        ]
    
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

    state_avg = False # whether metrics were computed for averaged state variables

    measures = [
        "FC_thrs",
        "FC_time",
        "FC_density",
        "pvals",
        "mi",
        "mi_density",
        "mi_fc_thrs",
        "mi_fc_time",
        "mi_fc_density",
        "cl",
        "cl_density",
        "cl_fc_thrs",
        "cl_fc_time",
        "cl_fc_density"
        ]
    
    original_weights = np.loadtxt(f"/Users/ajjain/Downloads/Code/tvb-educase-braintumor/data/connectomes/{input_names[0]}_connectome.txt")
    
    for input_name in input_names:
        input_folder = f"/Users/ajjain/Downloads/Code/tvb-educase-braintumor/results/MI/{input_name}/{model}/{marker}"
        matrices = load_matrices(input_folder, wong_wang_state_vars if model=="wong_wang" else oscillator_state_vars, monitors, measures, state_avg=state_avg)
        run_eval(matrices, monitors, measures, wong_wang_state_vars if model=="wong_wang" else oscillator_state_vars, original_weights, state_avg=False, density=True)


    
    
if __name__ == "__main__":
    main()

