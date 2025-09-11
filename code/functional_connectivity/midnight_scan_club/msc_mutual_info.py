import nilearn
from nilearn import datasets, plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
import os
import requests
import csv
import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import scipy
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics.cluster import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import torch
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# concatenate timeseries to get a single timeseries for each subject
def get_timeseries(tasks, cort_shape_path, cort_pooled_path):
    timeseries = []
    for id in tasks:

        # load the shape and pooled timeseries
        shape = np.loadtxt(f'{cort_shape_path}/{id}.csv', delimiter=',').astype(int)
        pooled = np.loadtxt(f'{cort_pooled_path}/{id}.csv', delimiter=',').reshape(shape)

        # concatenate the timeseries
        timeseries.append(pooled)

    return timeseries

def get_concat(timeseries, use_torch=True):
    """
    Concatenate the timeseries for each subject.
    """
    concat_timeseries = {}
    for subject, ts in timeseries.items():
        concat_timeseries[subject] = np.concatenate(ts, axis=0).T
    if use_torch:
        for subject in concat_timeseries:
            concat_timeseries[subject] = torch.tensor(np.float32(concat_timeseries[subject]), device=device)
    return concat_timeseries

def combine_cort_and_cereb(cort_concat_timeseries, cereb_concat_timeseries, use_torch=True):
    
    timeseries = {}
    for subject in cort_concat_timeseries:
        timeseries[subject] = np.concatenate((cort_concat_timeseries[subject].cpu().numpy(), cereb_concat_timeseries[subject].cpu().numpy()), axis=0)
    if use_torch:
        for subject in timeseries:
            timeseries[subject] = torch.tensor(np.float32(timeseries[subject]), device=device)
    return timeseries

def combine_timeseries(subjects, timeseries_array, use_torch=True):

    if len(timeseries_array) < 2:
        return get_concat(timeseries_array[0], use_torch=use_torch)
    
    concat_timeseries = {}
    for subject in subjects:
        concat_timeseries[subject] = timeseries_array[0][subject].cpu().numpy()
        for timeseries in timeseries_array[1:]:
            concat_timeseries[subject] = np.concatenate((concat_timeseries[subject].cpu().numpy(), timeseries[subject].cpu().numpy()), axis=0)

    if use_torch:
        for subject in concat_timeseries:
            concat_timeseries[subject] = torch.tensor(np.float32(concat_timeseries[subject]), device=device)

    return concat_timeseries

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

def pairwise_roi_mutual_information(timeseries, num_bins=10):
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

def save_data(data, subject, base_dir='/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club', measure="mutual_information", task="rest", atlas="Schaefer", atlas_subdir="schaefer_100", skl=False, num_bins=10):
    """
    Save the covariance matrix to a CSV file.
    """
    output_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/{measure}/{subject}/{atlas_subdir}'
    if skl:
        task = f'{task}_skl_{num_bins}bins'
    else:
        task = f'{task}_{num_bins}bins'
    os.makedirs(output_path, exist_ok=True)
    np.save(f'{output_path}/{task}.npy', data.cpu())

def load_data(subjects=['MSC01'], base_dir='/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club', measure="mutual_information", task="rest", atlas="Schaefer", atlas_subdir="schaefer_100", skl=False, num_bins=10):
    """
    Save the covariance matrix to a CSV file.
    """
    data = {}
    for subject in subjects:
        data_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/{measure}/{subject}/{atlas_subdir}'
        if skl:
            file_name = f'{task}_skl_{num_bins}bins'
        else:
            file_name = f'{task}_{num_bins}bins'

        data[subject] = np.load(f'{data_path}/{file_name}.npy')
    return data

def plot_connectome(cov_matrix, coords, title='connectome', edge_threshold="0%", edge_vmin=0, edge_vmax=1):
    """
    Plot the connectome using the mean correlation matrix and coordinates.
    """
    # Plotting the connectome
    plotting.plot_connectome(
        cov_matrix,
        coords,
        title=title,
        edge_vmin = edge_vmin,
        edge_vmax = edge_vmax,
        edge_cmap='cold_hot',
        edge_threshold=edge_threshold,  # Only plot edges above the 90th percentile
        colorbar=True,
        node_size=10,
    )
    plt.show()

def get_mi_matrices(atlases, subjects, tasks, sessions, base_dir):

    timeseries_array = []
    atlas_name = ''

    for atlas in atlases:

        atlas_name = atlas_name + '_' + atlas if atlas_name != '' else atlas

        timeseries = {}
        for subject in subjects:
            shape_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/roi_time_series/{subject}/func01/{atlas}/all_tasks/shape/'
            pooled_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/roi_time_series/{subject}/func01/{atlas}/all_tasks/pooled/'
            timeseries[subject] = get_timeseries(tasks, shape_path, pooled_path)
            for session in sessions[1:]:
                shape_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/roi_time_series/{subject}/{session}/{atlas}/all_tasks/shape/'
                pooled_path = f'{base_dir}/code/functional_connectivity/midnight_scan_club/output/roi_time_series/{subject}/{session}/{atlas}/all_tasks/pooled/'
                timeseries[subject] = timeseries[subject] + get_timeseries(tasks, shape_path, pooled_path)
        timeseries_array.append(timeseries)

    concat_timeseries = combine_timeseries(subjects, timeseries_array)
    print(concat_timeseries['MSC04'].shape)

    for subject in subjects:
        for task in tasks:
            print(f'Computing MI for subject {subject}, task {task}, atlases {atlas_name}')
            mi, joint_probs, product_of_marginals = pairwise_roi_mutual_information(concat_timeseries[subject], num_bins=100)
            save_data(mi, base_dir=base_dir, subject=subject, task=task, atlas=atlas, num_bins=100, atlas_subdir=atlas_name)
            save_data(joint_probs, base_dir=base_dir, subject=subject, measure='joint_probs', task=task, atlas=atlas, atlas_subdir=atlas_name, skl=False, num_bins=100)
            save_data(product_of_marginals, base_dir=base_dir, subject=subject, measure='product_of_marginals', task=task, atlas=atlas, atlas_subdir=atlas_name, skl=False, num_bins=100)

def main():

    # set the working directory to fmri_connectivity_trees root directory
    home_base_dir = '/Users/aj/dmello_lab/fmri_connectivity_trees' # directory where repository lives at home computer
    lab_base_dir = '/Users/ajjain/Downloads/Code/fmri_connectivity_trees' # directory where repository lives at lab computer
    utd_base_dir = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club'

    # set base directory depending on where the code is being run
    base_dir = home_base_dir if os.path.exists(home_base_dir) else lab_base_dir
    base_dir = utd_base_dir if os.path.exists(utd_base_dir) else base_dir

    os.chdir(base_dir)
    
    # path for shapes and pooled timeseries
    subjects = [
                'MSC01',
                'MSC02', 
                'MSC03',
                'MSC04',
                'MSC05', 
                'MSC06',
                'MSC07', 
                'MSC08',
                'MSC09',
                'MSC10'
                ]
    sessions = ['func01', 'func02', 'func03', 'func04', 'func05', 'func06', 'func07', 'func08', 'func09', 'func10']
    tasks = ['rest']
    atlases = ['glasser360', 'SUIT', 'Thalamus', 'Brainstem']

    get_mi_matrices(atlases, subjects, tasks, sessions, base_dir)

if __name__ == "__main__":
    main()