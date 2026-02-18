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

def load_mi(input_folder, monitor, state_var, state_avg=False):
    pass

def mi_to_graph(mi, selected_nodes=None):
    
    # Create a graph
    G = nx.Graph()
    num_variables = mi.shape[0]
    
    if selected_nodes is not None:
        for i in selected_nodes:
            G.add_node(i)
        
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                G.add_edge(selected_nodes[i], selected_nodes[j], weight=mi[i, j])

    else:
        # Add nodes (variables)
        for i in range(num_variables):
            G.add_node(i)

    # Add edges with mutual information as weights
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                G.add_edge(i, j, weight=mi[i, j])
    
    return G

def chow_liu_tree(mi):
    
    # Create a graph
    G = mi_to_graph(mi)

    # Find the maximum spanning tree
    # The weight parameter specifies the attribute used for edge weights
    chow_liu_tree = nx.maximum_spanning_tree(G, algorithm='kruskal', weight='weight')
    
    return G, chow_liu_tree


def calculate_chow_liu_trees(mi_matrices, state_vars):

    print("Calculating Chow-Liu trees...")

    chow_liu_trees = {}
    graphs = {}
    chow_liu_matrices = {}

    for i in range(len(state_vars) + 1):

        if i == len(state_vars):
            i = "avg"

        mi = mi_matrices[i]

        G, cl_tree = chow_liu_tree(mi)

        graphs[i] = G
        chow_liu_trees[i] = cl_tree
        chow_liu_matrices[i] = nx.to_numpy_array(cl_tree)

    return graphs, chow_liu_trees, chow_liu_matrices


def main():
    pass


if __name__ == "__main__":
    main()