import os
import glob

import pandas as pd
import nibabel as nib
import numpy as np
from scipy.stats import mannwhitneyu, permutation_test, wilcoxon

import seaborn as sns
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from nilearn.plotting import view_img_on_surf, view_img, plot_stat_map, plot_img_on_surf
from nilearn.reporting import get_clusters_table


#### Group map utils ####

def stats_func(x):
    """ Custom statistics function to compute the median of non-zero values.
    Args:
        x (array-like): Input array.
    Returns:
        float: mean of non-zero values or NaN if no non-zero values exist.
    """
    x = np.nanmean(x, axis=0)  # First compute the mean across the first axis
    return x


def plot_map_on_suit(img, cmap='coolwarm', cscale=None, title='', threshold=None, stats_func=stats_func, output_file=None):
    """ Plot the cerebellar flatmap for the given image.
    Args:
        img : A Nifti1Image of the t-statistic map.
        cmap (str): Colormap to use.
        cscale (list): Color scale limits [vmin, vmax].
        threshold (float): Threshold value for the data.
        title (str): Title for the plot.
    Returns:
        parcel_suit_fig: An interactive flatmap figure of the t-statistic map on the cerebellar surface.
    """
    parcel_flat = flatmap.vol_to_surf([img],
                                            stats=stats_func,
                                            space='SUIT',
                                            ignore_zeros=True)    
    # plot the flatmaps
    colorbar = True
    parcel_suit_fig = flatmap.plot(data=parcel_flat, 
                            render="matplotlib", 
                            hover='value', 
                            cmap = cmap,
                            threshold=threshold,
                            bordersize = 1, 
                            colorbar=colorbar,
                            cscale = cscale)
    parcel_suit_fig.set_title(title)
    if output_file is not None:
        parcel_suit_fig.figure.savefig(output_file, dpi=300)
        plt.close(parcel_suit_fig.figure)
        return
    return parcel_suit_fig

def plot_map_on_surf(img, cmap='coolwarm', cscale=[None, None], threshold=None, title='', output_file=None):
    """Plot a tstat map on the fsaverage surface.
    Args:
        img : A Nifti1Image of the t-statistic map.
        cmap (str): Colormap to use (default is 'coolwarm').
        cscale (list): Color scale limits [vmin, vmax] (default is [None, None]).
        threshold (float): Threshold value for the data (default is None).
        title (str): Title for the plot (default is '').
    Returns:
        view: An interactive view of the t-statistic map on the surface.
    """
    view = plot_img_on_surf(img, vmin=cscale[0], vmax=cscale[1], threshold=threshold, surf_mesh='fsaverage5', colorbar=True, cmap=cmap, inflate=True, title=title, output_file=output_file)
    return view

def get_group_map(subs, task, firstlevel_dir, space='MNI152NLin2009cAsym', tmap=None, percentile=80, group_thresh=2, mask=None, pos_only=False):
    """Get a group's average tstat map.
    Args:
        subs (list): List of subject IDs (e.g., ['sub-C001', 'sub-C002']).
        task (str): Task name (e.g., 'langlocaudio').
        firstlevel_dir (str): Path to the first-level analysis directory.
        space (str): template space ('MNI152NLin2009cAsym' or 'T1w').
        tmap (str): Name of the t-statistic map file (e.g., 'spmT_0001.nii'). If None, defaults to 'spmT_0001.nii'.
        percentile (float): Percentile threshold for individual maps (default is 80).
        group_thresh (int): Minimum number of subjects exceeding the threshold for group map (default is 2).
        mask (str): Path to a brain mask NIfTI file. If None, no masking is applied.
        pos_only (bool): If True, only positive t-values are considered.
    Returns:
        img: A Nifti1Image of the average t-statistic map.
    """
    tstat_imgs = []
    for sub in subs:
        tstat_nii = f'{firstlevel_dir}/{sub}/{task}_space-{space}/{tmap}.nii'
        tstat_img = nib.load(tstat_nii)
        if mask is not None:
            mask_img = nib.load(mask)
            tstat_data = tstat_img.get_fdata()
            mask_data = mask_img.get_fdata()
            tstat_data = np.where(mask_data > 0, tstat_data, 0)
            tstat_img = nib.Nifti1Image(tstat_data, tstat_img.affine, tstat_img.header)
        tstat_imgs.append(tstat_img)

    # Compute the group-level t-stat map by summing thresholded individual maps
    data = np.zeros(tstat_imgs[0].shape)
    for img in tstat_imgs:
        sub_data = img.get_fdata()
        t_thresh = np.percentile(sub_data, percentile)
        if pos_only:
            sub_data = np.where(sub_data > t_thresh, 1, 0)
        else:
            sub_data = np.where(sub_data > t_thresh, 1, np.where(sub_data < -t_thresh, -1, 0))
        data += sub_data
    data = np.where(np.abs(data) >= group_thresh, data, 0)
    img = nib.Nifti1Image(data, tstat_imgs[0].affine, tstat_imgs[0].header)
    return img


#### FROI analysis utils ####

def get_labels(task):
    parcel_labels = f"/mfs/io/groups/dmello/projects/citrus3t/code/parcel_segmentation/parcellation/{task}_parcels.txt"
    parcels = pd.read_csv(parcel_labels, header=None, names=['roi'])
    return parcels

def get_data_allsubj(data_dir, task, group=None, group_name='all'):
    parcels = get_labels(task)
    results = glob.glob(os.path.join(data_dir, f"{task}/*/*_avgcon.csv"))
    data = {
        'sub_id': [],
        'group': [],
        'task': [],
        'roi': [],
        'roi_name': [],
        'Y': []
    }
    for i, result in enumerate(results):
        sub_id = os.path.basename(result).split('_')[0]
        if group is not None and sub_id not in group:
            continue
        df = pd.read_csv(result)
        for i, row in df.iterrows():
            roi = row['roi']
            data['sub_id'].append(sub_id)
            data['group'].append(group_name)
            data['task'].append(task)
            data['roi'].append(int(roi))
            data['roi_name'].append(parcels['roi'][i])
            data['Y'].append(row['value'])
    
    data = pd.DataFrame(data)
    data = data.sort_values(by=['sub_id', 'group','task'], ignore_index=True)
    return data