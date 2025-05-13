import nilearn
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
import os
import requests
import csv
import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

# fetch different atlases
def fetch_atlas(atlas_name, atlas_dir=None):

    if atlas_name == 'HarvardOxford':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm', data_dir=atlas_dir)
    elif atlas_name == 'MSDL':
        atlas = datasets.fetch_atlas_msdl()
    elif atlas_name == 'Cerebellum':
        # Assuming the cerebellum atlas is already downloaded and available locally
        atlas = nib.load(os.path.join(atlas_dir, 'Cerebellum-MNIsegment-1segment.nii'))
    else:
        raise ValueError("Atlas not recognized. Please choose 'HarvardOxford', 'MSDL', or 'Cerebellum'.")
    return atlas

def get_masker(atlas, mask_type='cort'):

    if mask_type == 'cort':
        masker = NiftiMapsMasker(
        atlas.maps,
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory="nilearn_cache",
        memory_level=1,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    ).fit()
    
    # double check TR
    elif mask_type == 'whole':
        masker = NiftiLabelsMasker(
        atlas.maps,
        labels=atlas.labels,
        resampling_target="data",
        t_r=2,
        detrend=True,
        standardize="zscore_sample",
    ).fit()
        
    else:
        raise ValueError("mask_type must be either 'cort' or 'whole'.")

    return masker

# Function to construct the download URL
def construct_url(base_url, file_id, subject_id='MSC01', session='func01'):
    return f"{base_url}/s_sub-{subject_id}_ses-{session}_task-{file_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

# get pooled subject time series based on the atlas rois
def get_pooled(base_url, file_ids, masker, subject_id='MSC01', session='func01'):

    pooled_subject = []

    for file_id in file_ids:
        # skip empty file_ids
        if file_id == '':
            continue
        # construct the URL for the subject's fMRI data
        time_series = masker.transform(construct_url(base_url, file_id, subject_id=subject_id, session=session))
        pooled_subject.append(time_series)
    
    return pooled_subject

def save_data(pooled_subject, atlas_name, file_ids, output_dir='output/roi_time_series', subject_id="MSC01", tasks="all_tasks", session='func01'):

    this_output_dir = f'{output_dir}/{subject_id}/{session}/{atlas_name}/{tasks}'

    # create output directory if it doesn't exist
    if not os.path.exists(f'{this_output_dir}/pooled'):
        os.makedirs(f'{this_output_dir}/pooled')
    
    if not os.path.exists(f'{this_output_dir}/shape'):
        os.makedirs(f'{this_output_dir}/shape')

    for task, file_id in zip(
        pooled_subject,
        file_ids
    ):
        # Convert each subject's time series to a numpy array and get shape
        task = np.array(task)
        shape = task.shape

        np.savetxt(f'{this_output_dir}/pooled/{file_id}.csv', task, delimiter=',')
        np.savetxt(f'{this_output_dir}/shape/{file_id}.csv', shape, delimiter=',')


# extract whole brain time series using the HarvardOxford atlas
def extract_whole_time_series(base_url, file_ids, atlas_name='HarvardOxford', atlas_dir='atlases'):
    # fetch the atlas
    atlas = fetch_atlas(atlas_name, atlas_dir=atlas_dir)

    # create masker based on the atlas type
    masker = get_masker(atlas, mask_type='whole')  

    # get pooled subjects time series
    pooled_subject = get_pooled(base_url, file_ids, masker)

    # save the pooled data
    save_data(pooled_subject, atlas_name, file_ids, 'output/roi_time_series')

# extract cortical time series using the MSDL atlas
def extract_cort_time_series(base_url, file_ids, atlas_name='MSDL', tasks="all_tasks", subject_id='MSC01', session='func01'):

    # fetch the atlas
    atlas = fetch_atlas(atlas_name)

    # create masker based on the atlas type
    masker = get_masker(atlas, mask_type='cort')  

    # get pooled subjects time series
    pooled_subject = get_pooled(base_url, file_ids, masker, subject_id=subject_id, session=session)

    # save the pooled data
    save_data(pooled_subject, atlas_name, file_ids, 'output/roi_time_series', tasks=tasks, subject_id=subject_id, session=session)

def main():

    # set the working directory to fmri_connectivity_trees root directory
    working_dir = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club'
    os.chdir(working_dir)
    
    
    try_rest = False # just try running the rest condition
    tasks = "all_tasks"

    # set the subject and session
    subject_id = "MSC03"
    session = "func01"

    # CHANGE PATHS HERE
    base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"

    ids_path = "/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/midnight_scan_club/msc_file_ids.txt"
    with open(ids_path, 'r') as f:
            file_ids = f.readlines()
            file_ids = [x.strip() for x in file_ids]
    
    if try_rest:
        ids_path = "/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/midnight_scan_club/msc_rest_id.txt"
        with open(ids_path, 'r') as f:
            file_ids = f.readlines()
            file_ids = [x.strip() for x in file_ids]
        tasks = "rest"
    
    run_all_sessions = False
    if run_all_sessions:
        all_sessions = ['func01', 'func02', 'func03', 'func04', 'func05', 'func06', 'func07', 'func08', 'func09', 'func10']
        for session in all_sessions[1:]:
            base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"
            extract_cort_time_series(base_url, file_ids, subject_id=subject_id, session=session, tasks=tasks)
    else:
        extract_cort_time_series(base_url, file_ids, subject_id=subject_id, session=session, tasks=tasks)
    #extract_whole_time_series(abide_url, phenotype_file, num_subjects=num_subjects)


if __name__ == "__main__":
    main()



