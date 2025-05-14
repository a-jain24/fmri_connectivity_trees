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
def construct_url(base_url, file_id):
    return f"{base_url}/{file_id}_func_preproc.nii.gz"

def get_phenotypes(base_url, phenotype_file):

    # load phenotype file
    pheno_data = pd.read_csv(phenotype_file)

    # get abide file names
    abide_files = os.listdir(base_url)

    abide_ids = []
    phenotype = []

    for i in range(len(pheno_data)):

        # get file ids and diagnosis group
        file_id = pheno_data['FILE_ID'][i]
        dx = pheno_data['DX_GROUP'][i]

        # contruct the URL for the fMRI file
        url = f'{file_id}_func_preproc.nii.gz'

        # create phenotype mapping
        if url in abide_files: 
            abide_ids.append(file_id)
            if dx == 1: phenotype.append(1) # autism diagnosis
            else: phenotype.append(0) # tdc
    

    return abide_ids, phenotype

# get pooled subject time series based on the atlas rois
def get_pooled(base_url, abide_ids, phenotype, masker, num_subjects=30):

    pooled_subjects = []

    for func_file, phenotypic in zip(
        abide_ids[:num_subjects],
        phenotype[:num_subjects],
    ):
        time_series = masker.transform(construct_url(base_url, func_file))
        pooled_subjects.append(time_series)
    
    return pooled_subjects

def save_data(pooled_subjects, atlas_name, num_subjects, abide_ids, output_dir='output/roi_time_series'):

    this_output_dir = f'{output_dir}/{num_subjects}_{atlas_name}'

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(this_output_dir):
        os.makedirs(this_output_dir)
    
    if not os.path.exists(f'{this_output_dir}/pooled'):
        os.makedirs(f'{this_output_dir}/pooled')
    
    if not os.path.exists(f'{this_output_dir}/shape'):
        os.makedirs(f'{this_output_dir}/shape')

    for subject, id in zip(
        pooled_subjects,
        abide_ids[:num_subjects]
    ):
        # Convert each subject's time series to a numpy array and get shape
        subject = np.array(subject)
        shape = subject.shape

        # Save each subject's time series to a separate file
        subject_id = id  # Assuming the first element is the subject ID
        np.savetxt(f'{this_output_dir}/pooled/{subject_id}.csv', subject, delimiter=',')
        np.savetxt(f'{this_output_dir}/shape/{subject_id}.csv', shape, delimiter=',')


# extract whole brain time series using the HarvardOxford atlas
def extract_whole_time_series(base_url, phenotype_file, atlas_name='HarvardOxford', atlas_dir='atlases', num_subjects=30):
    # fetch the atlas
    atlas = fetch_atlas(atlas_name, atlas_dir=atlas_dir)

    # get phenotypes and file ids
    abide_ids, phenotype = get_phenotypes(base_url, phenotype_file)

    # create masker based on the atlas type
    masker = get_masker(atlas, mask_type='whole')  

    # get pooled subjects time series
    pooled_subjects = get_pooled(base_url, abide_ids, phenotype, masker, num_subjects)

    # save the pooled data
    save_data(pooled_subjects, atlas_name, num_subjects, abide_ids, 'output/roi_time_series')

# extract cortical time series using the MSDL atlas
def extract_cort_time_series(base_url, phenotype_file, atlas_name='MSDL', num_subjects=30):

    # fetch the atlas
    atlas = fetch_atlas(atlas_name)

    # get phenotypes and file ids
    abide_ids, phenotype = get_phenotypes(base_url, phenotype_file)

    # create masker based on the atlas type
    masker = get_masker(atlas, mask_type='cort')  

    # get pooled subjects time series
    pooled_subjects = get_pooled(base_url, abide_ids, phenotype, masker, num_subjects)

    # save the pooled data
    save_data(pooled_subjects, atlas_name, num_subjects, abide_ids, 'output/roi_time_series')

def main():

    # set the working directory to fmri_connectivity_trees root directory
    working_dir = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees'

    # CHANGE PATHS HERE
    abide_url = "/mfs/io/groups/dmello/projects/dynamric/tree_mri/datasets/abide/preprocessed_dataset/Outputs/dparsf/filt_noglobal/func_preproc"
    phenotype_file = f"{working_dir}/datasets/abide/phenotypic/Phenotypic_V1_0b_preprocessed1.csv"

    # SET THE NUMBER OF SUBJECTS, MAX IS 884
    num_subjects = 884  # You can change this to any number up to 884
    extract_cort_time_series(abide_url, phenotype_file, num_subjects=num_subjects)
    #extract_whole_time_series(abide_url, phenotype_file, num_subjects=num_subjects)


if __name__ == "__main__":
    main()



