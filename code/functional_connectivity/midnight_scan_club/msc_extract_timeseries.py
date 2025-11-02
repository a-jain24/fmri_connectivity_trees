import nilearn
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker, NiftiMapsMasker, NiftiLabelsMasker, MultiNiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
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
    elif atlas_name == "Schaefer":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
    elif atlas_name == 'MSDL':
        atlas = datasets.fetch_atlas_msdl()
    elif atlas_name == 'glasser360':
        atlas = {}
        atlas['img_path'] = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/glasser360/glasser360MNI.nii.gz'
        atlas['labels_path'] = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/glasser360/glasser360NodeNames.txt'
    elif atlas_name == 'SUIT':
        atlas = {}
        atlas['img_path'] = '/mfs/io/groups/dmello/projects/egcerebellum/code/connectivity/correlation/parcellations/atl-Anatom_space-MNI_dseg_resamplespace-MNI152NLin2009cAsym_res-2.nii'
        atlas['lut_path'] = '/mfs/io/groups/dmello/projects/egcerebellum/code/connectivity/correlation/parcellations/atl-Anatom.tsv'
    elif atlas_name == 'Morel_Left_Global_Thalamus':
        atlas = {}
        atlas['img_path'] = '/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/MorelAtlasMNI152/left-vols-1mm/global.nii.gz'
    elif atlas_name == 'Morel_Right_Global_Thalamus':
        atlas = {}
        atlas['img_path'] = '/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/MorelAtlasMNI152/right-vols-1mm/global.nii.gz'
    elif atlas_name == 'Morel_All':
        atlas = {}
        atlas['img_path'] = f'/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/MorelAtlasMNI152'

    else:
        raise ValueError("Atlas not recognized.")
    return atlas

def get_masker(atlas, atlas_name):

    if atlas_name == 'MSDL':
        masker = NiftiMapsMasker(
        atlas.maps,
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory="nilearn_cache",
        memory_level=1,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    )
    
    # double check TR
    elif atlas_name == 'HarvardOxford':
        masker = NiftiLabelsMasker(
        atlas.maps,
        labels=atlas.labels,
        resampling_target="data",
        t_r=2,
        detrend=True,
        standardize="zscore_sample",
    )
    
        
    elif atlas_name == 'Schaefer':
        masker = MultiNiftiLabelsMasker(
        labels_img=atlas.maps,  # Both hemispheres
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        n_jobs=2,
    )
    elif atlas_name == 'SUIT':
        masker = MultiNiftiLabelsMasker(
        labels_img = atlas['img_path'],  # Both hemispheres
        lut = atlas['lut_path'],
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        n_jobs=2,
    )
    
    elif atlas_name == 'glasser360':
        masker = MultiNiftiLabelsMasker(
        labels_img = atlas['img_path'],  # Both hemispheres
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        n_jobs=2,
    )
    
    elif atlas_name == 'Morel_Left_Global_Thalamus':
        masker = NiftiLabelsMasker(
        labels_img=atlas['img_path'],
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory="nilearn_cache",
        memory_level=1,
        standardize="zscore_sample",
        standardize_confounds=True,
        interpolation="nearest"
    )
    
    elif atlas_name == 'Morel_Right_Global_Thalamus':
        masker = NiftiLabelsMasker(
        labels_img=atlas['img_path'],
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory="nilearn_cache",
        memory_level=1,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        interpolation="nearest"
    )
    
    # collect all subregions in morel thalamic atlas as a dictionary of maskers
    elif atlas_name == 'Morel_All':

        maskers = {}
        for side in ['left', 'right']:
            path = f'{atlas['img_path']}/{side}-vols-1mm'
            for sub_atlas in os.listdir(path):

                if sub_atlas.endswith('.nii.gz'):
                    masker = NiftiLabelsMasker(
                        labels_img=f'{path}/{sub_atlas}',
                        resampling_target="data",
                        t_r=2,
                        detrend=True,
                        memory="nilearn_cache",
                        memory_level=1,
                        standardize="zscore_sample",
                        standardize_confounds="zscore_sample",
                        interpolation="nearest"
                    )

                    maskers[f'{side}_{sub_atlas.replace('.nii.gz', '')}'] = masker
            
        masker = maskers # rename dictionary of maskers
 
    else:
        raise ValueError("make sure atlas name is correct")

    return masker

# Function to construct the download URL
def construct_url(base_url, file_id, subject_id='MSC01', session='func01'):
    return f"{base_url}/s_sub-{subject_id}_ses-{session}_task-{file_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

# get confounds for image INCOMPLETE
def get_confounds(base_url, file_id, subject_id, session, strategy=["high_pass", "motion", "wm_csf"], motion="basic", wm_csf="basic", confounds_raw_suffix = '_desc-confounds_timeseries.tsv'):

    # extract confounds file name for msc
    confounds_path = f"{base_url}/sub-{subject_id}_ses-{session}_task-{file_id}{confounds_raw_suffix}"

    # confounds, sample_mask = load_confounds(
    #     confounds_file,
    #     strategy=strategy,
    #     motion=motion,
    #     wm_csf=wm_csf,
    # )

    # manually extract dataframe of compounds
    selected_cols = ['cosine00', 'cosine01', 'cosine02', 'cosine03', 'csf', 'rot_x', 'rot_y',
    'rot_z', 'trans_x', 'trans_y', 'trans_z', 'white_matter']
    confounds = pd.read_csv(
                    confounds_path,
                    sep='\t', 
                    on_bad_lines='skip',
                    encoding='latin-1',
                    engine='python',
                    # usecols=selected_cols,
                    header=0
                    )

    return confounds[selected_cols]

def get_sample_mask(mask_file='/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/mask20_no_eyeballs.nii'):

    return mask_file

# get pooled subject time series based on the atlas rois
def get_pooled(base_url, file_ids, masker, tasks='rest', atlas_name='glasser360', subject_id='MSC01', session='func01'):

    pooled_subject = []

    # use all sub atlases in a dictionary of maskers
    if atlas_name == 'Morel_All':

        # initialize dictionary to hold time series for each key
        pooled_subject_dict = {key: [] for key in masker.keys()}

        for file_id in file_ids:

            # construct the URL for the subject's fMRI data
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            # sample_mask = get_sample_mask()

            # extract time series for each subregion
            for key, sub_masker in masker.items():
                print("fitting sub atlas: ", key)

                try:
                    time_series = sub_masker.fit_transform(
                        file_url, 
                        confounds=confounds, 
                        # sample_mask=sample_mask
                        )
                    pooled_subject_dict[key].append(time_series)
                
                except Exception as e:
                    print(f"Error processing {file_id} with sub-atlas {key}: {e}")
                    continue
                
                # save individual just in case
                save_data(pooled_subject_dict[key], f"thalamus/{key}", [file_id], 'output/roi_time_series', tasks=tasks, subject_id=subject_id, session=session)
        
        # concatenate all subregion time series along the ROI axis
        pooled_subject = np.array(pooled_subject_dict[list(pooled_subject_dict.keys())[0]])  # initialize with first key
        for key in list(pooled_subject_dict.keys())[1:]:
            pooled_subject = np.concatenate((pooled_subject, np.array(pooled_subject_dict[key])), axis=2)

    # for all single self-contained atlases
    else:

        for file_id in file_ids:

            # construct the URL for the subject's fMRI data
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            # sample_mask = get_sample_masl()
            time_series = masker.fit_transform(
                file_url, 
                confounds=confounds, 
                # sample_mask=sample_mask
                )
            pooled_subject.append(time_series)
    
    return pooled_subject

# save pooled time series and shapes
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

# filter file ids based on subject and session
def filter_file_ids(file_ids, base_url, subject_id='MSC01', session='func01'):
    
    new_file_ids = []
    for file_id in file_ids:
        
        if file_id == '':
            continue
            
        url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
        
        # check if the file exists at the path given by url
        if not os.path.exists(url):
            print(f"File {file_id} not found at {url}, skipping...")
        
        else:
            new_file_ids.append(file_id)
    
    return new_file_ids


# extract cortical time series using the MSDL atlas
def extract_time_series(base_url, file_ids, atlas_name='MSDL', tasks="all_tasks", subject_id='MSC01', session='func01'):

    # fetch the atlas
    atlas = fetch_atlas(atlas_name)

    # create masker based on the atlas type
    masker = get_masker(atlas, atlas_name)  
    
    file_ids = filter_file_ids(file_ids, base_url, subject_id=subject_id, session=session)

    # get pooled subjects time series
    pooled_subject = get_pooled(base_url, file_ids, masker, atlas_name=atlas_name, subject_id=subject_id, session=session)

    # save the pooled data
    save_data(pooled_subject, atlas_name, file_ids, 'output/roi_time_series', tasks=tasks, subject_id=subject_id, session=session)

def main():

    # set the working directory to fmri_connectivity_trees root directory
    working_dir = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club'
    os.chdir(working_dir)
    
    
    try_rest = True # just try running the rest condition
    run_all_sessions = True # run all sessions if True or just try one
    tasks = "all_tasks"
    atlas_name = 'Morel_All'  # change this to the atlas you want to use

    subject_ids = [
                    "MSC01",
                    # "MSC02",
                    # "MSC03",
                    # "MSC04",
                    # "MSC05",
                    # "MSC06",
                    # "MSC07",
                    # "MSC08",
                    # "MSC09",
                    # "MSC10"
                    ]  # change this to the subjects you want to run
    session = "func01" # set if only running one session

    # CHANGE PATHS HERE
    # base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"

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
    
    
    for subject_id in subject_ids:
        if run_all_sessions:
            all_sessions = [
                            # 'func01', 
                            # 'func02', 
                            # 'func03', 
                            # 'func04', 
                            'func05', 
                            'func06', 
                            'func07', 
                            'func08', 
                            'func09', 
                            'func10'
                            ]
            for session in all_sessions:
                base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"
                print("Processing subject: ", subject_id, " session: ", session)
                extract_time_series(base_url, file_ids, subject_id=subject_id, atlas_name=atlas_name, session=session, tasks=tasks)
        else:
            base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"
            print("Processing subject: ", subject_id, " session: ", session)
            extract_time_series(base_url, file_ids, subject_id=subject_id, atlas_name=atlas_name, session=session, tasks=tasks)


if __name__ == "__main__":
    main()



