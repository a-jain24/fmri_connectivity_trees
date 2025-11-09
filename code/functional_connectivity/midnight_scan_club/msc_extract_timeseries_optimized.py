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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial

# fetch different atlases
def fetch_atlas(atlas_name, atlas_dir=None):

    if atlas_name == 'HarvardOxford':
        atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm', data_dir=atlas_dir)
    elif atlas_name == "Schaefer":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
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

def get_masker(atlas, atlas_name, use_memory_cache=True):

    # Set memory parameter based on flag
    memory_param = "nilearn_cache" if use_memory_cache else None

    if atlas_name == 'MSDL':
        masker = NiftiMapsMasker(
        atlas.maps,
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory=memory_param,
        memory_level=1,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
    )
    
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
        labels_img=atlas.maps,
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory=memory_param,
        n_jobs=2,
    )
    elif atlas_name == 'SUIT':
        masker = MultiNiftiLabelsMasker(
        labels_img = atlas['img_path'],
        lut = atlas['lut_path'],
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory=memory_param,
        n_jobs=2,
    )
    
    elif atlas_name == 'glasser360':
        masker = MultiNiftiLabelsMasker(
        labels_img = atlas['img_path'],
        resampling_target="data",
        detrend=True
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory=memory_param,
        n_jobs=2,
    )
    
    elif atlas_name == 'Morel_Left_Global_Thalamus':
        masker = NiftiLabelsMasker(
        labels_img=atlas['img_path'],
        resampling_target="data",
        t_r=2,
        detrend=True,
        memory=memory_param,
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
        memory=memory_param,
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
                        memory=memory_param,
                        memory_level=1,
                        standardize="zscore_sample",
                        standardize_confounds="zscore_sample",
                        interpolation="nearest"
                    )

                    maskers[f'{side}_{sub_atlas.replace('.nii.gz', '')}'] = masker
            
        masker = maskers
 
    else:
        raise ValueError("make sure atlas name is correct")

    return masker

# Function to construct the download URL
def construct_url(base_url, file_id, subject_id='MSC01', session='func01'):
    return f"{base_url}/s_sub-{subject_id}_ses-{session}_task-{file_id}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

# get confounds for image
def get_confounds(base_url, file_id, subject_id, session, strategy=["high_pass", "motion", "wm_csf"], motion="basic", wm_csf="basic", confounds_raw_suffix = '_desc-confounds_timeseries.tsv'):

    confounds_path = f"{base_url}/sub-{subject_id}_ses-{session}_task-{file_id}{confounds_raw_suffix}"

    selected_cols = ['cosine00', 'cosine01', 'cosine02', 'cosine03', 'csf', 'rot_x', 'rot_y',
    'rot_z', 'trans_x', 'trans_y', 'trans_z', 'white_matter']
    confounds = pd.read_csv(
                    confounds_path,
                    sep='\t', 
                    on_bad_lines='skip',
                    encoding='latin-1',
                    engine='python',
                    header=0
                    )

    return confounds[selected_cols]

def get_sample_mask(mask_file='/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/atlases/mask20_no_eyeballs.nii'):
    return mask_file

# OPTIMIZED: Process batches of sub-atlases sequentially (memory efficient)
def process_sub_atlas_batch(batch_items, file_url, confounds):
    """Process a batch of sub-atlases for one file."""
    results = []
    for key, masker_path, masker_config in batch_items:
        try:
            # Recreate masker in this process to avoid pickling issues
            masker = NiftiLabelsMasker(
                labels_img=masker_path,
                resampling_target="data",
                t_r=2,
                detrend=True,
                memory=None,  # No cache in workers to save memory
                memory_level=1,
                standardize="zscore_sample",
                standardize_confounds="zscore_sample",
                interpolation="nearest"
            )
            time_series = masker.fit_transform(file_url, confounds=confounds)
            results.append((key, time_series, None))
        except Exception as e:
            results.append((key, None, str(e)))
    return results

# Alternative: Sequential processing with progress tracking
def get_pooled_sequential(base_url, file_ids, masker, tasks='rest', atlas_name='glasser360', subject_id='MSC01', session='func01'):
    """Memory-efficient sequential processing with better progress tracking."""
    
    pooled_subject = []

    if atlas_name == 'Morel_All':
        pooled_subject_dict = {key: [] for key in masker.keys()}
        
        total_combinations = len(file_ids) * len(masker)
        current_progress = 0

        for file_id in file_ids:
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            
            print(f"\nProcessing {file_id}...")
            
            # Process each sub-atlas sequentially
            for idx, (key, sub_masker) in enumerate(masker.items(), 1):
                current_progress += 1
                print(f"  [{current_progress}/{total_combinations}] Processing sub-atlas: {key}")
                
                try:
                    time_series = sub_masker.fit_transform(file_url, confounds=confounds)
                    pooled_subject_dict[key].append(time_series)
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        # Concatenate all subregion time series along the ROI axis
        if pooled_subject_dict[list(pooled_subject_dict.keys())[0]]:
            pooled_subject = np.array(pooled_subject_dict[list(pooled_subject_dict.keys())[0]])
            for key in list(pooled_subject_dict.keys())[1:]:
                if pooled_subject_dict[key]:
                    pooled_subject = np.concatenate((pooled_subject, np.array(pooled_subject_dict[key])), axis=2)
        
        # Save individual sub-atlas data
        print("\nSaving individual sub-atlas data...")
        for key in pooled_subject_dict.keys():
            if pooled_subject_dict[key]:
                save_data(pooled_subject_dict[key], f"thalamus/{key}", file_ids, 'output/roi_time_series', 
                         tasks=tasks, subject_id=subject_id, session=session)

    else:
        for file_id in file_ids:
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            time_series = masker.fit_transform(file_url, confounds=confounds)
            pooled_subject.append(time_series)
    
    return pooled_subject

# OPTIMIZED: Chunked parallel processing (memory efficient)
def get_pooled_chunked(base_url, file_ids, masker, tasks='rest', atlas_name='glasser360', 
                       subject_id='MSC01', session='func01', chunk_size=4, n_jobs=2):
    """Process sub-atlases in small chunks to balance speed and memory usage."""
    
    pooled_subject = []

    if atlas_name == 'Morel_All':
        pooled_subject_dict = {key: [] for key in masker.keys()}
        
        # Prepare masker info for parallel processing
        masker_items = []
        for key, sub_masker in masker.items():
            masker_path = sub_masker.labels_img
            masker_items.append((key, masker_path, None))
        
        # Split into chunks
        chunks = [masker_items[i:i + chunk_size] for i in range(0, len(masker_items), chunk_size)]
        
        for file_id in file_ids:
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            
            print(f"\nProcessing {file_id} with {len(chunks)} chunks of ~{chunk_size} sub-atlases...")
            
            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                process_func = partial(process_sub_atlas_batch, file_url=file_url, confounds=confounds)
                chunk_results = list(executor.map(process_func, chunks))
            
            # Collect results from all chunks
            for chunk_result in chunk_results:
                for key, time_series, error in chunk_result:
                    if error:
                        print(f"  Error with {key}: {error}")
                        continue
                    pooled_subject_dict[key].append(time_series)
        
        # Concatenate all subregion time series
        if pooled_subject_dict[list(pooled_subject_dict.keys())[0]]:
            pooled_subject = np.array(pooled_subject_dict[list(pooled_subject_dict.keys())[0]])
            for key in list(pooled_subject_dict.keys())[1:]:
                if pooled_subject_dict[key]:
                    pooled_subject = np.concatenate((pooled_subject, np.array(pooled_subject_dict[key])), axis=2)
        
        # Save individual sub-atlas data
        print("\nSaving individual sub-atlas data...")
        for key in pooled_subject_dict.keys():
            if pooled_subject_dict[key]:
                save_data(pooled_subject_dict[key], f"thalamus/{key}", file_ids, 'output/roi_time_series', 
                         tasks=tasks, subject_id=subject_id, session=session)

    else:
        for file_id in file_ids:
            file_url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
            confounds = get_confounds(base_url, file_id, subject_id, session)
            time_series = masker.fit_transform(file_url, confounds=confounds)
            pooled_subject.append(time_series)
    
    return pooled_subject

def save_data(pooled_subject, atlas_name, file_ids, output_dir='output/roi_time_series', subject_id="MSC01", tasks="all_tasks", session='func01'):

    this_output_dir = f'{output_dir}/{subject_id}/{session}/{atlas_name}/{tasks}'

    os.makedirs(f'{this_output_dir}/pooled', exist_ok=True)
    os.makedirs(f'{this_output_dir}/shape', exist_ok=True)

    for task, file_id in zip(pooled_subject, file_ids):
        task = np.array(task)
        shape = task.shape

        np.savetxt(f'{this_output_dir}/pooled/{file_id}.csv', task, delimiter=',')
        np.savetxt(f'{this_output_dir}/shape/{file_id}.csv', shape, delimiter=',')

def filter_file_ids(file_ids, base_url, subject_id='MSC01', session='func01'):
    
    new_file_ids = []
    for file_id in file_ids:
        
        if file_id == '':
            continue
            
        url = construct_url(base_url, file_id, subject_id=subject_id, session=session)
        
        if not os.path.exists(url):
            print(f"File {file_id} not found at {url}, skipping...")
        else:
            new_file_ids.append(file_id)
    
    return new_file_ids


def extract_time_series(base_url, file_ids, atlas_name='MSDL', tasks="all_tasks", 
                       subject_id='MSC01', session='func01', processing_mode='sequential',
                       chunk_size=4, n_jobs=2):
    """
    Extract time series with different processing modes.
    
    Parameters:
    -----------
    processing_mode : str
        'sequential' - Memory efficient, slower (recommended for limited RAM)
        'chunked' - Balanced speed/memory (recommended for moderate RAM, ~32GB+)
        chunk_size : int (only for chunked mode)
        n_jobs : int (only for chunked mode, recommend 2-4 to avoid memory issues)
    """

    atlas = fetch_atlas(atlas_name)
    masker = get_masker(atlas, atlas_name, use_memory_cache=(processing_mode=='sequential'))
    
    file_ids = filter_file_ids(file_ids, base_url, subject_id=subject_id, session=session)

    if processing_mode == 'sequential':
        pooled_subject = get_pooled_sequential(base_url, file_ids, masker, atlas_name=atlas_name, 
                                              subject_id=subject_id, session=session)
    elif processing_mode == 'chunked':
        pooled_subject = get_pooled_chunked(base_url, file_ids, masker, atlas_name=atlas_name, 
                                           subject_id=subject_id, session=session, 
                                           chunk_size=chunk_size, n_jobs=n_jobs)
    else:
        raise ValueError("processing_mode must be 'sequential' or 'chunked'")

    atlas_name = "schaefer1000_17net"

    save_data(pooled_subject, atlas_name, file_ids, 'output/roi_time_series', tasks=tasks, 
             subject_id=subject_id, session=session)

def main():

    working_dir = '/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club'
    os.chdir(working_dir)
    
    try_rest = True
    run_all_sessions = True
    tasks = "all_tasks"
    atlas_name = 'Morel_All'
    
    # PROCESSING MODE OPTIONS:
    # 'sequential' - safest, uses least memory, slower
    # 'chunked' - faster, uses more memory, good compromise
    processing_mode = 'chunked'  # Change to 'sequential' if still getting memory errors
    chunk_size = 3  # Process 3 sub-atlases at a time (reduce if still getting killed)
    n_jobs = 2  # Number of parallel chunks (reduce to 1 if memory issues persist)

    subject_ids = [
        # 'MSC01', 
        # 'MSC02', 
        # 'MSC03', 
        # 'MSC04', 
        # 'MSC05', 
        # 'MSC06', 
        'MSC07', 
        # 'MSC08', 
        # 'MSC09', 
        # 'MSC10'
    ]
    session = "func01"

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
                'func01',
                'func02',
                'func03', 
                'func04', 
                'func05', 
                'func06', 
                'func07', 
                'func08', 
                'func09', 
                'func10'
                ]
            for session in all_sessions:
                base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"
                print(f"\n{'='*60}")
                print(f"Processing subject: {subject_id}, session: {session}")
                print(f"Mode: {processing_mode}, Chunk size: {chunk_size}, Jobs: {n_jobs}")
                print(f"{'='*60}")
                extract_time_series(base_url, file_ids, subject_id=subject_id, atlas_name=atlas_name, 
                                   session=session, tasks=tasks, processing_mode=processing_mode,
                                   chunk_size=chunk_size, n_jobs=n_jobs)
        else:
            base_url = f"/mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224/sub-{subject_id}/ses-{session}/func"
            print(f"\n{'='*60}")
            print(f"Processing subject: {subject_id}, session: {session}")
            print(f"{'='*60}")
            extract_time_series(base_url, file_ids, subject_id=subject_id, atlas_name=atlas_name, 
                               session=session, tasks=tasks, processing_mode=processing_mode,
                               chunk_size=chunk_size, n_jobs=n_jobs)


if __name__ == "__main__":
    main()