% FIND functional/structural files
clear all;
%subjects = {'sub-MSC01', 'sub-MSC02'};
subjects = {'sub-MSC01', 'sub-MSC02', 'sub-MSC03', 'sub-MSC04', 'sub-MSC05', 'sub-MSC06', 'sub-MSC07', 'sub-MSC08', 'sub-MSC09', 'sub-MSC10'};

NSUBJECTS=length(subjects);
cwd='/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/code/functional_connectivity/midnight_scan_club/conn';
addpath('/mfs/io/groups/dmello/software/conn-fmri/conn_20b')
outlier_dir='/mfs/io/groups/dmello/projects/hypocerebellum/derivatives/outliers_05';
fmriprep_dir='/mfs/io/groups/dmello/projects/hypocerebellum/derivatives/fmriprep';
brain_mask='/mfs/io/groups/dmello/projects/hypocerebellum/code/conn/parcellations/mask20_no_eyeballs.nii';

% ROIs ???
roi_dir='/mfs/io/groups/dmello/projects/egcerebellum/derivatives/l1_output/sub-EG16/langloc_EG1621/froi_clust_SN_atl-EGLanglocParcels';
subj_langloc_roi=strcat(roi_dir, '/sub-EG1621_con_0021_froi.nii');


task='rest';
TR=2; % Repetition time

conditions={'rest'};
nconditions = length(conditions);
cond_timings={{{[0] [Inf]}}}; % in each cell (condition), each cell (session) contains an array of timings (the first array) and the duration (the second array).


FUNCTIONAL_FILE={};
STRUCTURAL_FILE={};
for nsub=1:NSUBJECTS
    funcfiles_currsubj = conn_dir(strcat(fmriprep_dir,'/', subjects{nsub}, '/func/','s_',subjects{nsub},'_task-', task,'_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'));
    FUNCTIONAL_FILE{nsub}{1} = funcfiles_currsubj;
    structfiles_currsubj = cellstr(conn_dir(strcat(fmriprep_dir,'/', subjects{nsub}, '/anat/',subjects{nsub},'*_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')));
    STRUCTURAL_FILE = [STRUCTURAL_FILE,structfiles_currsubj];
end

if rem(length(FUNCTIONAL_FILE),NSUBJECTS),error('mismatch number of functional files %n', length(FUNCTIONAL_FILE));end
if rem(length(STRUCTURAL_FILE),NSUBJECTS),error('mismatch number of anatomical files %n', length(FUNCTIONAL_FILE));end
nsessions=length(FUNCTIONAL_FILE{1});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% CONN-SPECIFIC SECTION: RUNS PREPROCESSING/SETUP/DENOISING/ANALYSIS STEPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Prepares batch structure
clear batch;
batch.filename=fullfile(cwd,'conn_control_task-rest_thresh-allthresh_roi-EGParcels_mask-wholebrainmask.mat');
batch.parallel.N = 0;

%------- SETUP & PREPROCESSING step -------                  
% misc
batch.Setup.isnew=1;
batch.Setup.done=1;
batch.Setup.overwrite=1;
batch.Setup.RT=TR;
batch.Setup.nsubjects=NSUBJECTS;
batch.Setup.acquisitiontype=1;
batch.Setup.secondarydataset.functionals_type=1;
batch.Setup.analyses=[1,2];
batch.Setup.voxelmask=1;
batch.Setup.voxelmaskfile=GM_mask;
batch.Setup.voxelresolution=1;
batch.Setup.analysisunits=2; % raw signals
batch.Setup.outputfiles=[0,1,1,1,1,0];

% Functional 
batch.Setup.functionals=repmat({{}},[1,NSUBJECTS]);       % Point to functional volumes for each subject/session
for nsub=1:NSUBJECTS
    for nses=1:nsessions
        batch.Setup.functionals{nsub}{nses}{1}=FUNCTIONAL_FILE{nsub}{nses};
    end
end

% Structural
batch.Setup.structurals=repmat({{}},[1,NSUBJECTS]);
for nsub=1:NSUBJECTS
    for nses=1:nsessions
        batch.Setup.structurals{nsub}{nses}{1}=STRUCTURAL_FILE{nsub}; 
    end
end 

% ROIs
batch.Setup.rois.names={'Grey Matter', 'White Matter', 'CSF', 'subj.langloc'};

for nroi=1:length(batch.Setup.rois.names)
    for nsub=1:NSUBJECTS
    % Add roi files for each subject
        if nroi == 1
            roi = conn_dir(strcat(fmriprep_dir,'/', subjects{nsub}, '/anat/',subjects{nsub},'*_run-1_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz'));
        elseif nroi == 2
            roi = conn_dir(strcat(fmriprep_dir,'/', subjects{nsub}, '/anat/',subjects{nsub},'*_run-1_space-MNI152NLin2009cAsym_res-2_label-WM_probseg.nii.gz'));
        elseif nroi == 3
            roi = conn_dir(strcat(fmriprep_dir,'/', subjects{nsub}, '/anat/',subjects{nsub},'*_run-1_space-MNI152NLin2009cAsym_res-2_label-CSF_probseg.nii.gz'));
        elseif nroi == 4
            roi = subj_langloc_roi;
        end
        for nses=1:nsessions
            batch.Setup.rois.files{nroi}{nsub}{nses}=cellstr(roi);
        end
    end
end

batch.Setup.rois.dimensions={1 5 5 1};
batch.Setup.rois.mask=[0 0 0 0];
batch.Setup.rois.subjectspecific=[1 1 1 0];
batch.Setup.rois.sessionspecific=[0 0 0 0];
batch.Setup.rois.multiplelabels=[0 0 0 1];
batch.Setup.rois.regresscovariates=[0 1 1 0];
batch.Setup.rois.unsmoothedvolumes=[1 1 1 1];
batch.Setup.rois.weighted=[0 0 0 0];

% Conditions
batch.Setup.conditions.names = conditions;
for ncond=1:nconditions
    for nsub=1:NSUBJECTS
        for nses=1:nsessions
            batch.Setup.conditions.onsets{ncond}{nsub}{nses} = cond_timings{ncond}{nses}{1};
            batch.Setup.conditions.durations{ncond}{nsub}{nses} = cond_timings{ncond}{nses}{2};
        end
    end
end

% Outliers
covariates = {'motion'};
ncovariates = length(covariates);
batch.Setup.covariates.names = covariates;
for ncov=1:ncovariates
    for nsub=1:NSUBJECTS
        for nses=1:nsessions
            batch.Setup.covariates.files{ncov}{nsub}{nses} = strcat(outlier_dir,'/', subjects{nsub}, '_task-', task,'_run-', int2str(nses), '_desc-confounds_timeseries_fd.txt');
        end
    end
end

%------- DENOISING step -------
batch.Denoising.done = 1;       
batch.Denoising.overwrite=1;
confounds_names = {'White Matter' 'CSF' 'motion'};
for ncond=1:nconditions
    curr_conf = strcat(['Effect of ', conditions{ncond}]);
    confounds_names{end+1} = curr_conf;
end
batch.Denoising.confounds.names=confounds_names;
batch.Denoising.filter=[0.01, 0.1];                 % frequency filter (band-pass values, in Hz)
batch.Denoising.despiking=0;
batch.Denoising.regbp=1;
batch.Denoising.detrending=1;


% Run all analyses before the first and second level analyses
global CONN_gui; CONN_gui.usehighres=true;
conn_batch(batch);

%------- FIRST-LEVEL ANALYSIS step ------- 
l1_names = {'roi-to-roi', 'SBC_01'};
l1_types = {1, 2};

% Rest Condition
for nl1names=1:length(l1_names)
    conn_batch( 'filename', batch.filename, ...
        'Analysis.name', l1_names{nl1names}, ...
        'Analysis.type', l1_types{nl1names}, ...
        'Analysis.measure', 1, ...
        'Analysis.weight', 2, ...
        'Analysis.modulation', 0, ...
        'Analysis.done', 1)
end

% ------- SECOND-LEVEL ANALYSIS step -------

rois = {'subj.langloc.Cerebellum_VI_CrusI_CrusII_VIIb_L', 'subj.langloc.Cerebellum_CrusII_VIIb_VIIIa_L', 'subj.langloc.Cerebellum_lateralCrusI_CrusII_L', 'subj.langloc.Cerebellum_VI_lateralVI_CrusI_L', 'subj.langloc.Cerebellum_lateralCrusI_CrusII_VIIb_VIIIa_L', 'subj.langloc.Cerebellum_IX_L', 'subj.langloc.Cerebellum_VI_CrusI_CrusII_R', 'subj.langloc.Cerebellum_lateralVI_CrusI_R', 'subj.langloc.Cerebellum_lateralCrusII_VIIb_R'};

% Rest Condition

% roi-to-voxel
for nroi=1:length(rois)
    conn_batch( 'filename', batch.filename, ...
    'Results.analysis_number', 'SBC_01', ...
    'Results.between_subjects.effect_names', {'AllSubjects'}, ...
    'Results.between_subjects.contrast', [1], ...
    'Results.between_conditions.effect_names', conditions, ...
    'Results.between_conditions.contrast', [1], ...
    'Results.between_sources.effect_names', {rois{nroi}}, ...
    'Results.between_sources.contrast', [1], ...
    'Results.display', 0)
end

% roi-to-roi
conn_batch( 'filename', batch.filename, ...
    'Results.analysis_number', 'roi-to-roi', ...
    'Results.between_subjects.effect_names', {'AllSubjects'}, ...
    'Results.between_subjects.contrast', [1], ...
    'Results.between_conditions.effect_names', conditions, ...
    'Results.between_conditions.contrast', [1], ...
    'Results.between_sources.effect_names', {rois{nroi}}, ...
    'Results.between_sources.contrast', [1], ...
    'Results.display', 0)