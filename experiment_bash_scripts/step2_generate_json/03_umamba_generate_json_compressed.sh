#!/bin/bash

# Define the base directory for the project
# BASE_DIR="/home/linuxuser/user"
# BASE_DIR="/home/stud/facchi/user"
BASE_DIR="/home/stud/user/user"
BASE_DIR="/home/stud/user/projects"

mode=train_predict
# BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
BASE_DATA_DIR="/data1/user/Umamba_data/nnUNet_raw"
BASE_DATA_DIR="/data1/user/Umamba_data/nnUNet_raw"
datasettype=UMAMBA
fileending=".nii.gz"
# benchmark_dataroot="$BASE_DIR/data/reference_labels"
benchmark_dataroot="/var/datasets/user/reference_labels"
benchmark_dataroot="/data1/user/reference_labels"
# groups='/home/linuxuser/user/data/pazienti/patients.json' 
# groups='/var/datasets/user/pazienti/patients.json'
groups='/mnt/turing/user/pazienti/patients.json'
groups='/mnt/user/pazienti/patients.json'



#python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset431_ChoroidPlexus_T1_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1']"  
        
#python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset432_ChoroidPlexus_FLAIR_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['FLAIR']" 

python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" \
--datasettype $datasettype \
--fileending $fileending \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--groups $groups \
--modality "['T1xFLAIR']" 

#python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset434_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1', 'FLAIR']" 

#python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset435_ChoroidPlexus_T1_FLAIR_sym_UMAMBA" \
#--datasettype $datasettype \
#--fileending $fileending \
#--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
#--groups $groups \
#--modality "['T1', 'FLAIR']" 


#python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset994_Dir_z" \
#--datasettype $datasettype \
#--fileending $fileending \
#--modality "['T1xFLAIR']" 
