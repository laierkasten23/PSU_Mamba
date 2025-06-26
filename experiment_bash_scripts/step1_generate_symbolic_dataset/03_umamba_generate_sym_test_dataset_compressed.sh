#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/stud/user/user"

datasettype=UMAMBA

# : Change the path to the new dataset you want to use as test dataset
path="/mnt/turing/user/pazienti"

path="/var/datasets/user/ANON_user_SM_2"
output_dir="/data1/user/Umamba_data/nnUNet_raw/"

skip_validation=True
test_data_only=True


###### 
python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id 733 \
--task_name 'ChP_preanalysis_SM02_T1xFLAIR_sym_UMAMBA' \
--test_data_only "$test_data_only" \
--datasettype "$datasettype" \
--output_dir "$output_dir" \
--skip_validation "$skip_validation" \
--fileending '.nii.gz' \
--modality "T1xFLAIR"

    
    
    