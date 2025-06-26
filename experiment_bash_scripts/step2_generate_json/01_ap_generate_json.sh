#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxuser/user"
BASE_DIR="/home/stud/facchi/user"

mode=train_predict
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/user"
datasettype=ASCHOPLEX 
train_val_ratio=1.0
num_folds=4
groups='/home/linuxuser/user/data/pazienti/patients.json' 
groups='/var/datasets/user/pazienti/patients.json'
benchmark_dataroot="$BASE_DATA_DIR/reference_labels"


python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset011_ChoroidPlexus_T1_sym_AP" \
--benchmark_dataroot $benchmark_dataroot \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--modality "['T1']"  
        
python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset011_ChoroidPlexus_FLAIR_sym_AP" \
--benchmark_dataroot $benchmark_dataroot \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--modality "['FLAIR']" 

python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset011_ChoroidPlexus_T1xFLAIR_sym_AP" \
--benchmark_dataroot $benchmark_dataroot \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--modality "T1xFLAIR" 


python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset011_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_AP" \
--benchmark_dataroot $benchmark_dataroot \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--modality "['T1', 'FLAIR']"

#------------------------------------------------------------
python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset011_ChoroidPlexus_T1_FLAIR_sym_AP" \
--benchmark_dataroot $benchmark_dataroot \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--modality "['T1', 'FLAIR']" 

 
