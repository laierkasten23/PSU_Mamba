#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DIR="/home/studenti/facchi/lia_masterthesis"

mode=train_predict
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/LIA"
datasettype=ASCHOPLEX 
train_val_ratio=1.0
num_folds=4
groups='/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
groups='/var/datasets/LIA/pazienti/patients.json'


python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_T1_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--modality "['T1']"  
        
python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_FLAIR_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--modality "['FLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_T1xFLAIR_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--modality "T1xFLAIR" 

python "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_T1_FLAIR_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--modality "['T1', 'FLAIR']" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--modality "['T1', 'FLAIR']" 
