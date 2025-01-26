#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/studenti/lia/lia_masterthesis"

mode=test
# BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
datasettype=UMAMBA
fileending=".nii.gz"

DATASET_NAME="Dataset733_ChP_preanalysis_SM01_T1xFLAIR_sym_UMAMBA"

benchmark_dataroot="/var/datasets/LIA/reference_labels"


python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
 --mode "test" \
 --datasettype $datasettype \
 --dataroot "$BASE_DATA_DIR/$DATASET_NAME" \
 --fileending $fileending \
 --groups_json_path None \
 --modality "['T1xFLAIR']"  
        

