#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset033_ChoroidPlexus_T1_FLAIR_sym_PHU" 

# Preprocessing
nnUNetv2_plan_and_preprocess -d 331 --verify_dataset_integrity

# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
#nnUNetv2_train 331 3d_fullres all -tr nnUNetTrainerUMambaBot

# Inference
#nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 331 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta
