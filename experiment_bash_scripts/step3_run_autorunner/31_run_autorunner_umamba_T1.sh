#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset431_ChoroidPlexus_T1_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"
OUTPUT_FOLDER="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1"

# Preprocessing
nnUNetv2_plan_and_preprocess -d 431 --verify_dataset_integrity

# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
nnUNetv2_train 431 3d_fullres all -tr nnUNetTrainerUMambaBot

nnUNetv2_train 431 3d_fullres 0 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 431 3d_fullres 1 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 431 3d_fullres 2 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 431 3d_fullres 3 -tr nnUNetTrainerUMambaBot

# Inference
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 431 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta

nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot --disable_tta
nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot 
nnUNetv2_predict -i /path/to/input -o /path/to/output -d 431 -c 3d_fullres -tr nnUNetTrainerUMambaBot
