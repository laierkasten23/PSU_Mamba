#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset033_ChoroidPlexus_T1_FLAIR_sym_PHU" 

nnUNetv2_plan_and_preprocess -d 331 --verify_dataset_integrity

nnUNetv2_train 331 3d_fullres all -tr nnUNetTrainerUMambaBot

