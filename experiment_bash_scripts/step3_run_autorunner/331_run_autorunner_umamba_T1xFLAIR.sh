#!/bin/bash

# Define the base directory for the project
BASE_DATA_DIR="/data1/user/Umamba_data/nnUNet_raw"
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"

OUTPUT_FOLDER="/data1/user/Umamba_data/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetTrainerMambaFirstStem_PCA_32_with_patch_size_8"
OUTPUT_FOLDER_PP="/data1/user/Umamba_data/umamba_predictions/working_directory_T1xFLAIR/pred_pp/nnUNetTrainerMambaFirstStem_PCA_32_with_patch_size_8"

# Preprocessing
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_First_general_64 --verify_dataset_integrity 
nnUNetv2_plan_and_preprocess -d 299 -c 3d_fullres -overwrite_plans_name nnUNetPlans_First_general_128 --verify_dataset_integrity 


nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_64 --npz --c
nnUNetv2_train 433 3d_fullres 1 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_64 --npz
nnUNetv2_train 433 3d_fullres 2 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_64 --npz
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_64 --npz


nnUNetv2_train 299 3d_fullres 0 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_128 --npz --c # 
nnUNetv2_train 299 3d_fullres 1 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_128 --npz --c # 
nnUNetv2_train 299 3d_fullres 2 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_128 --npz --c # 
nnUNetv2_train 299 3d_fullres 3 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general_128 --npz --c # 

nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -p nnUNetPlans_First_general -f 0 1 2 3


***Run inference like this:***
nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerMambaFirstStem_PCA_PSU_Mamba -c 3d_fullres -p nnUNetPlans_First_general_128


