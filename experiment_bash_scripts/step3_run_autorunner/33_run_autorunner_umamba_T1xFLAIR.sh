#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DATA_DIR="$BASE_DIR/data/Umamba_data/nnUNet_raw"
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"
OUTPUT_FOLDER="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_raw"
OUTPUT_FOLDER_PP="/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_pp"
# Preprocessing
nnUNetv2_plan_and_preprocess -d 433 --verify_dataset_integrity

# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
nnUNetv2_train 433 3d_fullres all -tr nnUNetTrainerUMambaBot

nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 433 3d_fullres 1 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 433 3d_fullres 2 -tr nnUNetTrainerUMambaBot
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerUMambaBot

#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerUMambaBot -f 0 1 2 3

***Run inference like this:***

nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaBot -c 3d_fullres -p nnUNetPlans

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3/plans.json

# Inference
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 431 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta

nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot --disable_tta
nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot 
nnUNetv2_predict -i /path/to/input -o /path/to/output -d 431 -c 3d_fullres -tr nnUNetTrainerUMambaBot
