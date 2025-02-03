#!/bin/bash

# Define the base directory for the project
BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs"

OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetPlans_wo_Mamba"
OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_pp/nnUNetPlans_wo_Mamba"

OUTPUT_FOLDER="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_raw/nnUNetPlans_Aug"
OUTPUT_FOLDER_PP="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1xFLAIR/pred_pp/nnUNetPlans_Aug"

# Preprocessing
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_enc --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_enc_z_scan --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_enc_y_scan --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_wo_Mamba --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_Aug --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_test --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 433 -c 3d_fullres -overwrite_plans_name nnUNetPlans_convtest --verify_dataset_integrity

nnUNetv2_plan_experiment -d 433 -c 3d_fullres

# Train 3D models using Mamba block in bottleneck (U-Mamba_Bot)
nnUNetv2_train 433 3d_fullres all -tr nnUNetTrainerUMambaBot

nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerUMambaBot -p nnUNetPlans_test --npz
nnUNetv2_train 433 3d_fullres 1 -tr nnUNetTrainerUMambaBot -p nnUNetPlans_64 --npz
nnUNetv2_train 433 3d_fullres 2 -tr nnUNetTrainerUMambaBot -p nnUNetPlans_64 --npz
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerUMambaBot -p nnUNetPlans_64 --npz
nnUNetv2_train 433 3d_fullres 1 -tr nnUNetTrainerUMambaBot -p nnUNetPlans_test --npz

nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerDA5 -p nnUNetPlans_test 
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerConvexHull -p nnUNetPlans_convtest 
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerConvexHullUMambaBot -p nnUNetPlans_convtest 
# wo Mamba 

nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans_wo_Mamba --npz



## Enc: 
nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerUMambaEnc -p nnUNetPlans_enc --npz
nnUNetv2_train 433 3d_fullres 3 -tr nnUNetTrainerUMambaEnc -p nnUNetPlans_enc_z_scan --npz 
nnUNetv2_train 433 3d_fullres 0 -tr nnUNetTrainerUMambaEnc -p nnUNetPlans_enc_y_scan --npz

#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerUMambaBot -p nnUNetPlans_64 -f 0 1 2 3
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerUMambaEnc -p nnUNetPlans_enc -f 0 1 2 3
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerUMambaEnc -p nnUNetPlans_enc_z_scan -f 0 1 2 3
nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainer -p nnUNetPlans_wo_Mamba -f 0 1 2 3

***Run inference like this:***

nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaBot -c 3d_fullres -p nnUNetPlans_64
nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaEnc -c 3d_fullres -p nnUNetPlans_enc
nnUNetv2_predict -d Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans_wo_Mamba

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans_64__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans_64__3d_fullres/crossval_results_folds_0_1_2_3/plans.json
nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaEnc__nnUNetPlans_enc__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaEnc__nnUNetPlans_enc__3d_fullres/crossval_results_folds_0_1_2_3/plans.json
nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainer__nnUNetPlans_wo_Mamba__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainer__nnUNetPlans_wo_Mamba__3d_fullres/crossval_results_folds_0_1_2_3/plans.json

# Inference
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 431 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta

nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot --disable_tta
nnUNetv2_predict -i "/home/linuxlia/Lia_Masterthesis/data/Umamba_data/nnUNet_raw/Dataset431_ChoroidPlexus_T1_sym_UMAMBA/imagesTs" -o "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions/working_directory_T1" -d 431 -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot 
nnUNetv2_predict -i /path/to/input -o /path/to/output -d 431 -c 3d_fullres -tr nnUNetTrainerUMambaBot
