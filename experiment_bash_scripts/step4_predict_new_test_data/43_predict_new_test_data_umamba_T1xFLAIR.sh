#!/bin/bash

# Define the base directory for the project
BASE_DATA_DIR="/var/datasets/LIA/Umamba_data/nnUNet_raw"
datasetname="Dataset733_ChP_preanalysis_SM01_T1xFLAIR_sym_UMAMBA" 
datasetname="Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA" 
BASE_PRED_DIR="/home/studenti/lia/lia_masterthesis/phuse_thesis_2024/thesis_experiments/umamba_predictions"

# 1. Scp the new test data to the server (from local to server)
# scp -r /Users/liaschmid/Documents/Uni_Heidelberg/7_Semester_Thesis/Lia_SM_2 lia@hpc.phuselab.di.unimi.it:/var/datasets/LIA

# 2. Anonymize the new test data
python3 /home/studenti/lia/lia_masterthesis/phuse_thesis_2024/Code_data_preprocessing/step0_anonymize_dataset.py --path '/var/datasets/LIA/Lia_SM_2' --new_folder_name 'ANON_Lia_SM_2'
    
# 3. Convert the new test data to nnUNet format
# see script phuse_thesis_2024/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py, e.g. 
# python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" --path "$path" --task_id 733 --task_name 'ChP_preanalysis_SM02_T1xFLAIR_sym_UMAMBA' --test_data_only "$test_data_only" --datasettype "$datasettype" --output_dir "$output_dir" --skip_validation "$skip_validation" --fileending '.nii.gz' --modality "T1xFLAIR"

# TODO: Input path to test data you want to predict
INPUT_FOLDER="$BASE_DATA_DIR/$datasetname/imagesTs_PA2"

OUTPUT_FOLDER="$BASE_PRED_DIR/working_directory_preanalysis_SM02_T1xFLAIR/pred_raw/nnUNetPlans_64"
OUTPUT_FOLDER_PP="$BASE_PRED_DIR/working_directory_preanalysis_SM02_T1xFLAIR/pred_pp/nnUNetPlans_64"

# Create output folders if they do not exist
mkdir -p $OUTPUT_FOLDER
mkdir -p $OUTPUT_FOLDER_PP

# Preprocessing
#nnUNetv2_plan_and_preprocess -d 733 -c 3d_fullres -overwrite_plans_name nnUNetPlans_64 --verify_dataset_integrity

#nnUNetv2_plan_experiment -d 433 -c 3d_fullres



#nnUNetv2_train 332 3d_fullres all -tr nnUNetTrainerUMambaBot
#nnUNetv2_find_best_configuration 433 -c 3d_fullres -tr nnUNetTrainerUMambaBot -p nnUNetPlans_64 -f 0 1 2 3

***Run inference like this:***

nnUNetv2_predict -d $datasetname -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 -tr nnUNetTrainerUMambaBot -c 3d_fullres -p nnUNetPlans_64

***Once inference is completed, run postprocessing like this:***

nnUNetv2_apply_postprocessing -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER_PP -pp_pkl_file /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans_64__3d_fullres/crossval_results_folds_0_1_2_3/postprocessing.pkl -np 8 -plans_json /var/datasets/LIA/Umamba_data/nnUNet_results/Dataset433_ChoroidPlexus_T1xFLAIR_sym_UMAMBA/nnUNetTrainerUMambaBot__nnUNetPlans_64__3d_fullres/crossval_results_folds_0_1_2_3/plans.json

# Inference
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 431 -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta

nnUNetv2_predict -i /path/to/input -o /path/to/output -d 431 -c 3d_fullres -tr nnUNetTrainerUMambaBot
