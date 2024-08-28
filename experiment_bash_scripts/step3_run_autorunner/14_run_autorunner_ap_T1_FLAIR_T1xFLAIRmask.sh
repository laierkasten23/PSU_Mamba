#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DIR="/home/studenti/facchi/lia_masterthesis"
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/LIA"
datasetname="Dataset011_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_AP" 


python3 "$BASE_DIR/phuse_thesis_2024/Code_general_functions/step3_run_AutoRunner.py" \
--work_dir "$BASE_DIR/phuse_thesis_2024/thesis_experiments/01_aschoplex_from_scratch/working_directory_T1_FLAIR_T1xFLAIRmask_240826" \
--dataroot "$BASE_DATA_DIR/$datasetname" \
--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
--algos DynUnet128dice DynUnet128diceCE UNETR128diceCE \
--templates_path_or_url "$BASE_DIR/phuse_thesis_2024/01_aschoplex_from_scratch/DNN_models/algorithm_templates/"