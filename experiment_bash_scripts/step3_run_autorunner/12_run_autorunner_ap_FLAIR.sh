#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxuser/user"
BASE_DATA_DIR="$BASE_DIR/data"
datasetname="Dataset011_ChoroidPlexus_FLAIR_sym_AP" 


python3 "$BASE_DIR/project_dir/Code_general_functions/step3_run_AutoRunner.py" \
--work_dir "$BASE_DIR/project_dir/_experiments/01_aschoplex_from_scratch/working_directory_FLAIR_240823" \
--dataroot "$BASE_DATA_DIR/$datasetname" \
--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
--algos DynUnet128dice DynUnet128diceCE UNETR128diceCE \
--templates_path_or_url "$BASE_DIR/project_dir/01_aschoplex_from_scratch/DNN_models/algorithm_templates/"
    