#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxuser/user"
BASE_DIR="/home/stud/facchi/user"
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/user"
datasetname="Dataset022_ChoroidPlexus_T1_FLAIR_sym_PHU" 


python3 "$BASE_DIR/project_dir/Code_general_functions/step3_run_AutoRunner.py" \
--work_dir "$BASE_DIR/project_dir/_experiments/02_labgplex/working_directory_T1_FLAIR_240825" \
--dataroot "$BASE_DATA_DIR/$datasetname" \
--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
--algos SwinUnetr128dice SwinUnetr128diceCE DynUnet128dice \
--templates_path_or_url "$BASE_DIR/project_dir/02_labgplex_segmentation/DNN_models/algorithm_templates_yaml/"