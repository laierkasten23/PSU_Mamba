#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DIR="/home/studenti/facchi/lia_masterthesis"
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/LIA"
datasetname="Dataset022_ChoroidPlexus_T1xFLAIR_sym_PHU" 


python3 "$BASE_DIR/phuse_thesis_2024/Code_general_functions/step3_run_AutoRunner.py" \
--work_dir "$BASE_DIR/phuse_thesis_2024/thesis_experiments/02_phusegplex/working_directory_T1xFLAIR_240825" \
--dataroot "$BASE_DATA_DIR/$datasetname" \
--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
--algos SwinUnetr128dice SwinUnetr128diceCE DynUnet128dice \
--templates_path_or_url "$BASE_DIR/phuse_thesis_2024/02_phusegplex_segmentation/DNN_models/algorithm_templates_yaml/"