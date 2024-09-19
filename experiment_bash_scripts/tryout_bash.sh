# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# train step 2, finetune with small learning rate
# please replace the weight variable into your actual weight

 
   
#BASE_DIR="/home/linuxlia/Lia_Masterthesis"
#BASE_DIR="/home/studenti/facchi/lia_masterthesis"
#BASE_DATA_DIR="$BASE_DIR/data"
#BASE_DATA_DIR="/var/datasets/LIA"
#datasetname="Dataset022_ChoroidPlexus_T1_sym_PHU" 
  
#python3 "$BASE_DIR/phuse_thesis_2024/Code_general_functions/step3_run_AutoRunner.py" \
#--work_dir "$BASE_DIR/phuse_thesis_2024/02_phusegplex_segmentation/monai_training/working_directory_T1_try_SwinUnetrCE_240828" \
#--dataroot "$BASE_DATA_DIR/$datasetname" \
#--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
#--algos SwinUnetr128diceCE \
#--templates_path_or_url "$BASE_DIR/phuse_thesis_2024/02_phusegplex_segmentation/DNN_models/algorithm_templates_yaml/" 

#!/bin/bash
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
#BASE_DIR="/home/studenti/facchi/lia_masterthesis"

mode=train_predict
BASE_DATA_DIR="$BASE_DIR/data"
#BASE_DATA_DIR="/var/datasets/LIA"
datasettype=ASCHOPLEX 
train_val_ratio=1.0
num_folds=4
groups='/home/linuxlia/Lia_Masterthesis/data/pazienti/patients.json' 
#groups='/var/datasets/LIA/pazienti/patients.json'
benchmark_dataroot="$BASE_DATA_DIR/reference_labels"


python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
--mode $mode \
--dataroot "$BASE_DATA_DIR/Dataset022_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_PHU" \
--datasettype $datasettype \
--train_val_ratio $train_val_ratio \
--num_folds $num_folds \
--groups $groups \
--benchmark_dataroot "$BASE_DATA_DIR/reference_labels" \
--json_dir "/home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/JSON_file_experiments" \
--modality "['T1', 'FLAIR']" 








 

