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


BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DIR="/home/studenti/facchi/lia_masterthesis"
path="/var/datasets/LIA/pazienti_test_affine"
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/LIA"
datasettype=ASCHOPLEX
datasetname="Dataset999_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym__AFFINE_TEST_AP" 
train_test_index_list="006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" 
train_test_index_list="002,003,006,007,009,010,011,012,013,014,015,017,018,019,020,021,022"
task_id=999

#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
#--path "$path" \
#--task_id "$task_id" \
#--task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym__AFFINE_TEST_AP' \
#--datasettype "$datasettype" \
#--train_test_index_list "$train_test_index_list" \
#--modality 'T1' 'FLAIR' \
#--use_single_label_for_bichannel True


BASE_DIR="/home/linuxlia/Lia_Masterthesis"
BASE_DIR="/home/studenti/facchi/lia_masterthesis"

mode=train_predict
BASE_DATA_DIR="$BASE_DIR/data"
BASE_DATA_DIR="/var/datasets/LIA"
datasettype=ASCHOPLEX 
train_val_ratio=1.0
num_folds=4
groups='/var/datasets/LIA/pazienti_test_affine/patients.json'

benchmark_dataroot="$BASE_DATA_DIR/reference_labels"
# Define the indices
indices=(1 2 3 4 5)

#python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step2_create_json_nnunetv2.py" \
#--mode $mode \
#--dataroot "$BASE_DATA_DIR/Dataset999_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym__AFFINE_TEST_AP" \
#--benchmark_dataroot $benchmark_dataroot \
#--datasettype $datasettype \
#--train_val_ratio $train_val_ratio \
#--groups $groups \
#--modality "['T1', 'FLAIR']" 

datasetname="Dataset888_ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_AP_affinetest" 

python3 "$BASE_DIR/phuse_thesis_2024/Code_general_functions/step3_run_AutoRunner.py" \
--work_dir "$BASE_DIR/phuse_thesis_2024/thesis_experiments/01_aschoplex_from_scratch/working_directory_T1_FLAIR_T1xFLAIRmask_affinetest_weird_excluded" \
--dataroot "$BASE_DATA_DIR/$datasetname" \
--json_path "$BASE_DATA_DIR/$datasetname/dataset_train_val_pred.json" \
--algos DynUnet128dice DynUnet128diceCE UNETR128diceCE \
--templates_path_or_url "$BASE_DIR/phuse_thesis_2024/01_aschoplex_from_scratch/DNN_models/algorithm_templates/"