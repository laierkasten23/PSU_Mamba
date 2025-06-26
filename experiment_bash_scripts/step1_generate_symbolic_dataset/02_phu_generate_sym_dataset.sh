#!/bin/bash

# Define the base directory for the project
BASE_DIR="/home/linuxuser/user"
BASE_DIR="/home/stud/facchi/user"


task_id=22
datasettype=ASCHOPLEX
path="$BASE_DIR/data/pazienti"
path="/var/datasets/user/pazienti"
train_test_index_list="056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" 


python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id "$task_id" \
--task_name 'ChoroidPlexus_T1_sym_PHU' \
--datasettype "$datasettype" \
--train_test_index_list "$train_test_index_list" \
--modality "T1"

python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id "$task_id" \
--task_name 'ChoroidPlexus_FLAIR_sym_PHU' \
--datasettype "$datasettype" \
--train_test_index_list "$train_test_index_list" \
--modality "FLAIR"
 
    
python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id "$task_id" \
--task_name 'ChoroidPlexus_T1xFLAIR_sym_PHU' \
--datasettype "$datasettype" \
--train_test_index_list "$train_test_index_list" \
--modality "T1xFLAIR"

python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id "$task_id" \
--task_name 'ChoroidPlexus_T1_FLAIR_sym_PHU' \
--datasettype "$datasettype" \
--train_test_index_list "$train_test_index_list" \
--modality 'T1' 'FLAIR'
    
python3 "$BASE_DIR/project_dir/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--task_id "$task_id" \
--task_name 'ChoroidPlexus_T1_FLAIR_T1xFLAIRmask_sym_PHU' \
--datasettype "$datasettype" \
--train_test_index_list "$train_test_index_list" \
--modality 'T1' 'FLAIR' \
--use_single_label_for_bichannel True
    
    
    