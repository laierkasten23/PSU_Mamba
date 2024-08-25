# Define the base directory for the project
BASE_DIR="/home/linuxlia/Lia_Masterthesis"
#BASE_DIR="/home/studenti/facchi/lia_masterthesis"


task_id=11
datasettype=ASCHOPLEX
path="$BASE_DIR/data/pazienti"
#path="$BASE_DIR/data/pazienti_test_affine"
#path="/var/datasets/LIA/pazienti"
train_test_index_list="056,063,006,052,003,024,100,019,025,071,045,067,102,101,083,011,049,033,061,042,020,097,088,047,028,053,018,073,015,066,050,030,085,048,098,037,070,010,064,036,039,054,057,041,077,013,040,017,007,078,059,096,082,062,087,058,084,095,012,051,043,074,001,080,002,086,093,031,023,089,046,021,022,014,065,060,009" 


python3 "$BASE_DIR/phuse_thesis_2024/Code_data_preprocessing/step1_1_dataset_creator_symbolic.py" \
--path "$path" \
--train_test_index_list "$train_test_index_list" \
--datasettype 'reference' 
    

 
