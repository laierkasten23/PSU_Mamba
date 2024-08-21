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

lr=1e-2
fold=0
weight=model.pt

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 \
    train.py -fold $fold -train_num_workers 4 -interval 10 -num_samples 1 \
    -learning_rate $lr -max_epochs 1000 -task_id 01 -pos_sample_num 1 \
    -expr_name baseline -tta_val True -checkpoint $weight -multi_gpu True

python step3_run_AutoRunner.py  \
  --work_dir /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/01_aschoplex_from_scratch/monai_training/working_directory_2008_UNETRt128diceCE_AP \
  --dataroot /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP  \
  --json_path /home/linuxlia/Lia_Masterthesis/data/Dataset001_ChoroidPlexus_T1_sym_AP/dataset_train_val_pred.json  \
  --templates_path_or_url /home/linuxlia/Lia_Masterthesis/phuse_thesis_2024/01_aschoplex_from_scratch/DNN_models/algorithm_templates/ \
  --algos UNETR128diceCE  