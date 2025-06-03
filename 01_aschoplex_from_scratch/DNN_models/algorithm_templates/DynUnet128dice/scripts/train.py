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

import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Optional, Sequence, Union, Callable

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import monai
from monai import transforms
from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.utils import DEFAULT_FMT
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice, HausdorffDistanceMetric
from monai.utils import RankFilter, set_determinism
from torch.nn.modules.loss import _Loss 
from monai.auto3dseg.utils import datafold_read

from Code_general_functions.extract_reference_label import get_reference_label_path, get_reference_label_paths    

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"monai_default": {"format": DEFAULT_FMT}},
    "loggers": {
        "monai.apps.auto3dseg.auto_runner": {"handlers": ["file", "console"], "level": "DEBUG", "propagate": False}
    },
    "filters": {"rank_filter": {"()": RankFilter}},
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "runner.log",
            "mode": "a",  # append or overwrite
            "level": "DEBUG",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
    },
}



def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("amp")
    ckpt_path = parser.get_parsed_content("ckpt_path")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    determ = parser.get_parsed_content("determ")
    finetune = parser.get_parsed_content("finetune")
    fold = parser.get_parsed_content("fold")
    num_images_per_batch = parser.get_parsed_content("num_images_per_batch")
    num_iterations = parser.get_parsed_content("num_iterations")
    num_iterations_per_validation = parser.get_parsed_content("num_iterations_per_validation")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio = parser.get_parsed_content("overlap_ratio")
    patch_size_valid = parser.get_parsed_content("patch_size_valid")
    softmax = parser.get_parsed_content("softmax")
    num_workers = parser.get_parsed_content("num_workers")
    num_workers_val = parser.get_parsed_content("num_workers_val")

    train_transforms = parser.get_parsed_content("transforms_train")
    val_transforms = parser.get_parsed_content("transforms_validate")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    if determ:
        set_determinism(seed=0)

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1

    CONFIG["handlers"]["file"]["filename"] = parser.get_parsed_content("log_output_file")
    print("log_output_file: ", parser.get_parsed_content("log_output_file"))
    logging.config.dictConfig(CONFIG)
    print("CONFIG: ", CONFIG)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
    logger.debug(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.debug(f"World_size: {world_size}")
    logger.debug(f"Batch size: {num_images_per_batch}")

    datalist = ConfigParser.load_config_file(data_list_file_path)

    # Get reference label path if available, else throw error message
    if "data_benchmark_base_dir" in datalist:
        data_benchmark_base_dir = datalist["data_benchmark_base_dir"]
    else:
        raise ValueError("data_benchmark_base_dir not found in data_list_file_path")

    # Data loading
    # Added this instead:
    train_files, val_files = datafold_read(datalist=data_list_file_path, basedir=data_file_base_dir, fold=fold)
 
    # get list of all image paths of the train_files list 
    #str_imgs_train = [os.path.join(data_file_base_dir, item['image']) for item in train_files]
    str_imgs_train = [os.path.join(data_file_base_dir, item['image'][0] if isinstance(item['image'], list) else item['image'])for item in train_files if item['image']]
    str_imgs_val = [os.path.join(data_file_base_dir, item['image'][0] if isinstance(item['image'], list) else item['image']) for item in val_files if item['image']]
    #str_imgs_val = [os.path.join(data_file_base_dir, item['image']) for item in val_files]
    
    # T1xFLAIR img-seg comparison
    str_ref_seg_tr = get_reference_label_paths(str_imgs_train, data_benchmark_base_dir)
    str_ref_seg_val = get_reference_label_paths(str_imgs_val, data_benchmark_base_dir)
    
    # Add reference label to train_files and val_files
    for item, str_ref_t in zip(train_files, str_ref_seg_tr):
        item["ref_label"] = str_ref_t

    for item, str_ref_v in zip(val_files, str_ref_seg_val):
        item["ref_label"] = str_ref_v
    
    random.shuffle(train_files)
    
    if torch.cuda.device_count() > 1:
        train_files = partition_dataset(data=train_files, shuffle=False, num_partitions=world_size, even_divisible=True)[
            dist.get_rank()
        ]

    # Get just the image paths as list: TxFLAIR comparison
    logger.debug(f"Train_files: {len(train_files)}")

    
    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    logger.debug(f"validation_files: {len(val_files)}")

    if torch.cuda.device_count() >= 4:
        train_ds = monai.data.CacheDataset(
            data=train_files, 
            transform=train_transforms, 
            cache_rate=1.0, 
            num_workers=num_workers, 
            progress=False
        )
        val_ds = monai.data.CacheDataset(
            data=val_files, 
            transform=val_transforms, 
            cache_rate=1.0, 
            num_workers=num_workers_val, 
            progress=False
        )
    else:
        train_ds = monai.data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=float(torch.cuda.device_count()) / 4.0,
            num_workers=num_workers,
            progress=False,
        )
        val_ds = monai.data.CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=float(torch.cuda.device_count()) / 4.0,
            num_workers=num_workers_val,
            progress=False,
        )

    train_loader = DataLoader(train_ds, num_workers=2, batch_size=num_images_per_batch, shuffle=True)
    val_loader = DataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = parser.get_parsed_content("network")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
        )
        post_label = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(to_onehot=output_classes)])
    else:
        post_pred = transforms.Compose(
            [transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)]
        )

    loss_function = parser.get_parsed_content("loss")

    optimizer_part = parser.get_parsed_content("optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    num_epochs_per_validation = num_iterations_per_validation // len(train_loader)
    num_epochs_per_validation = max(num_epochs_per_validation, 1)
    num_epochs = num_epochs_per_validation * (num_iterations // num_iterations_per_validation)

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        logger.debug(f"num_epochs: {num_epochs}")
        logger.debug(f"num_epochs_per_validation: {num_epochs_per_validation}")
        logger.debug(f"num_iterations: {num_iterations}")
        logger.debug(f"num_iterations_per_validation: {num_iterations_per_validation}")

    lr_scheduler_part = parser.get_parsed_content("lr_scheduler", instantiate=False)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

    if finetune["activate"] and os.path.isfile(finetune["pretrained_ckpt_name"]):
        print("[info] fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        logger.debug("Fine-tuning pre-trained checkpoint {:s}".format(finetune["pretrained_ckpt_name"]))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
            logger.debug(f"Model loaded from {finetune['pretrained_ckpt_name']}")
        else:
            model.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
    else:
        logger.debug("Training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            logger.debug("Amp enabled")

    val_interval = num_epochs_per_validation
    best_metric = -1; best_hd_metric = 3000
    best_metric_epoch = -1; best_hd_metric_epoch = 3000
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes  # number of classes or channels 

    # Initialize the Hausdorff distance metric
    compute_hausdorff_distance_95 = HausdorffDistanceMetric(include_background=False, percentile=95)

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "Events"))

        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tmetric_ref\thd_metric\thd_metric_ref\tloss\tloss_ref\tlr\ttime\titer\n")

    start_time = time.time()
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_last_lr()[0]
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")
            print(f"learning rate is set to {lr}")
            logger.debug(f"Epoch {epoch + 1}/{num_epochs}")
            logger.debug(f"Learning rate is set to {lr}")


        model.train()
        epoch_loss = 0
        epoch_loss_t1xflair = 0 # also track loss for T1xFLAIR, not used for backpropagation
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        loss_torch_t1xflair = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels, ref_labels = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["ref_label"].to(device)
            # Extract the index of the image from the batch_data
            image_index = batch_data["subject_id"]

            for param in model.parameters():
                param.grad = None

            if amp:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs.float(), labels)
                    ref_loss = loss_function(outputs.float(), ref_labels).detach()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs.float(), labels)
                ref_loss = loss_function(outputs.float(), ref_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_t1xflair += ref_loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            loss_torch_t1xflair[0] += ref_loss.item()   # T1xFLAIR loss
            loss_torch_t1xflair[1] += 1.0
            epoch_len = len(train_loader)
            idx_iter += 1

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print(f"[{str(datetime.now())[:19]}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_loss_ref: {ref_loss.item():.4f}")
                logger.debug(
                    f"[{str(datetime.now())[:19]}] " + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_loss_ref: {ref_loss.item():.4f}, image_index: {image_index}"
                    )
                writer.add_scalar("Loss/train", loss.item(), epoch_len * epoch + step)
                writer.add_scalar("Loss/train_T1xFLAIR", ref_loss.item(), epoch_len * epoch + step)

        lr_scheduler.step()

        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        loss_torch_t1xflair = loss_torch_t1xflair.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            loss_torch_epoch_t1xflair = loss_torch_t1xflair[0] / loss_torch_t1xflair[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, average loss_T1xFLAIR: {loss_torch_epoch_t1xflair:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )
            logger.debug(
                    f"Epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, average loss_T1xFLAIR: {loss_torch_epoch_t1xflair:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

        del inputs, labels, ref_labels, outputs     # free up memory
        torch.cuda.empty_cache()

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
                hd_metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
                metric_sum = 0.0; hd_metric_sum = 0.0
                metric_count = 0; hd_metric_count = 0
                metric_mat = []; hd_metric_mat = []
                ref_metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
                ref_hd_metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
                ref_metric_sum = 0.0; ref_hd_metric_sum = 0.0
                ref_metric_count = 0; ref_hd_metric_count = 0
                ref_metric_mat = []; ref_hd_metric_mat = []
                val_images = None
                val_labels = None
                val_ref_labels = None
                val_outputs = None

                _index = 0
                logger.debug("Validation: computing dice scores")
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_ref_labels = val_data["ref_label"].to(device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        val_outputs = sliding_window_inference(
                            val_images,
                            patch_size_valid,
                            num_sw_batch_size,
                            model,
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]

                    if softmax:
                        val_labels = post_label(val_labels[0, ...])
                        val_labels = val_labels[None, ...]

                        val_ref_labels = post_label(val_ref_labels[0, ...])
                        val_ref_labels = val_ref_labels[None, ...]

                    dice_value = compute_dice(y_pred=val_outputs, y=val_labels, include_background=False)
                    ref_dice_value = compute_dice(y_pred=val_outputs, y=val_ref_labels, include_background=False)

                    hausdorff_value = compute_hausdorff_distance_95(y_pred=val_outputs, y=val_labels)
                    ref_hausdorff_value = compute_hausdorff_distance_95(y_pred=val_outputs, y=val_ref_labels)

                    print(_index + 1, "/", len(val_loader), "Dice Scores: ", dice_value, "Reference Dice Scores: ", ref_dice_value)
                    logger.debug(f"{_index + 1} / {len(val_loader)} 'dice_value': {dice_value}, 'reference dice_value': {ref_dice_value}") 
                    logger.debug(f"{_index + 1} / {len(val_loader)} 'hausdorff_value': {hausdorff_value}, 'reference hausdorff_value': {ref_hausdorff_value}")

                    metric_count += len(dice_value); ref_metric_count += len(ref_dice_value)
                    metric_sum += dice_value.sum().item(); ref_metric_sum += ref_dice_value.sum().item()
                    metric_vals = dice_value.cpu().numpy(); ref_metric_vals = ref_dice_value.cpu().numpy()
                    hd_metric_count += len(hausdorff_value); ref_hd_metric_count += len(ref_hausdorff_value)
                    hd_metric_sum += hausdorff_value.sum().item(); ref_hd_metric_sum += ref_hausdorff_value.sum().item()
                    hd_metric_vals = hausdorff_value.cpu().numpy(); ref_hd_metric_vals = ref_hausdorff_value.cpu().numpy()

                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)
                    if len(ref_metric_mat) == 0:
                        ref_metric_mat = ref_metric_vals
                    else:
                        ref_metric_mat = np.concatenate((ref_metric_mat, ref_metric_vals), axis=0)
                    if len(hd_metric_mat) == 0:
                        hd_metric_mat = hd_metric_vals
                    else:
                        hd_metric_mat = np.concatenate((hd_metric_mat, hd_metric_vals), axis=0)
                    if len(ref_hd_metric_mat) == 0:
                        ref_hd_metric_mat = ref_hd_metric_vals
                    else:
                        ref_hd_metric_mat = np.concatenate((ref_hd_metric_mat, ref_hd_metric_vals), axis=0)

                    # Iterate over the classes and compute the metric
                    # ensure that the metrics are robust to missing values and correctly accumulate the Dice coefficients for each class
                    # one for the sum of valid metric values and one for the count of valid entries
                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(dice_value[0, _c], nan=0.0) # replace NaN with 0
                        val1 = 1.0 - torch.isnan(dice_value[0, 0]).float()  # replace NaN with 1
                        # ensure that the metrics are accumulated correctly, even if some values are missing
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1
                    # Do the same for the reference labels
                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(ref_dice_value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(ref_dice_value[0, 0]).float()
                        ref_metric[2 * _c] += val0 * val1
                        ref_metric[2 * _c + 1] += val1
                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(hausdorff_value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(hausdorff_value[0, 0]).float()
                        hd_metric[2 * _c] += val0 * val1
                        hd_metric[2 * _c + 1] += val1
                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(ref_hausdorff_value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(ref_hausdorff_value[0, 0]).float()
                        ref_hd_metric[2 * _c] += val0 * val1   # access every even element in the tensor (starting from 0)
                        ref_hd_metric[2 * _c + 1] += val1      # access every odd element in the tensor (starting from 1)

                    _index += 1     # increment the index to move to the next batch

                if torch.cuda.device_count() > 1:   # synchronize the metrics across all processes
                    dist.barrier()  # wait for all processes to reach this point
                    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)  # sum the metrics across all processes
                    dist.all_reduce(ref_metric, op=torch.distributed.ReduceOp.SUM)
                    dist.all_reduce(hd_metric, op=torch.distributed.ReduceOp.SUM)
                    dist.all_reduce(ref_hd_metric, op=torch.distributed.ReduceOp.SUM)


                metric = metric.tolist(); hd_metric = hd_metric.tolist()
                ref_metric = ref_metric.tolist(); ref_hd_metric = ref_hd_metric.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    for _c in range(metric_dim):
                        logger.debug(f"Dice Evaluation metric - class {_c + 1}: {metric[2 * _c] / metric[2 * _c + 1]} Dice Reference metric - class {_c + 1}: {ref_metric[2 * _c] / ref_metric[2 * _c + 1]}")
                        logger.debug(f"HD Evaluation metric - class {_c + 1}: {hd_metric[2 * _c] / hd_metric[2 * _c + 1]} HD Reference metric - class {_c + 1}: {ref_hd_metric[2 * _c] / ref_hd_metric[2 * _c + 1]}")
                    avg_metric = 0; avg_hd_metric = 0
                    avg_metric_ref = 0; avg_hd_metric_ref = 0
                    for _c in range(metric_dim):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                        avg_metric_ref += ref_metric[2 * _c] / ref_metric[2 * _c + 1]
                        avg_hd_metric += hd_metric[2 * _c] / hd_metric[2 * _c + 1]
                        avg_hd_metric_ref += ref_hd_metric[2 * _c] / ref_hd_metric[2 * _c + 1]
                    avg_metric = avg_metric / float(metric_dim)
                    avg_metric_ref = avg_metric_ref / float(metric_dim)
                    avg_hd_metric = avg_hd_metric / float(metric_dim)
                    avg_hd_metric_ref = avg_hd_metric_ref / float(metric_dim)

                    logger.debug(f"Dice: Avg_metric: {avg_metric} Avg_metric_ref: {avg_metric_ref}")
                    logger.debug(f"Hausdorff: Avg_metric: {avg_hd_metric} Avg_metric_ref: {avg_hd_metric_ref}")

                    writer.add_scalar("Dice Accuracy/validation", avg_metric, epoch)
                    writer.add_scalar("Dice Accuracy/validation_ref", avg_metric_ref, epoch)
                    writer.add_scalar("Hausdorff/validation", avg_hd_metric, epoch)
                    writer.add_scalar("Hausdorff/validation_ref", avg_hd_metric_ref, epoch)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), os.path.join(ckpt_path, "best_metric_model.pt"))
                        else:
                            torch.save(model.state_dict(), os.path.join(ckpt_path, "best_metric_model.pt"))
                        print("saved new best metric model")
                        logger.debug("Saved new best metric model")

                        dict_file = {}
                        dict_file["best_avg_dice_score"] = float(best_metric)
                        dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                        dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                        with open(os.path.join(ckpt_path, "progress.yaml"), "a") as out_file:
                            yaml.dump([dict_file], stream=out_file)

                    if avg_hd_metric < best_hd_metric:
                        best_hd_metric = avg_hd_metric
                        best_hd_metric_epoch = epoch + 1
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), os.path.join(ckpt_path, "best_hd_metric_model.pt"))
                        else:
                            torch.save(model.state_dict(), os.path.join(ckpt_path, "best_hd_metric_model.pt"))
                        print("saved new best hausdorff metric model")
                        logger.debug("Saved new best hausdorff metric model")

                        dict_file = {}
                        dict_file["best_avg_hd_score"] = float(best_hd_metric)
                        dict_file["best_avg_hd_score_epoch"] = int(best_hd_metric_epoch)
                        dict_file["best_avg_hd_score_iteration"] = int(idx_iter)
                        with open(os.path.join(ckpt_path, "hd_progress.yaml"), "a") as out_file:
                            yaml.dump([dict_file], stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f}, current reference mean dice: {:.4f},  best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, avg_metric_ref, best_metric, best_metric_epoch
                        )
                    )
                    logger.debug(
                        "Current epoch: {} current mean dice: {:.4f}, current reference mean dice: {:.4f},  best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, avg_metric_ref, best_metric, best_metric_epoch
                        )
                    )
                    logger.debug(
                        "Current epoch: {} current mean hausdorff: {:.4f}, current reference mean hausdorff: {:.4f},  best mean hausdorff: {:.4f} at epoch {}".format(
                            epoch + 1, avg_hd_metric, avg_hd_metric_ref, best_hd_metric, best_hd_metric_epoch
                        )
                    )


                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
                        f.write(
                            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}\t{:d}\n".format(
                                epoch + 1, avg_metric, avg_metric_ref, avg_hd_metric, avg_hd_metric_ref, loss_torch_epoch, loss_torch_epoch_t1xflair, lr, elapsed_time, idx_iter
                            )
                        )

                if torch.cuda.device_count() > 1:
                    dist.barrier()

            torch.cuda.empty_cache()

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        logger.debug(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}.")

        writer.flush()
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()

    return best_metric


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
