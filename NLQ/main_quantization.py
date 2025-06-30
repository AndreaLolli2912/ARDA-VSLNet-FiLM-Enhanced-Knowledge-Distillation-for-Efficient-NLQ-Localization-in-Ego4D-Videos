"""Main entry point for model quantization"""
import os

import nltk
import numpy as np
import submitit
import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import options

from model.DeepVSLNet import DeepVSLNet, build_optimizer_and_scheduler
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_test_loader, get_train_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.quantization_helpers import apply_post_training_static_quantization
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)

def main(configs, parser):
    
    print(f"Running with {configs}", flush=True)

    # set tensorflow configs
    set_th_config(configs.seed)

    # prepare or load dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset.get("n_chars", -1)
    configs.word_size = dataset.get("n_words", -1)

    # get train and test loader
    visual_features = load_video_features(
        os.path.join("data", "features", configs.task, configs.fv), configs.max_pos_len
    )
    # If video agnostic, randomize the video features.
    if configs.video_agnostic:
        visual_features = {
            key: np.random.rand(*val.shape) for key, val in visual_features.items()
        }
    train_loader = get_train_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    val_loader = (
        None
        if dataset["val_set"] is None
        else get_test_loader(dataset["val_set"], visual_features, configs)
    )
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)

    # # Device configuration NOTE: The model quantization is supported for CPU only
    # cuda_str = "cuda" if configs.gpu_idx is None else "cuda:{}".format(configs.gpu_idx)
    # device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    # print(f"Using device={device}")
    device = "cpu"

    # create model dir
    home_dir = os.path.join(
        configs.model_dir,
        "_".join(
            [
                configs.model_name,
                configs.task,
                configs.fv,
                str(configs.max_pos_len),
                configs.predictor,
            ]
        ),
    )
    if configs.suffix is not None:
        home_dir = home_dir + "_" + configs.suffix
    model_dir = os.path.join(home_dir, "model")

    writer = None
    if configs.log_to_tensorboard is not None:
        log_dir = os.path.join(configs.tb_log_dir, configs.log_to_tensorboard)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Writing to tensorboard: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

    # train and test
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(
        vars(configs),
        os.path.join(model_dir, "configs.json"),
        sort_keys=True,
        save_pretty=True,
    )
    # build model
    model = DeepVSLNet(
        configs=configs, word_vectors=dataset.get("word_vector", None)
    ).to(device)

    # load weights
    model_dir_teacher = configs.model_dir_teacher
    filename = get_last_checkpoint(model_dir_teacher, suffix="t7")
    model.load_state_dict(torch.load(filename))

    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    
    # quantization
    model = apply_post_training_static_quantization(
        float_model=model,
        calibration_loader=train_loader,
        num_calibration_batches=500
    )

    score_writer = open(
        os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
    )
    print("start evaluation...", flush=True)
    model.eval()
    print(
        f"\nEpoch: {0 + 1:2d} | Step: {0:5d}", flush=True
    )
    result_save_path = os.path.join(
        model_dir,
        f"{configs.model_name}_{0}_{0}_preds.json",
    )
    results, mIoU, (score_str, score_dict) = eval_test(
        model=model,
        data_loader=val_loader,
        device=device,
        mode="val",
        epoch=0 + 1,
        global_step=0,
        gt_json_path=configs.eval_gt_json,
        result_save_path=result_save_path,
        model_name=configs.model_name,
    )
    print(score_str, flush=True)
    if writer is not None:
        for name, value in score_dict.items():
            kk = name.replace("\n", " ")
            writer.add_scalar(f"Val/{kk}", value, 0)
    score_writer.write(score_str)
    score_writer.flush()
    torch.save(
        model.state_dict(),
        os.path.join(
            model_dir,
            "{}_{}.t7".format(configs.model_name, 0),
        ),
    )
    model.train()
    score_writer.close()


def create_executor(configs):
    executor = submitit.AutoExecutor(folder=configs.slurm_log_folder)

    executor.update_parameters(
        timeout_min=configs.slurm_timeout_min,
        constraint=configs.slurm_constraint,
        slurm_partition=configs.slurm_partition,
        gpus_per_node=configs.slurm_gpus,
        cpus_per_task=configs.slurm_cpus,
    )
    return executor

if __name__ == "__main__":

    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    configs, parser = options.read_command_line()
    if not configs.slurm:
        main(configs, parser)
    else:
        executor = create_executor(configs)

        job = executor.submit(main, configs, parser)
        print("job=", job.job_id)

        # wait for it
        if configs.slurm_wait:
            job.result()
