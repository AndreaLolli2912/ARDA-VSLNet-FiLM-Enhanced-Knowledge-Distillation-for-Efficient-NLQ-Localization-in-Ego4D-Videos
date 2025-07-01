"""Main script to train/test models for Ego4D NLQ dataset.
"""
import argparse
import os
from tqdm import tqdm
import numpy as np
import options
import torch
import torch.nn as nn
import submitit
from torch.utils.tensorboard.writer import SummaryWriter
import nltk

import torch.fx.experimental.proxy_tensor as pt
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from model.VSLNet import build_optimizer_and_scheduler, VSLNet
from model.VSLBase import VSLBase
from model.DeepVSLNet import DeepVSLNet
from model.DeepVSLNet_cbkd import TeacherVSLNetCBDK
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_test_loader, get_train_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)


from model.layers import FiLM

def calibrate(model, data_loader, num_batches):
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            (
                _, # metadata (unused)
                vfeats,
                vfeat_lens,
                word_ids,
                char_ids,
                *_,  # labels (unused)
            ) = data

            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            
            # generate mask
            query_mask = (
                (torch.zeros_like(word_ids) != word_ids).float().to(device)
            )
            # generate mask
            video_mask = convert_length_to_mask(vfeat_lens).to(device)

            # forward pass to collect observer stats
            model(word_ids, char_ids, vfeats, video_mask, query_mask)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # da pytorch stat quant
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

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
    test_loader = get_test_loader(
        dataset=dataset["test_set"], video_features=visual_features, configs=configs
    )
    configs.num_train_steps = len(train_loader) * configs.epochs
    num_train_batches = len(train_loader)

    # Device configuration
    cuda_str = "cuda" if configs.gpu_idx is None else "cuda:{}".format(configs.gpu_idx)
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

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

    # train
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(
        vars(configs),
        os.path.join(model_dir, "configs.json"),
        sort_keys=True,
        save_pretty=True,
    )
    # print(f"{configs.model_name=}")
    
    model_to_quantize = DeepVSLNet(
        configs=configs, word_vectors=dataset.get("word_vector", None)
    )
    model_directory_weight = configs.model_dir_teacher
    filename = get_last_checkpoint(model_directory_weight, suffix="t7")
    model_to_quantize.load_state_dict(torch.load(filename))
    model_to_quantize.to("cpu")
    model_to_quantize.eval()

    optimizer, scheduler = build_optimizer_and_scheduler(model_to_quantize, configs=configs)
    # start training
    best_metric = -1.0
    score_writer = open(
        os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
    )
    
    qconfig = get_default_qconfig("x86")
    qconfig_mapping = (QConfigMapping()
                       .set_global(qconfig)
                       .set_object_type(nn.Embedding, None)
                       .set_object_type(nn.LayerNorm, None)
                       .set_object_type(nn.Softmax, None)
                       .set_object_type(nn.LSTM, None)
                       .set_object_type(FiLM, None)
                       .set_module_name("embedding_net", None)
                       .set_module_name("query_affine", None)
                    )
    example_inputs = (next(iter(train_loader))[0])

    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    num_calibration_batches = num_train_batches//4
    calibrate(prepared_model, train_loader, num_calibration_batches)
    quantized_model = convert_fx(prepared_model)
    
    global_step = 0
    for epoch in range(1):       
        quantized_model.eval()
        print(
            f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
        )
        result_save_path = os.path.join(
            model_dir,
            f"{configs.model_name}_{epoch}_{global_step}_preds.json",
        )
        # Evaluate on val, keep the top 3 checkpoints.
        results, mIoU, (score_str, score_dict) = eval_test(
            model=quantized_model,
            data_loader=val_loader,
            device=device,
            mode="val",
            epoch=epoch + 1,
            global_step=global_step,
            gt_json_path=configs.eval_gt_json,
            result_save_path=result_save_path,
            model_name=configs.model_name,
        )
        print(score_str, flush=True)
        if writer is not None:
            for name, value in score_dict.items():
                kk = name.replace("\n", " ")
                writer.add_scalar(f"Val/{kk}", value, global_step)

        score_writer.write(score_str)
        score_writer.flush()
        # Recall@1, 0.3 IoU overlap --> best metric.
        if results[0][0] >= best_metric:
            best_metric = results[0][0]
            torch.save(
                quantized_model.state_dict(),
                os.path.join(
                    model_dir,
                    "{}_{}.t7".format(configs.model_name, global_step),
                ),
            )
            # only keep the top-3 model checkpoints
            filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                
        
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
