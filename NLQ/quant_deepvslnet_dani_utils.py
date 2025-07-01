"""Main script to train/test models for Ego4D NLQ dataset with static quantization support.
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
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx, get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import nltk

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


def calibrate_model(model, data_loader, device, num_batches=100):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Calibrating model")):
            if i >= num_batches:
                break
            
            (
                _,
                vfeats,
                vfeat_lens,
                word_ids,
                char_ids,
                s_labels,
                e_labels,
                h_labels,
            ) = data
            
            # Prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            
            if hasattr(model.module, 'configs'):
                configs = model.module.configs
            else:
                configs = model.configs
                
            if configs.predictor == "bert":
                word_ids = {key: val.to(device) for key, val in word_ids.items()}
                query_mask = (
                    (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                    .float()
                    .to(device)
                )
            else:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                query_mask = (
                    (torch.zeros_like(word_ids) != word_ids).float().to(device)
                )
            
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            
            # Forward pass for calibration
            if configs.model_name in ["vslnet", "deepvslnet", "teachercbkd"]:
                _ = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            elif configs.model_name == "vslbase":
                _ = model(word_ids, char_ids, vfeats, video_mask, query_mask)


def prepare_model_for_quantization(model, example_inputs, backend="x86"):
    model.eval()
    
    # Get default quantization config
    qconfig_mapping = get_default_qconfig_mapping(backend)
    
    # Prepare model for quantization
    model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    
    return model_prepared


def quantize_model(model_prepared):
    model_quantized = convert_fx(model_prepared)
    return model_quantized


def get_example_inputs(train_loader, device, configs):
    for data in train_loader:
        (
            _,
            vfeats,
            vfeat_lens,
            word_ids,
            char_ids,
            s_labels,
            e_labels,
            h_labels,
        ) = data
        
        # Prepare features
        vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
        
        if configs.predictor == "bert":
            word_ids = {key: val.to(device) for key, val in word_ids.items()}
            query_mask = (
                (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                .float()
                .to(device)
            )
        else:
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            query_mask = (
                (torch.zeros_like(word_ids) != word_ids).float().to(device)
            )
        
        video_mask = convert_length_to_mask(vfeat_lens).to(device)
        
        return (word_ids, char_ids, vfeats, video_mask, query_mask)


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

    # train and test
    if configs.mode.lower() == "train":
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
        if configs.model_name == "vslnet":
            model = VSLNet(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)
    
        elif configs.model_name == "vslbase":
            model = VSLBase(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)

        elif configs.model_name == "deepvslnet":
            model = DeepVSLNet(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)
        elif configs.model_name == "teachercbkd":
            model = TeacherVSLNetCBDK(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)

        optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
        
        # Check if quantization is enabled
        use_quantization = getattr(configs, 'use_quantization', False)
        quantized_model = None
        
        if use_quantization:
            print("Preparing model for quantization...")
            # Get example inputs for FX graph tracing
            example_inputs = get_example_inputs(train_loader, device, configs)
            
            # Prepare model for quantization
            model_prepared = prepare_model_for_quantization(model, example_inputs)
            
            # Calibrate the model
            print("Calibrating model...")
            calibrate_model(model_prepared, train_loader, device, num_batches=50)
            
            # Convert to quantized model
            print("Converting to quantized model...")
            quantized_model = quantize_model(model_prepared)
            
            # Save quantized model
            quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
            torch.save(quantized_model.state_dict(), quantized_model_path)
            print(f"Quantized model saved to: {quantized_model_path}")
            
            # Use quantized model for training/inference
            model = quantized_model
        
        # start training
        best_metric = -1.0
        score_writer = open(
            os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        global_step = 0
        for epoch in range(configs.epochs):
            model.train()
            for data in tqdm(
                train_loader,
                total=num_train_batches,
                desc="Epoch %3d / %3d" % (epoch + 1, configs.epochs),
            ):
                global_step += 1
                (
                    _,
                    vfeats,
                    vfeat_lens,
                    word_ids,
                    char_ids,
                    s_labels,
                    e_labels,
                    h_labels,
                ) = data
                # prepare features
                vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                s_labels, e_labels, h_labels = (
                    s_labels.to(device),
                    e_labels.to(device),
                    h_labels.to(device),
                )
                if configs.predictor == "bert":
                    word_ids = {key: val.to(device) for key, val in word_ids.items()}
                    query_mask = (
                        (
                            torch.zeros_like(word_ids["input_ids"])
                            != word_ids["input_ids"]
                        )
                        .float()
                        .to(device)
                    )
                else:
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    query_mask = (
                        (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    )
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)
                
                # compute logits
                if configs.model_name in ["vslnet", "deepvslnet", "teachercbkd"]:
                    h_score, start_logits, end_logits = model(
                        word_ids, char_ids, vfeats, video_mask, query_mask
                    )
                elif configs.model_name == "vslbase":
                    start_logits, end_logits = model(
                        word_ids, char_ids, vfeats, video_mask, query_mask
                    )

                # Skip gradient computation for quantized models during inference
                if use_quantization and hasattr(model, 'training') and not model.training:
                    continue

                # compute loss
                loc_loss = model.compute_loss(
                    start_logits, end_logits, s_labels, e_labels
                )

                if configs.model_name in ["vslnet", "deepvslnet", "teachercbkd"]:
                    highlight_loss = model.compute_highlight_loss(
                    h_score, h_labels, video_mask
                    )
                    total_loss = loc_loss + configs.highlight_lambda * highlight_loss
                elif configs.model_name == "vslbase":
                    total_loss = loc_loss
                
                # compute and apply gradients (skip for quantized models)
                if not use_quantization:
                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), configs.clip_norm
                    )
                    optimizer.step()
                    scheduler.step()
                
                if configs.model_name in ["vslnet", "deepvslnet", "teachercbkd"]:
                    if writer is not None and global_step % configs.tb_log_freq == 0:
                        writer.add_scalar("Loss/Total", total_loss.detach().cpu(), global_step)
                        writer.add_scalar("Loss/Loc", loc_loss.detach().cpu(), global_step)
                        writer.add_scalar("Loss/Highlight", highlight_loss.detach().cpu(), global_step)
                        writer.add_scalar("Loss/Highlight (*lambda)", (configs.highlight_lambda * highlight_loss.detach().cpu()), global_step)
                        if not use_quantization:
                            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
                elif configs.model_name == "vslbase":
                    if writer is not None and global_step % configs.tb_log_freq == 0:
                        writer.add_scalar("Loss/Total", total_loss.detach().cpu(), global_step)
                        writer.add_scalar("Loss/Loc", loc_loss.detach().cpu(), global_step)
                        if not use_quantization:
                            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

                # evaluate
                if (
                    global_step % eval_period == 0
                    or global_step % num_train_batches == 0
                ):
                    model.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    result_save_path = os.path.join(
                        model_dir,
                        f"{configs.model_name}_{epoch}_{global_step}_preds.json",
                    )
                    # Evaluate on val, keep the top 3 checkpoints.
                    results, mIoU, (score_str, score_dict) = eval_test(
                        model=model,
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
                        model_save_path = os.path.join(
                            model_dir,
                            "{}_{}.t7".format(configs.model_name, global_step),
                        )
                        torch.save(model.state_dict(), model_save_path)
                        
                        # Save quantized version if using quantization
                        if use_quantization:
                            quantized_save_path = os.path.join(
                                model_dir,
                                "{}_quantized_{}.t7".format(configs.model_name, global_step),
                            )
                            torch.save(model.state_dict(), quantized_save_path)
                        
                        # only keep the top-3 model checkpoints
                        filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                    model.train()
            
        score_writer.close()
        
    elif configs.mode.lower() == "test":
        # Load and test quantized model
        if getattr(configs, 'use_quantization', False):
            print("Loading quantized model for testing...")
            # Load the original model first
            if configs.model_name == "deepvslnet":
                model = DeepVSLNet(
                    configs=configs, word_vectors=dataset.get("word_vector", None)
                ).to(device)
            # Add other model types as needed
            
            # Get example inputs for quantization
            example_inputs = get_example_inputs(test_loader, device, configs)
            
            # Prepare and quantize the model
            model_prepared = prepare_model_for_quantization(model, example_inputs)
            calibrate_model(model_prepared, test_loader, device, num_batches=10)
            quantized_model = quantize_model(model_prepared)
            
            # Load quantized weights if available
            quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
            if os.path.exists(quantized_model_path):
                quantized_model.load_state_dict(torch.load(quantized_model_path))
                print(f"Loaded quantized model from: {quantized_model_path}")
            
            model = quantized_model
        
        # Run evaluation with quantized model
        model.eval()
        result_save_path = os.path.join(model_dir, f"{configs.model_name}_test_preds.json")
        results, mIoU, (score_str, score_dict) = eval_test(
            model=model,
            data_loader=test_loader,
            device=device,
            mode="test",
            epoch=0,
            global_step=0,
            gt_json_path=configs.eval_gt_json,
            result_save_path=result_save_path,
            model_name=configs.model_name,
        )
        print("Test Results:")
        print(score_str, flush=True)


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
        if configs.slurm_wait:
            job.result()