"""Main entry point for knowledge distillation"""
import argparse
from copy import deepcopy
import os
from tqdm import tqdm
import numpy as np
import options
import torch
import torch.nn as nn
import submitit
import nltk

from NLQ.model.DeepVSLNet_cbkd import build_optimizer_and_scheduler, DeepVSLNet
from utils.cbkd_helpers import (
    freeze_module,
    unfreeze_module,
    run_cbkd_stage
)
from utils.cbkd_config import CBKDConfig
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

    teacher = DeepVSLNet(
        configs=configs, word_vectors=dataset.get("word_vector", None)
    ).to(device)
    # (Optionally load pretrained teacher checkpoint here)
    # teacher.load_state_dict(torch.load(configs.teacher_ckpt_path))

    # Bottom‐up Stage‐by‐stage distillation
    cbkd_config = CBKDConfig()
    distilled_blocks = {}
    total_blocks    = 4

    student_i = None
    for stage_idx in [4, 3, 2, 1]:
        pruned_block_i, student_i = run_cbkd_stage(
            teacher          = teacher,
            distilled_blocks = distilled_blocks,
            stage_idx        = stage_idx,
            cbkd_cfg         = cbkd_config,   # renamed
            train_loader     = train_loader,
            val_loader       = val_loader,
            total_blocks     = total_blocks,
            device           = device
        )
        # Save the newly‐pruned block into our dict
        distilled_blocks[stage_idx] = pruned_block_i

    # At this point, `student_i` is the model after Stage 4.
    # Block 1–4 are all pruned; Block 4 and predictor were just trained, others are frozen.
    # Final “Thawing” Stage (Stage N+1)
    if cbkd_config.finetune_all:
        print("\n>>> Starting final Thawing Stage (unfreeze all blocks + predictor) <<<\n")

        # Unfreeze every block and the predictor head
        for b_idx in range(1, total_blocks + 1):
            unfreeze_module(getattr(student_i, f"block{b_idx}"))
        unfreeze_module(student_i.block4["predictor"])

        # Build a new AdamW optimizer over _all_ parameters (with decay splitting)
        no_decay       = ["bias", "layer_norm", "LayerNorm"]
        decay_params   = []
        nodecay_params = []
        for n, p in student_i.named_parameters():
            if any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_ft = torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": 0.01},
                {"params": nodecay_params, "weight_decay": 0.0}
            ],
            lr=cbkd_config.lr_finetune
        )

        # (Optional) If you want a scheduler in the thaw, add it here.
        #       Otherwise, we skip scheduling and just train for the few epochs.
        scheduler_ft = None
        # total_steps_ft = cbkd_config.epochs_finetune * len(train_loader)
        # scheduler_ft = get_linear_schedule_with_warmup(
        #     optimizer_ft,
        #     num_warmup_steps = int(total_steps_ft * cbkd_config.warmup_proportion),
        #     num_training_steps= total_steps_ft
        # )

        # 3.4) Training loop for thawing stage (highlight + CE only)
        for epoch in range(cbkd_config.epochs_finetune):
            student_i.train()
            running_loss = 0.0

            for (
                word_ids,
                char_ids,
                video_features,
                v_mask,
                q_mask,
                start_labels,
                end_labels
            ) in train_loader:
                # Move batch to device
                word_ids       = word_ids.to(device)
                char_ids       = char_ids.to(device)
                video_features = video_features.to(device)
                v_mask         = v_mask.to(device)
                q_mask         = q_mask.to(device)
                start_labels   = start_labels.to(device)
                end_labels     = end_labels.to(device)

                optimizer_ft.zero_grad()

                # Forward pass
                h_score, start_logits, end_logits = student_i(
                    word_ids, char_ids, video_features, v_mask, q_mask
                )

                # Compute head‐only losses
                loss_hl = student_i.block3["highlight_layer"].compute_loss(
                    h_score, start_labels, v_mask
                )
                loss_se = student_i.block4["predictor"].compute_cross_entropy_loss(
                    start_logits, end_logits, start_labels, end_labels
                )
                loss = loss_hl + loss_se
                loss.backward()
                optimizer_ft.step()
                if scheduler_ft is not None:
                    scheduler_ft.step()

                running_loss += loss.item() * video_features.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(
                f"[Thawing] Epoch {epoch+1}/{cbkd_config.epochs_finetune} | "
                f"Loss: {epoch_loss:.4f}"
            )

        # 3.5) Save the final student checkpoint
        torch.save(student_i.state_dict(), cbkd_config.student_save_path)
        print(f"\nFinal student model saved to {cbkd_config.student_save_path}\n")

    else:
        print("\nSkipping final Thawing Stage (cbkd_config.finetune_all=False)\n")

if __name__ == "__main__":
    configs, parser = options.read_command_line()
    main(configs, parser)
