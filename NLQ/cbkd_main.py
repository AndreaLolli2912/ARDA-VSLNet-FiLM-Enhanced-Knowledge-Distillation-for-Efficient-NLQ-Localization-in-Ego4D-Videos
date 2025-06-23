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
from torch.utils.tensorboard.writer import SummaryWriter
import nltk

from model.DeepVSLNet_cbkd import build_optimizer_and_scheduler, TeacherVSLNetCBDK
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
                "shallow_vslnet",
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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(
        vars(configs),
        os.path.join(model_dir, "configs.json"),
        sort_keys=True,
        save_pretty=True,
    )

    # build teacher model
    teacher = TeacherVSLNetCBDK(
        configs=configs, word_vectors=dataset.get("word_vector", None)
    ).to(device)

    # load pretrained teacher checkpoint here
    model_dir_teacher = configs.model_dir_teacher
    filename = get_last_checkpoint(model_dir_teacher, suffix="t7")
    teacher.load_state_dict(torch.load(filename))

    # Bottom‐up Stage‐by‐stage distillation
    cbkd_config = CBKDConfig()
    distilled_blocks = {}
    total_blocks    = 4

    student_i = None
    for stage_idx in [4, 3, 2, 1]:
        pruned_block_i, student_i = run_cbkd_stage(
            teacher           = teacher,
            distilled_blocks  = distilled_blocks,
            stage_idx         = stage_idx,
            configs           = configs,
            cbkd_cfg          = cbkd_config,   # renamed
            train_loader      = train_loader,
            total_blocks      = total_blocks,
            device            = device
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

        optimizer, scheduler = build_optimizer_and_scheduler(model=student_i, configs=configs)

        # 3.4) Training loop for thawing stage (highlight + CE only)
        best_metric = -1.0
        score_writer = open(
            os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        global_step = 0
        for epoch in range(cbkd_config.epochs_finetune):
            student_i.train()

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
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                # generate mask
                query_mask = (
                    (torch.zeros_like(word_ids) != word_ids).float().to(device)
                )
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)
                # compute logits

                h_score, start_logits, end_logits = student_i(
                    word_ids, char_ids, vfeats, video_mask, query_mask
                )

                # compute loss
                loc_loss = student_i.compute_loss(
                    start_logits, end_logits, s_labels, e_labels
                )

                highlight_loss = student_i.compute_highlight_loss(
                h_score, h_labels, video_mask
                )
                total_loss = loc_loss + configs.highlight_lambda * highlight_loss

                # compute and apply gradients
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    student_i.parameters(), configs.clip_norm
                )  # clip gradient
                optimizer.step()
                scheduler.step()

                if writer is not None and global_step % configs.tb_log_freq == 0:
                    writer.add_scalar("Loss/Total", total_loss.detach().cpu(), global_step)
                    writer.add_scalar("Loss/Loc", loc_loss.detach().cpu(), global_step)
                    writer.add_scalar("Loss/Highlight", highlight_loss.detach().cpu(), global_step)
                    writer.add_scalar("Loss/Highlight (*lambda)", (configs.highlight_lambda * highlight_loss.detach().cpu()), global_step)
                    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

                # evaluate
                if (
                    global_step % eval_period == 0
                    or global_step % num_train_batches == 0
                ):
                    student_i.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    result_save_path = os.path.join(
                        model_dir,
                        f"shallow_vslnet_{epoch}_{global_step}_preds.json",
                    )
                    # Evaluate on val, keep the top 3 checkpoints.
                    results, mIoU, (score_str, score_dict) = eval_test(
                        model=student_i,
                        data_loader=val_loader,
                        device=device,
                        mode="val",
                        epoch=epoch + 1,
                        global_step=global_step,
                        gt_json_path=configs.eval_gt_json,
                        result_save_path=result_save_path,
                        model_name="shallow_vslnet",
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
                            student_i.state_dict(),
                            os.path.join(
                                model_dir,
                                f"shallow_vslnet_{global_step}.t7",
                            ),
                        )
                        # only keep the top-3 model checkpoints
                        filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                    student_i.train()

        score_writer.close()

        # 3.5) Save the final student model and checkpoint
        # Save weights only (state_dict)
        # torch.save(student_i.state_dict(), cbkd_config.student_weights_path)
        # print(f"\nFinal student weights saved to {cbkd_config.student_weights_path}\n")

        torch.save(student_i.state_dict(), "content/prova")

        # Save full scripted model (architecture + weights)
        student_i.eval()  # switch to eval mode before scripting
        scripted_student = torch.jit.script(student_i)
        scripted_student.save("content/prova")
        print(f"\nFinal student scripted model saved to {cbkd_config.student_scripted_path}\n")

    else:
        print("\nSkipping final Thawing Stage (cbkd_config.finetune_all=False)\n")

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
