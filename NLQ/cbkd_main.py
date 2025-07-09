"""Main entry point for knowledge distillation"""
import argparse
from copy import deepcopy
import os
from tqdm import tqdm
import numpy as np
import options
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
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
    # test_loader = get_test_loader(
    #     dataset=dataset["test_set"], video_features=visual_features, configs=configs
    # )
    cbkd_config = CBKDConfig()
    configs.num_train_steps = len(train_loader) * cbkd_config.epochs_finetune
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
    state_dict = torch.load(filename)
    state_dict.pop("linear_modulation.film_generator.weight", None)
    state_dict.pop("linear_modulation.film_generator.bias", None)   
    teacher.load_state_dict(state_dict)

    # Bottom‐up Stage‐by‐stage distillation
    student_i = None
    total_blocks = 4
    distilled_blocks = {}
    
    for stage_idx in [4, 3, 2]:
        pruned_block_i, student_i = run_cbkd_stage(
            teacher           = teacher,
            distilled_blocks  = distilled_blocks,
            stage_idx         = stage_idx,
            configs           = configs,
            cbkd_cfg          = cbkd_config,
            train_loader      = train_loader,
            total_blocks      = total_blocks,
            device            = device
        )
        # Save the newly‐pruned block into our dict
        distilled_blocks[stage_idx] = pruned_block_i

    # Final “Thawing” Stage (Stage N+1)
    print(">>> Starting final Thawing Stage (unfreeze all blocks + predictor) <<<", flush=True)

    # Unfreeze every block and the predictor head
    for b_idx in range(1, total_blocks + 1):
        unfreeze_module(getattr(student_i, f"block{b_idx}"))

    optimizer, scheduler = build_optimizer_and_scheduler(model=student_i, configs=configs)
    
    # print(student_i.block2, flush=True)
    # print(student_i.block3, flush=True)
    # print(student_i.block4, flush=True)

    # 3.4) Training loop for thawing stage (highlight + CE only)
    best_metric = -1.0
    score_writer = open(
        os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
    )

    T          = cbkd_config.temperature
    alpha_kd   = cbkd_config.alpha_kd
    beta_hl_kd = cbkd_config.beta_hl_kd
    lambda_hl  = configs.highlight_lambda

    print("start training...", flush=True)
    global_step = 0
    for epoch in range(cbkd_config.epochs_finetune):
        student_i.train()
        for data in tqdm(
            train_loader,
            total=num_train_batches,
            desc="Epoch %3d / %3d" % (epoch + 1, cbkd_config.epochs_finetune),
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

            if configs.compute_gflops:

                teacher.eval()
                student_i.eval()
                vfeats_1      = vfeats[:1]
                vfeat_lens_1  = vfeat_lens[:1]
                word_ids_1    = word_ids[:1] if not isinstance(word_ids, dict) else {k:v[:1] for k,v in word_ids.items()}
                char_ids_1    = char_ids[:1]
                video_mask_1  = video_mask[:1]
                query_mask_1  = query_mask[:1]
                with torch.no_grad():
                    teacher_macs, _  = profile(
                        teacher,
                        inputs=(word_ids_1, char_ids_1, vfeats_1, video_mask_1, query_mask_1)
                    )
                    student_macs, _  = profile(
                        student_i,
                        inputs=(word_ids_1, char_ids_1, vfeats_1, video_mask_1, query_mask_1)
                    )

                teacher_gflops = 2 * teacher_macs  / 1e9
                student_gflops = 2 * student_macs  / 1e9

                print(f"Teacher GFLOPs: {teacher_gflops:.2f}")
                print(f"Student GFLOPs: {student_gflops:.2f}")
                print(f"Compute reduction: {100*(1-student_gflops/teacher_gflops):.2f}%")

            # compute logits
            stu_h, stu_s, stu_e = student_i(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )
            # compute loss
            loc_loss = student_i.compute_loss(stu_s, stu_e, s_labels, e_labels)
            hl_loss  = student_i.compute_highlight_loss(stu_h, h_labels, video_mask)

            #  forward treacher
            with torch.no_grad():
                teacher.eval()
                tch_h, tch_s, tch_e = teacher(
                    word_ids, char_ids, vfeats, video_mask, query_mask
                )

            # KD helper
            def kd_kl(student_logits, teacher_logits):
                return F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax( teacher_logits / T, dim=-1),
                    reduction="batchmean"
                ) * (T ** 2)

            kd_start = kd_kl(stu_s, tch_s)
            kd_end   = kd_kl(stu_e, tch_e)
            kd_hl    = F.binary_cross_entropy_with_logits(
                        stu_h, torch.sigmoid(tch_h)
            )
            total_loss = (
                loc_loss +
                lambda_hl * hl_loss +
                alpha_kd  * (kd_start + kd_end + beta_hl_kd * kd_hl)
            )

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
                writer.add_scalar("Loss/Highlight", hl_loss.detach().cpu(), global_step)
                writer.add_scalar("Loss/Highlight (*lambda)", (configs.highlight_lambda * hl_loss.detach().cpu()), global_step)
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
                    f"{configs.model_name}_{epoch}_{global_step}_preds.json",
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
                        student_i.state_dict(),
                        os.path.join(
                            model_dir,
                            f"{configs.model_name}_{global_step}.t7",
                        ),
                    )
                    # only keep the top-3 model checkpoints
                    filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                student_i.train()

    score_writer.close()

    # 3.5) Save the final student model and checkpoint
    # Save weights only (state_dict)
    torch.save(
        student_i.state_dict(),
        os.path.join(
            model_dir,
            f"{configs.model_name}._{global_step}.t7",
            )
        )
        # Save full scripted model (architecture + weights)
        # student_i.eval()  # switch to eval mode before scripting
        # scripted_student = torch.jit.script(student_i)

        # scripted_student.save(
        #     os.path.join(model_dir, f"architecture_{configs.model_name}.pt"),
        # )

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
