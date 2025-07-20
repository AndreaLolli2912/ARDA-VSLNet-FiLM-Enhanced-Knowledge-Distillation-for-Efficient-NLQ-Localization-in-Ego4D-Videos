"""Main script to train/test models for Ego4D NLQ dataset.
"""
import argparse
import os
from tqdm import tqdm
import numpy as np
import options
import torch
import torch.nn as nn
from thop import profile
import submitit
from torch.utils.tensorboard.writer import SummaryWriter
import nltk

from model.DeepVSLNet import build_optimizer_and_scheduler, DeepVSLNet
from model.LightVSLNet import LightVSLNet
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

    feature_map_weight = configs.feature_map_weight
    ce_loss_weight = configs.ce_loss_weight
    weight_highlight_distillation_loss = configs.weight_highlight_distillation_loss
    # temperature
    T = 2

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

    mse_loss = nn.MSELoss()
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
        teacher = DeepVSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        student = LightVSLNet(
            configs=configs, word_vectors=dataset.get("word_vector", None)
        ).to(device)

        optimizer, scheduler = build_optimizer_and_scheduler(student, configs=configs)
        # start training
        best_metric = -1.0
        score_writer = open(
            os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )
        print("start training...", flush=True)
        global_step = 0
        model_dir_teacher = configs.model_dir_teacher
        filename = get_last_checkpoint(model_dir_teacher, suffix="t7")
        teacher.load_state_dict(torch.load(filename))

        for epoch in range(configs.epochs):
            teacher.eval()
            student.train()
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
                    # print(f"{configs.predictor=}")
                    word_ids = {key: val.to(device) for key, val in word_ids.items()}
                    # generate mask
                    query_mask = (
                        (
                            torch.zeros_like(word_ids["input_ids"])
                            != word_ids["input_ids"]
                        )
                        .float()
                        .to(device)
                    )
                else:
                    # print(f"{configs.predictor=}")
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    # generate mask
                    query_mask = (
                        (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    )
                # generate mask
                video_mask = convert_length_to_mask(vfeat_lens).to(device)
                
                with torch.no_grad():
                    h_score_teacher, start_logits_teacher, end_logits_teacher = teacher(
                        word_ids, char_ids, vfeats, video_mask, query_mask
                    )
                
                h_score_student, start_logits_student, end_logits_student = student(
                    word_ids, char_ids, vfeats, video_mask, query_mask
                )

                # Loss supervision student vs GT (CrossEntropy)
                loc_loss = student.compute_loss(
                    start_logits_student, end_logits_student, s_labels, e_labels
                )

                # BCE for highlight (student vs GT)
                highlight_loss = student.compute_highlight_loss(
                    h_score_student, h_labels, video_mask
                )

                # DISTILLATION START/END
                highlight_distill_loss = torch.nn.functional.binary_cross_entropy(
                    h_score_student, h_score_teacher, reduction='none'
                )
                video_mask = video_mask.type(torch.float32)
                highlight_distill_loss = torch.sum(highlight_distill_loss * video_mask) / (torch.sum(video_mask) + 1e-12)

                # === Start / End distillation (softmax / temperature) ===
                soft_targets_start = torch.softmax(start_logits_teacher / T, dim=-1)
                log_probs_start = torch.log_softmax(start_logits_student / T, dim=-1)
                teacher_start_loss = torch.sum(
                    soft_targets_start * (soft_targets_start.log() - log_probs_start)
                ) / (start_logits_student.size(0) * (T ** 2))

                soft_targets_end = torch.softmax(end_logits_teacher / T, dim=-1)
                log_probs_end = torch.log_softmax(end_logits_student / T, dim=-1)
                teacher_end_loss = torch.sum(
                    soft_targets_end * (soft_targets_end.log() - log_probs_end)
                ) / (end_logits_student.size(0) * (T ** 2))

                teacher_span_loss = teacher_start_loss + teacher_end_loss

                loss_span = feature_map_weight * teacher_span_loss + ce_loss_weight * loc_loss
                loss_qgh = configs.highlight_lambda * highlight_loss + weight_highlight_distillation_loss * highlight_distill_loss

                total_loss = loss_span + loss_qgh


                # compute and apply gradients
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    student.parameters(), configs.clip_norm
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
                    student.eval()
                    print(
                        f"\nEpoch: {epoch + 1:2d} | Step: {global_step:5d}", flush=True
                    )
                    result_save_path = os.path.join(
                        model_dir,
                        f"{configs.model_name}_{epoch}_{global_step}_preds.json",
                    )
                    # Evaluate on val, keep the top 3 checkpoints.
                    results, mIoU, (score_str, score_dict) = eval_test(
                        model=student,
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
                            student.state_dict(),
                            os.path.join(
                                model_dir,
                                "{}_{}.t7".format(configs.model_name, global_step),
                            ),
                        )
                        # only keep the top-3 model checkpoints
                        filter_checkpoints(model_dir, suffix="t7", max_to_keep=3)
                    student.train()

        if configs.compute_gflops:

            teacher.eval()
            student.eval()
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
                    student,
                    inputs=(word_ids_1, char_ids_1, vfeats_1, video_mask_1, query_mask_1)
                )

            teacher_gflops = 2 * teacher_macs  / 1e9
            student_gflops = 2 * student_macs  / 1e9

            student.train()

            print(f"Teacher GFLOPs: {teacher_gflops:.2f}")
            print(f"Student GFLOPs: {student_gflops:.2f}")
            print(f"Compute reduction: {100*(1-student_gflops/teacher_gflops):.2f}%")

        score_writer.close()

    elif configs.mode.lower() == "test":
        if not os.path.exists(model_dir):
            raise ValueError("No pre-trained weights exist")
        # load previous configs
        pre_configs = load_json(os.path.join(model_dir, "configs.json"))
        parser.set_defaults(**pre_configs)
        configs = parser.parse_args()
        # build model
        if configs.model_name == "vslnet":
            # print(f"{configs.model_name=}")
            student = VSLNet(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)
        
        elif configs.model_name == "vslbase":
            # print(f"{configs.model_name=}")
            student = VSLBase(
                configs=configs, word_vectors=dataset.get("word_vector", None)
            ).to(device)

        # get last checkpoint file
        filename = get_last_checkpoint(model_dir, suffix="t7")
        student.load_state_dict(torch.load(filename))
        student.eval()
        result_save_path = filename.replace(".t7", "_test_result.json")
        results, mIoU, score_str = eval_test(
            model=student,
            data_loader=test_loader,
            device=device,
            mode="test",
            result_save_path=result_save_path,
            model_name=configs.model_name,
        )
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

        # wait for it
        if configs.slurm_wait:
            job.result()
