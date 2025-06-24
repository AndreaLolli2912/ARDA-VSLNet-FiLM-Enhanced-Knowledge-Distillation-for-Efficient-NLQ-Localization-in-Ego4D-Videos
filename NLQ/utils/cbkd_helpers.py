"""Stuff for CBKD"""
from argparse import Namespace
from copy import deepcopy
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from utils.runner_utils import convert_length_to_mask
from utils.cbkd_config import CBKDConfig
from model.DeepVSLNet_cbkd import TeacherVSLNetCBDK
from model.layers import DepthwiseSeparableConvBlock

def freeze_module(module: nn.Module):
    """
    Recursively sets requires_grad=False on all parameters of the given module.
    """
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module: nn.Module):
    """
    Recursively sets requires_grad=True on all parameters of the given module.
    """
    for param in module.parameters():
        param.requires_grad = True

def make_pruned_ds_block(teacher_ds_block: nn.Module, keep_ratio: float) -> nn.Module:
    """
    Wraps a DepthwiseSeparableConvBlock(dim=128, …) so that internally it runs at
    floor(dim * keep_ratio) channels, but externally still takes/returns (batch, seq_len, 128).
    """

    # 1) Grab original DS parameters
    orig_dim    = teacher_ds_block.depthwise_separable_conv[0][0].in_channels
    kernel_size = teacher_ds_block.depthwise_separable_conv[0][0].kernel_size[0]
    drop_rate   = teacher_ds_block.dropout.p
    num_layers  = len(teacher_ds_block.depthwise_separable_conv)

    # 2) Compute new (smaller) “internal” channel count
    new_dim = max(1, int(orig_dim * keep_ratio))

    # 3) Build adapters + smaller DS block
    down_adapter = nn.Linear(orig_dim, new_dim)
    pruned_inner = DepthwiseSeparableConvBlock(
        dim=new_dim,
        kernel_size=kernel_size,
        drop_rate=drop_rate,
        num_layers=num_layers
    )
    up_adapter   = nn.Linear(new_dim, orig_dim)

    class PrunedDSWrapper(nn.Module):
        def __init__(self, down, inner, up):
            super().__init__()
            self.down  = down
            self.inner = inner
            self.up    = up

        def forward(self, x):
            # x: (batch, seq_len, orig_dim=128)
            z = self.down(x)      # → (batch, seq_len, new_dim)
            z = self.inner(z)     # → (batch, seq_len, new_dim)
            z = self.up(z)        # → (batch, seq_len, orig_dim=128)
            return z

    return PrunedDSWrapper(down_adapter, pruned_inner, up_adapter)

def make_pruned_conv1d(orig_conv: nn.Conv1d, keep_ratio: float) -> nn.Sequential:
    """
    Given a teacher Conv1d layer `orig_conv`, produce a pruned version that preserves
    the original input/output channel dimensions via 1x1 adapters.

    Steps:
     1) 1x1 down-projection from C_in -> C_in_kept
     2) KxK convolution from C_in_kept -> C_out_kept
     3) 1x1 up-projection from C_out_kept -> C_out

    Where:
      C_in_kept  = max(1, floor(orig_conv.in_channels * keep_ratio))
      C_out_kept = max(1, floor(orig_conv.out_channels * keep_ratio))

    Handles both standard and depthwise convolutions:
      - If orig_conv.groups == orig_conv.in_channels, it's depthwise → set groups=in_kept.
      - Otherwise, use groups=1 for the pruned middle conv.
    """
    C_in, C_out = orig_conv.in_channels, orig_conv.out_channels

    # Compute pruned channel counts
    in_kept  = max(1, int(C_in  * keep_ratio))
    out_kept = max(1, int(C_out * keep_ratio))

    # 1) 1×1 down-projection: C_in -> in_kept
    adapter_down = nn.Conv1d(C_in, in_kept, kernel_size=1)
    # Xavier‐init weights, zero‐init bias
    nn.init.xavier_uniform_(adapter_down.weight)
    nn.init.zeros_(adapter_down.bias)

    # 2) Pruned K×K convolution: in_kept -> out_kept
    #    If orig_conv was depthwise (groups == C_in), use groups=in_kept; otherwise groups=1.
    if orig_conv.groups == C_in:
        # Depthwise case (orig_conv was Conv1d(dim, dim, kernel_size=K, groups=dim))
        pruned_groups = in_kept
    else:
        # Standard or grouped conv → collapse to a single group
        pruned_groups = 1

    pruned_conv = nn.Conv1d(
        in_kept,
        out_kept,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        dilation=orig_conv.dilation,
        groups=pruned_groups,
        bias=(orig_conv.bias is not None)
    )

    # 3) 1×1 up-projection: out_kept -> C_out
    adapter_up = nn.Conv1d(out_kept, C_out, kernel_size=1)
    nn.init.xavier_uniform_(adapter_up.weight)
    nn.init.zeros_(adapter_up.bias)

    # Initialize pruned_conv weights by slicing teacher weights (optional)
    with torch.no_grad():
        W = orig_conv.weight.data  # shape: [C_out, C_in, K]
        b = orig_conv.bias.data if orig_conv.bias is not None else None

        # Copy the “top-left” in_kept × out_kept slice
        pruned_conv.weight.data.copy_(W[:out_kept, :in_kept, :])
        if orig_conv.bias is not None:
            pruned_conv.bias.data.copy_(b[:out_kept])

    return nn.Sequential(adapter_down, pruned_conv, adapter_up)

def prune_block2(
    teacher_featenc: nn.Module,
    keep_ratio_ds: float,
    keep_ratio_attn: float
) -> nn.Module:
    """
    Input:  teacher_featenc = FeatureEncoder(dim=128, …).
    Output: a pruned FeatureEncoder where:
      A) conv_block → make_pruned_ds_block(conv_block, keep_ratio_ds)
      B) each Conv1d in attention_block (query, key, value, out_layer)
         → make_pruned_conv1d(orig_conv, keep_ratio_attn)
      C) film_layer (Linear(128→128) x2) remains unchanged.

    Args:
      teacher_featenc   : a FeatureEncoder instance.
      keep_ratio_ds     : fraction of channels to keep in each DS layer (e.g. 0.25).
      keep_ratio_attn   : fraction of channels to keep in each attention conv (e.g. 0.25).

    Returns:
      A new, deep-copied FeatureEncoder with submodules replaced.
    """

    # 1) Deep‐copy
    pruned_featenc = deepcopy(teacher_featenc)

    # ── A) Prune the DS block ─────────────────────────────────────────────────
    orig_ds = pruned_featenc.conv_block  # DepthwiseSeparableConvBlock(dim=128,…)
    pruned_ds = make_pruned_ds_block(orig_ds, keep_ratio=keep_ratio_ds)
    pruned_featenc.conv_block = pruned_ds

    # ── B) Prune each Conv1d inside attention_block ────────────────────────────
    attn = pruned_featenc.attention_block
    for attr in ("query", "key", "value", "out_layer"):
        orig_layer = getattr(attn, attr)      # Conv1D wrapper
        orig_conv  = orig_layer.conv1d         # nn.Conv1d(128→128)
        pruned_conv = make_pruned_conv1d(orig_conv, keep_ratio=keep_ratio_attn)
        setattr(orig_layer, "conv1d", pruned_conv)
    
    return pruned_featenc

def prune_block3(
    teacher_block3: nn.Module,
    keep_ratio_cqa: float,
    keep_ratio_concat: float
) -> nn.Module:
    """
    Prune Block 3 by applying keep_ratio_cqa to CQAttention.cqa_linear.conv1d
    and keep_ratio_concat to CQConcatenate.conv1d.conv1d. Leave HighLightLayer.conv1d untouched.

    Args:
      teacher_block3     : Block 3 (ModuleDict with keys "cq_attention", "cq_concat", "highlight_layer")
      keep_ratio_cqa     : e.g. 0.5 to prune Conv1d(4*dim→dim) → (2*dim→dim→dim)
      keep_ratio_concat  : e.g. 0.5 to prune Conv1d(2*dim→dim) → (dim→dim→dim)

    Returns:
      A deep‐copied nn.Module where the specified Conv1d’s have been replaced by
      make_pruned_conv1d at the given keep_ratios.
    """

    pruned_block3 = deepcopy(teacher_block3)

    # 1) Prune CQAttention.cqa_linear.conv1d  (Conv1d(4*dim → dim))
    cq_attn = pruned_block3["cq_attention"]
    orig_cqa = cq_attn.cqa_linear.conv1d  # nn.Conv1d(4*dim → dim)
    pruned_cqa = make_pruned_conv1d(orig_cqa, keep_ratio=keep_ratio_cqa)
    cq_attn.cqa_linear.conv1d = pruned_cqa

    # 2) Prune CQConcatenate.conv1d.conv1d  (Conv1d(2*dim → dim))
    cq_concat = pruned_block3["cq_concat"]
    orig_concat = cq_concat.conv1d.conv1d  # nn.Conv1d(2*dim → dim)
    pruned_concat = make_pruned_conv1d(orig_concat, keep_ratio=keep_ratio_concat)
    cq_concat.conv1d.conv1d = pruned_concat

    # 3) Leave HighLightLayer.conv1d unchanged  (Conv1d(dim → 1))

    return pruned_block3

def prune_block4(
    teacher_block4: nn.Module,
    keep_ratio_enc: float,
    keep_ratio_pred: float
) -> nn.Module:
    """
    Prune Block 4 = ConditionedPredictor (transformer style).  This:
      A) Replaces `predictor.encoder` (a FeatureEncoder) with prune_block2(..., keep_ratio_enc, keep_ratio_enc)
      B) Replaces each Conv1D inside start_block and end_block via make_pruned_conv1d(..., keep_ratio_pred)

    Args:
      teacher_block4 : a ModuleDict containing "predictor" which is a ConditionedPredictor.
      keep_ratio_enc : e.g. 0.25 to shrink the internal FeatureEncoder from dim=128→32.
      keep_ratio_pred: e.g. 0.25 to shrink each head Conv1D (2·128→128 or 128→1).

    Returns:
      pruned_block4: a deep‐copied ModuleDict where the encoder and both head convs are replaced.
    """
    # 1) Deep‐copy so we don’t touch the teacher directly
    pruned_block4 = deepcopy(teacher_block4)
    predictor_mod = pruned_block4["predictor"]

    # ── A) Prune the internal FeatureEncoder inside ConditionedPredictor ───────
    #    (This is exactly the same logic as Block 2’s prune_block2, except we use
    #    keep_ratio_ds=keep_ratio_enc and keep_ratio_attn=keep_ratio_enc.)
    orig_encoder = predictor_mod.encoder  # a FeatureEncoder(dim=128,…)
    pruned_encoder = prune_block2(
        orig_encoder,
        keep_ratio_ds   = keep_ratio_enc,
        keep_ratio_attn = keep_ratio_enc
    )
    predictor_mod.encoder = pruned_encoder

    # ── B) Prune start_block’s two Conv1D layers ───────────────────────────────
    #    start_block = Sequential( Conv1D(2·dim→dim), ReLU, Conv1D(dim→1) )
    start_seq = predictor_mod.start_block
    #   - start_seq[0] is a Conv1D wrapper whose .conv1d is nn.Conv1d(2*128→128)
    #   - start_seq[2] is a Conv1D wrapper whose .conv1d is nn.Conv1d(128→1)
    orig_s1 = start_seq[0].conv1d
    orig_s2 = start_seq[2].conv1d
    start_seq[0].conv1d = make_pruned_conv1d(orig_s1, keep_ratio=keep_ratio_pred)
    start_seq[2].conv1d = make_pruned_conv1d(orig_s2, keep_ratio=keep_ratio_pred)

    # ── C) Prune end_block’s two Conv1D layers ─────────────────────────────────
    #    end_block = Sequential( Conv1D(2·dim→dim), ReLU, Conv1D(dim→1) )
    end_seq = predictor_mod.end_block
    orig_e1 = end_seq[0].conv1d
    orig_e2 = end_seq[2].conv1d
    end_seq[0].conv1d = make_pruned_conv1d(orig_e1, keep_ratio=keep_ratio_pred)
    end_seq[2].conv1d = make_pruned_conv1d(orig_e2, keep_ratio=keep_ratio_pred)

    return pruned_block4

def run_cbkd_stage(
    teacher: TeacherVSLNetCBDK,
    distilled_blocks: dict,
    stage_idx: int,
    configs: Namespace,
    cbkd_cfg: CBKDConfig,
    train_loader: torch.utils.data.DataLoader,
    total_blocks: int,
    device: torch.device
):
    """
    Perform a single CBKD stage (bottom-up), where `stage_idx` ∈ {4,3,2,1}:

     1) Take a fresh copy of the teacher as `student_i`.
     2) Replace teacher.block{stage_idx} with a pruned version.
     3) Freeze all blocks with index > stage_idx (pulled from distilled_blocks[j]).
     4) Unfreeze only the newly-pruned block_i (and the predictor head if stage_idx == 4).
     5) Build an AdamW optimizer (with weight‐decay grouping) over exactly those trainable parameters,
        using `lr_block{stage_idx}` from `cbkd_cfg`.
     6) Optionally build a linear‐warmup scheduler using `cbkd_cfg.warmup_proportion` if provided.
     7) Train *only* with head‐level losses (highlight + start/end CE). No intermediate MSE.
     8) (Optionally) perform a validation pass on `val_loader` at the end of each epoch.

    Args:
      teacher          : pretrained DeepVSLNet (on CPU or GPU).
      distilled_blocks : dict { block_idx → pruned+frozen nn.Module } for all idx > stage_idx.
      stage_idx        : which block to prune & train (4 → 3 → 2 → 1).
      cbkd_cfg         : CBKDConfig (contains keep‐ratios, per‐stage LRs, epochs, etc.).
      train_loader     : DataLoader of (word_ids, char_ids, video_feats, v_mask, q_mask, start_lbl, end_lbl).
      val_loader       : DataLoader for validation (same format as train_loader).
      total_blocks     : total number of blocks in teacher (4 for DeepVSLNet).
      device           : torch.device (e.g. torch.device("cuda") or torch.device("cpu")).

    Returns:
      pruned_block_i : the newly‐pruned block_i (frozen at the end of this stage).
      student_i      : the full student model after finishing this stage.
    """

    # 1) Start from a fresh copy of the teacher
    student_i = deepcopy(teacher).to(device)
    student_i.train()

    # 2) Freeze all blocks deeper than stage_idx, using the ones in distilled_blocks[j]
    for j in range(total_blocks, stage_idx, -1): # NOTE: DA SISTEMARE
        block_j = distilled_blocks[j]  # must already exist
        setattr(student_i, f"block{j}", block_j)
        freeze_module(getattr(student_i, f"block{j}"))
    
    # 3) Freeze all blocks shallower than stage_idx
    for j in range(1, stage_idx):
        print(j)
        freeze_module(getattr(student_i, f"block{j}"))

    # 3) Build (or copy) the pruned version of block_i
    if stage_idx == 4:
        orig_block4 = teacher.block4
        pruned_block_i = prune_block4(
            orig_block4,
            keep_ratio_enc  = cbkd_cfg.keep_ratio_block4_enc,
            keep_ratio_pred = cbkd_cfg.keep_ratio_block4_pred
        ).to(device)
        
    
    elif stage_idx == 3:
        orig_block3 = teacher.block3
        pruned_block_i = prune_block3(
            orig_block3,
            keep_ratio_cqa    = cbkd_cfg.keep_ratio_block3_cqa,
            keep_ratio_concat = cbkd_cfg.keep_ratio_block3_concat
        ).to(device)

    elif stage_idx == 2:
        orig_block2 = teacher.block2["feature_encoder"]
        pruned_block_i = prune_block2(
            orig_block2,
            keep_ratio_ds   = cbkd_cfg.keep_ratio_block2_ds,
            keep_ratio_attn = cbkd_cfg.keep_ratio_block2_attn
        ).to(device)

    elif stage_idx == 1:
        # CBKD does not prune Block1—just copy it verbatim
        pruned_block_i = deepcopy(teacher.block1).to(device)

    else:
        raise ValueError(f"Invalid stage_idx {stage_idx}. Must be in [1..{total_blocks}].")

    
    # 3.1) Insert the pruned block_i into student_i
    setattr(student_i, f"block{stage_idx}", pruned_block_i)
    
    # 4) Unfreeze only pruned_block_i (and predictor if stage_idx == 4)
    unfreeze_module(pruned_block_i)
    

    # 5) Build optimizer over exactly the trainable parameters
    no_decay       = ["bias", "layer_norm", "LayerNorm"]
    decay_params   = []
    nodecay_params = []

    
    # Which parameters are trainable at this stage?
    trainable_params = list(pruned_block_i.parameters())

    prefix = f"block{stage_idx}."

    # Group them into “decay” vs “no_decay” as in AdamW
    for n, p in student_i.named_parameters():
        if not n.startswith(prefix):
            continue  # skip params outside the current pruned block
        
        if not p.requires_grad:
            continue  # skip frozen params

        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr = getattr(cbkd_cfg, f"lr_block{stage_idx}")
    )

    # 5.1) Build a linear‐warmup scheduler

    num_steps = getattr(cbkd_cfg, f"epochs_block{stage_idx}") * len(train_loader)
    if hasattr(cbkd_cfg, "warmup_proportion"):
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(num_steps * configs.warmup_proportion),
            num_training_steps= num_steps
        )
    else:
        scheduler = None

    # 6) Training loop for this stage (head‐only CBKD)
    global_step = 0
    for epoch in range(getattr(cbkd_cfg, f"epochs_block{stage_idx}")):
        student_i.train()

        for data in train_loader:
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
            video_mask = convert_length_to_mask(vfeat_lens).to(device)

            # Forward through full student_i
            h_score, start_logits, end_logits = student_i(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )

            # Compute head‐only losses (highlight + start/end CE)
            loc_loss = student_i.compute_loss(
                start_logits, end_logits, s_labels, e_labels
            )
            highlight_loss = student_i.compute_highlight_loss(
                h_score, h_labels, video_mask
            )
            total_loss = loc_loss + configs.highlight_lambda * highlight_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                student_i.parameters(), configs.clip_norm
            )  # clip gradient
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        print(f"Stage {stage_idx}, Epoch {epoch+1}/{getattr(cbkd_cfg, f'epochs_block{stage_idx}')}")

    # 7) Freeze pruned_block_i before returning
    freeze_module(pruned_block_i)

    return pruned_block_i, student_i
