from dataclasses import dataclass

@dataclass
class CBKDConfig:
    """
    All the “fixed” keep-ratios, learning rates, and epoch counts for CBKD stages.
    Edit these once here if you want to change the overall compression schedule.
    """

    # ── KEEP RATIOS ────────────────────────────────────────────────────────────
    # Block 2: ~25% parameter reduction → keep 75% channels internally
    keep_ratio_block2_ds:    float = 0.30
    keep_ratio_block2_attn:  float = 0.37

    # Block 3: ~50% parameter reduction → keep 50% channels internally
    keep_ratio_block3_cqa:    float = 0.111
    keep_ratio_block3_concat: float = 0.2

    # Block 4: ~75% parameter reduction → keep 25% channels internally
    keep_ratio_block4_enc:   float = 0.001

    keep_ratio_block4_pred:  float = 0.001

    # ── LEARNING RATES ──────────────────────────────────────────────────────────
    lr_block1: float = 1e-4
    lr_block2: float = 1e-4   # LR for distilling (pruned) Block 2
    lr_block3: float = 5e-5   # LR for distilling Block 3
    lr_block4: float = 5e-5   # LR for distilling Block 4
    lr_finetune: float = 2e-5 # LR for final “thaw-all” pass, if used

    # ── EPOCH COUNTS ────────────────────────────────────────────────────────────
    epochs_block1: int = 1
    epochs_block2: int = 1    # how many epochs to train just Block 2
    epochs_block3: int = 1    # how many epochs to train just Block 3
    epochs_block4: int = 1    # how many epochs to train just Block 4
    epochs_finetune: int = 1  # how many epochs to fine-tune the entire student

    # ── OTHER TRAINING FLAGS ───────────────────────────────────────────────────
    finetune_all: bool = True # whether to run the final full‐model tuning step

    # ── CHECKPOINT PATHS & DATA SETTINGS ───────────────────────────────────────
    teacher_ckpt_path: str = "checkpoints/teacher.pt"
    student_save_path: str = "checkpoints/student_cbkd.pt"

    task: str = "nlq_official_v1"
    fv:   str = "official"
    max_pos_len: int = 128
    batch_size:  int = 32
    num_workers: int = 4

    # ── (You can add any additional “fixed” fields here, e.g. warmup proportion, seed, etc.) ──
