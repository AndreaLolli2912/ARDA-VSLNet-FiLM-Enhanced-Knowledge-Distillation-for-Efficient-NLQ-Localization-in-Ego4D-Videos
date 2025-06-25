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
    lr_block2: float = 0.0005
    lr_block3: float = 0.0005
    lr_block4: float = 0.0005
    lr_finetune: float = 0.0005

    # ── EPOCH COUNTS ────────────────────────────────────────────────────────────
    epochs_block2: int = 30
    epochs_block3: int = 15
    epochs_block4: int = 5
    epochs_finetune: int = 30

    # ── OTHER TRAINING FLAGS ───────────────────────────────────────────────────
    finetune_all: bool = True # whether to run the final full‐model tuning step