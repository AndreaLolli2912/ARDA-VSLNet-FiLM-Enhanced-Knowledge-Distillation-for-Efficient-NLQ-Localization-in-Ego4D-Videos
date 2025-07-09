from dataclasses import dataclass

@dataclass
class CBKDConfig:
    """
    All the “fixed” keep-ratios, learning rates, and epoch counts for CBKD stages.
    Edit these once here if you want to change the overall compression schedule.
    """

    # keep ratios
    # Block 2: ~25% parameter reduction → keep 75% channels internally
    keep_ratio_block2_ds:    float = 0.1
    keep_ratio_block2_attn:  float = 0.1

    # Block 3: ~50% parameter reduction → keep 50% channels internally
    keep_ratio_block3_cqa:    float = 0.111
    keep_ratio_block3_concat: float = 0.2

    # Block 4: ~60% parameter reduction → keep 40% channels internally
    keep_ratio_block4_enc:   float = 0.05
    keep_ratio_block4_pred:  float = 0.05

    # learning rates
    lr_block2: float = 0.0005
    lr_block3: float = 0.0005
    lr_block4: float = 0.0005
    lr_finetune: float = 0.0005

    # epochs count
    epochs_block2: int = 30
    epochs_block3: int = 15
    epochs_block4: int = 30
    epochs_finetune: int = 30

    # other
    finetune_all: bool = True 

    # loss hyperparameters
    temperature: float = 2
    alpha_kd: float = 1
    beta_hl_kd: float = 0.25
