"""Quantization helper functions"""
import copy

import torch
from torch import nn
from torch.ao.quantization.qconfig import QConfig, float_qparams_weight_only_qconfig
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.observer import default_observer

from model.layers import Conv1D, Conv1DReLU
from model.QuantizedDeepVSLNet import QuantizedDeepVSLNet
from utils.runner_utils import convert_length_to_mask

def fuse_sequential_block(
    block: nn.Sequential,
    layers_to_fuse: list[str],
    inplace: bool = False
) -> nn.Sequential:
    """
    Fuse specified layers in a Sequential block.
    """
    return torch.quantization.fuse_modules(block, layers_to_fuse, inplace=inplace)

def fuse_modulelist_blocks(
    blocks: nn.ModuleList,
    fuse_map: list[str],
    inplace: bool = False
) -> nn.ModuleList:
    """
    Applies fusion to each nn.Sequential in a ModuleList using a fixed fuse pattern.
    """
    fused = []
    for block in blocks:
        block_copy = block if inplace else copy.deepcopy(block)
        fused_block = fuse_sequential_block(block_copy, fuse_map, inplace=inplace)
        fused.append(fused_block)
    return nn.ModuleList(fused)

def fuse_depthwise_separable_conv_block(conv_block, inplace=False):
    """
    Fuses the pointwise Conv1d + ReLU in a DepthwiseSeparableConvBlock.
    """
    conv_block_copy = conv_block if inplace else copy.deepcopy(conv_block)
    fuse_pattern = ['1', '2']  # pointwise conv + ReLU
    conv_block_copy.depthwise_separable_conv = fuse_modulelist_blocks(
        conv_block_copy.depthwise_separable_conv,
        fuse_map=fuse_pattern,
        inplace=inplace
    )
    return conv_block_copy

def fuse_feature_encoder(feature_encoder, inplace=False):
    """
    Fuses all submodules in the feature encoder.
    """
    encoder = feature_encoder if inplace else copy.deepcopy(feature_encoder)
    encoder.conv_block = fuse_depthwise_separable_conv_block(encoder.conv_block, inplace=inplace)
    return encoder

def fuse_conv1d_relu_in_sequential(seq: nn.Sequential) -> nn.Sequential:
    """Support function for correct handling of fusion on Conv1D blocks"""
    layers = []
    i = 0
    while i < len(seq):
        if (
            isinstance(seq[i], Conv1D)
            and i + 1 < len(seq)
            and isinstance(seq[i + 1], nn.ReLU)
        ):
            fused = Conv1DReLU(seq[i])
            layers.append(fused)
            i += 2  # skip next
        else:
            layers.append(seq[i])
            i += 1
    return nn.Sequential(*layers)

def fuse_predictor_head(predictor_head: nn.Sequential, inplace=False) -> nn.Sequential:
    """Fuses conv1d+relu layers in predictor head."""
    block = predictor_head if inplace else copy.deepcopy(predictor_head)
    return fuse_conv1d_relu_in_sequential(block)

def fuse_conditioned_predictor(conditioned_predictor, inplace=False):
    """
    Fuses encoder and start/end heads in a conditioned predictor module.
    """
    predictor = conditioned_predictor if inplace else copy.deepcopy(conditioned_predictor)
    predictor.encoder = fuse_feature_encoder(predictor.encoder, inplace=inplace)
    predictor.start_block = fuse_predictor_head(predictor.start_block, inplace=inplace)
    predictor.end_block = fuse_predictor_head(predictor.end_block, inplace=inplace)
    return predictor

def fuse_model(model, inplace=False):
    """
    Top-level model fusion function.
    """
    model_copy = model if inplace else copy.deepcopy(model)
    model_copy.feature_encoder = fuse_feature_encoder(model_copy.feature_encoder, inplace=inplace)
    model_copy.predictor = fuse_conditioned_predictor(model_copy.predictor, inplace=inplace)
    return model_copy

def assign_qconfig(model, qconfig_global):
    """
    Skips embedding_net entirely, quantizes Conv1D and nn.Linear everywhere else.
    """
    for name, module in model.named_modules():
        # 1) Skip all of embedding_net (word + char lookups and char-CNN)
        if name.startswith("embedding_net"):
            module.qconfig = None

        # 2) Quantize any 1×1 conv (Conv1D) or nn.Linear downstream
        elif isinstance(module, (Conv1D, nn.Linear)):
            module.qconfig = qconfig_global

        # 3) Everything else (LayerNorm, Dropout, FloatFunctional, etc.) stays in FP32
        else:
            module.qconfig = None

def run_static_quantization_calibration(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_batches: int,
):  
    """
    Runs post-training calibration to collect activation statistics
    required for static quantization.

    Parameters:
    - model (nn.Module): The model with observers inserted (via prepare).
    - data_loader (DataLoader): DataLoader providing representative training data.
    - num_batches (int): Number of batches to use for calibration.
    """
    device = "cpu"
    model.to(device)
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

def apply_post_training_static_quantization(
        float_model: torch.nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        num_calibration_batches: int,
) -> torch.nn.Module:
    """
    Applies post-training static quantization to a floating-point model.

    Parameters:
    - float_model (nn.Module): Pretrained float model to quantize.
    - calibration_loader (DataLoader): Data loader for calibration dataset.
    - num_calibration_batches (int): Number of batches for calibration.
    - skip_embedding (bool): If True, embedding layers will not be quantized.

    Returns:
    - quantized_model (nn.Module): Quantized model ready for inference.
    """
    float_model.eval()
    float_model.to("cpu")
    

    # 1) fuse
    fused_model = fuse_model(float_model)

    # 2) pick engine
    torch.backends.quantized.engine = "fbgemm"

    # 3) build global qconfig
    qconfig_global = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=default_observer.with_args(dtype=torch.qint8)
    )
    
    # 4) assign new qconfigs
    assign_qconfig(fused_model, qconfig_global)
    

    # 5) prepare + calibrate + convert
    quant_ready_model = torch.ao.quantization.prepare(fused_model)
    run_static_quantization_calibration(
        quant_ready_model, calibration_loader, num_calibration_batches
    )
    quantized_model = torch.ao.quantization.convert(quant_ready_model, inplace=False)
    quantized_model.eval()

    return quantized_model
