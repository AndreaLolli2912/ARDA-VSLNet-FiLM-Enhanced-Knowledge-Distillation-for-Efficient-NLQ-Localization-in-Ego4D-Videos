import copy

import torch
from torch import nn

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
        float_model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        num_calibration_batches: int
):
    """
    Applies post-training static quantization to a given float model.

    Steps:
    1. Fuse supported modules (e.g., Conv+BN+ReLU).
    2. Attach quantization configuration and insert observers.
    3. Calibrate using representative training data.
    4. Convert the model to a quantized version.

    Parameters:
    - float_model (nn.Module): The pre-trained float32 model.
    - calibration_loader (DataLoader): Representative data for calibration.
    - num_calibration_batches (int): Number of batches to use for calibration.

    Returns:
    - nn.Module: The quantized version of the model.
    """
    float_model.eval()
    float_model.to("cpu")
    
    # 1: fuse modules
    fused_model = fuse_model(float_model)
    fused_model.qconfig = torch.ao.quantization.default_qconfig
    
    # 2: insert observers
    quant_ready_model = QuantizedDeepVSLNet(fused_model)
    torch.ao.quantization.prepare(quant_ready_model, inplace=True)

    # 3: calibration
    run_static_quantization_calibration(
        quant_ready_model, calibration_loader, num_calibration_batches
    )

    # 4: convert to quantized
    quantized_model = torch.ao.quantization.convert(quant_ready_model, inplace=False)
    quantized_model.eval()

    return quantized_model
