# copied from https://docs.pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html
import os
import sys
import time
import numpy as np

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader


def calibrate(model, data_loader, num_batches):
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


