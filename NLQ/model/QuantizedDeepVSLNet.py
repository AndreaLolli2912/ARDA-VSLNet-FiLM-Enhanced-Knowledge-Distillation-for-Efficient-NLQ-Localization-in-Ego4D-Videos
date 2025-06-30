"""Instance of Quantized VSLNet"""
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

from model.DeepVSLNet import DeepVSLNet
from model.layers import (
    Embedding,
    VisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    HighLightLayer,
    BertEmbedding,
    FiLM,
)


def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler

class QuantizedDeepVSLNet(nn.Module):
    def __init__(self, base_model: DeepVSLNet):
        super().__init__()
        self.configs = base_model.configs

        self.query_quant = QuantStub()
        self.video_quant = QuantStub()
        self.dequant = DeQuantStub()

        self.video_affine = base_model.video_affine
        self.embedding_net = base_model.embedding_net
        self.feature_encoder = base_model.feature_encoder
        self.cq_attention = base_model.cq_attention
        self.cq_concat = base_model.cq_concat
        self.highlight_layer = base_model.highlight_layer
        self.predictor = base_model.predictor
        

        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)
    
    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        video_features = self.video_quant(video_features)

        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(word_ids, char_ids)

        query_features = self.query_quant(query_features)

        query_features = self.feature_encoder(query_features, mask=q_mask)

        video_features = self.feature_encoder(
            video_features,
            mask=v_mask,
            query_feats=query_features,
            film_mode=self.configs.film_mode
        )

        # Cross-attention and prediction
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        start_logits, end_logits = self.predictor(features, mask=v_mask)
        
        return self.dequant(h_score), self.dequant(start_logits), self.dequant(end_logits)

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )
