"""Instance of Quantized VSLNet"""
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

from model.layers import (
    Embedding,
    VisualProjection
)

from model.qlayers import (
    QFeatureEncoder,
    QCQAttention,
    QCQConcatenate,
    QHighLightLayer,
    QConditionedPredictor
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
    def __init__(self, configs, word_vectors):
        super().__init__()
        self.configs = configs

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.video_affine = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )
        self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
        )
        self.feature_encoder = QFeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
            quant=self.quant,
            dequant=self.dequant
        )
        self.cq_attention = QCQAttention(dim=configs.dim, drop_rate=configs.drop_rate, quant=self.quant, dequant=self.dequant)
        self.cq_concat = QCQConcatenate(dim=configs.dim, quant=self.quant, dequant=self.dequant)
        self.highlight_layer = QHighLightLayer(dim=configs.dim, quant=self.quant, dequant=self.dequant)
        self.predictor = QConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
            quant=self.quant,
            dequant=self.dequant
        )

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
        
        video_features = self.quant(video_features)
        video_features = self.video_affine(video_features)
        
        query_features = self.embedding_net(word_ids, char_ids)
        query_features = self.quant(query_features)

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
        return self.quant(self.predictor.extract_index(
            start_logits=self.dequant(start_logits), end_logits=self.dequant(end_logits)
            ))

    def compute_highlight_loss(self, scores, labels, mask):
        return self.quant(self.highlight_layer.compute_loss(
            scores=self.dequant(scores), labels=self.dequant(labels), mask=self.dequant(mask)
            ))

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.quant(self.predictor.compute_cross_entropy_loss(
            start_logits=self.dequant(start_logits),
            end_logits=self.dequant(end_logits),
            start_labels=self.dequant(start_labels),
            end_labels=self.dequant(end_labels),
            ))
