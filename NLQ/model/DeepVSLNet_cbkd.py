"""VSLNet Baseline for Ego4D Episodic Memory -- Natural Language Queries.
"""
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

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


class DeepVSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(DeepVSLNet, self).__init__()
        self.configs = configs

        # ─── Block 1 layers ───────────────────────────────────────────────────
        # Project raw video features into hidden dimension
        self.video_affine = VisualProjection(
            visual_dim=configs.video_feature_dim,
            dim=configs.dim,
            drop_rate=configs.drop_rate,
        )

        # Project BERT’s 768→hidden dim (initialized here, but will skip reinit)
        self.query_affine = nn.Linear(768, configs.dim)

        # Instantiate the BERT embedding (frozen inside BertEmbedding)
        self.embedding_net = BertEmbedding(configs.text_agnostic)


        # ─── Block 2 layers ───────────────────────────────────────────────────
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
        )

        # ─── Block 3 layers ───────────────────────────────────────────────────
        self.cq_attention   = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat      = CQConcatenate(dim=configs.dim)
        self.highlight_layer = HighLightLayer(dim=configs.dim)

        # ─── Block 4 layers ───────────────────────────────────────────────────
        self.predictor = ConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
        )

        # ─── Now that all submodules (including BERT) are defined, initialize non‐BERT weights
        self.init_parameters()

        # ─── Group layers into explicit blocks for CBKD ─────────────────────────
        self.block1 = nn.ModuleDict({
            "video_affine":   self.video_affine,
            "embedding_net":  self.embedding_net,
            "query_affine":   self.query_affine,
        })

        self.block2 = nn.ModuleDict({
            "feature_encoder": self.feature_encoder,
        })

        self.block3 = nn.ModuleDict({
            "cq_attention":    self.cq_attention,
            "cq_concat":       self.cq_concat,
            "highlight_layer": self.highlight_layer,
        })

        self.block4 = nn.ModuleDict({
            "predictor": self.predictor,
        })

    def init_parameters(self):
        # Gather every module under self.embedding_net.embedder
        bert_modules = set(self.embedding_net.embedder.modules())

        def init_weights(m):
            # If this module is part of the pretrained BERT, skip it:
            if m in bert_modules:
                return

            # Otherwise, initialize your newly added layers:
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        # Apply init_weights to every submodule in the network
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):

        # ─── Block 1 ────────────────────────────────────────────────────────────
        video_features = self.block1["video_affine"](video_features)
        query_features = self.block1["embedding_net"](word_ids)
        query_features = self.block1["query_affine"](query_features)
        
        # ─── Block 2 ────────────────────────────────────────────────────────────
        query_features = self.block2["feature_encoder"](query_features, mask=q_mask)
        video_features = self.block2["feature_encoder"](video_features, mask=v_mask)

        # ─── Block 3 ────────────────────────────────────────────────────────────
        features = self.block3["cq_attention"](video_features, query_features, v_mask, q_mask)
        features = self.block3["cq_concat"](features, query_features, q_mask)
        h_score = self.block3["highlight_layer"](features, v_mask)
        features = features * h_score.unsqueeze(2)

        # ─── Block 4 ────────────────────────────────────────────────────────────
        start_logits, end_logits = self.block4["predictor"](features, mask=v_mask)
        return h_score, start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        return self.block4["predictor"].extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    def compute_highlight_loss(self, scores, labels, mask):
        return self.block3["highlight_layer"].compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.block4["predictor"].compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )
