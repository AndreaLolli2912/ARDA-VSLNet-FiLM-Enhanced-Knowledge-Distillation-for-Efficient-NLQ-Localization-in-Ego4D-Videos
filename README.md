# ARDAVSLNet: FiLM-Enhanced Knowledge Distillation for Efficient Natural Language Query Localization in Ego4D Videos

Project for "Machine Learning and Deep Learning" 2025/2026 made in collaboration with my colleagues [Di Chiara Rino](https://github.com/RinDiC99), [Famà Daniele](https://github.com/danielefam) and [Fico Alice](https://github.com/AliceFico).

## Data Exploration
The exploratory analysis lives in **`notebooks/data_exploration.ipynb`**.

### Place the annotation files

Put the official NLQ annotation package in **exactly** this location:

```
Ego4D-NLQ/
└── ego4d_data/
    └── v2/annotations/
        ├── manifest.csv
        ├── nlq_train.json
        ├── nlq_val.json
        └── nlq_test_unannotated.json
```

*(If you are working with v1 annotations, simply replace `v2` with `v1`.)*
Open the notebook and run all, and the plots will be generated automatically. That’s it — no videos are required for the exploration step.

## Benchmark: VSLBase & VSLNet with Omnivore and EgoVLP features

The VSLBase implementation is located in `NLQ/model/VSLBase.py` (VSLNet is analogous). All experiments can be reproduced from `notebooks/Final_Ego4D_NLQ.ipynb`.

The notebook expects the pre‑extracted video features to be available locally. Ensure that the path variables in `vars.sh` match your directory structure.

Example command‑line run (VSLBase):

```bash
python main.py \
    --model_name vslbase \
    --predictor glove      # set to bert to use the BERT encoder     
    ...
```

Our variation replaces the original GloVe text encoder with BERT; toggle it via `--predictor bert`.

## Extension 1: FiLM Conditioning Layer

We add the **FiLM conditioning layer** described in *“FiLM: Visual Reasoning with a General Conditioning Layer.”*  The implementation is at the end of `Ego4D-NLQ/NLQ/model/layers.py`. While the new model instance which supports the film layer can be found in `Ego4D-NLQ/NLQ/model/DeepVSLNet.py`

The insertion point of the FiLM block is selected at runtime with the CLI flag `--film_mode`. Allowed values are `before_encoder`, `after_encoder`, `inside_encoder:after_pos`, `inside_encoder:after_conv`, `inside_encoder:after_attn`, `inside_encoder:multi`, and `off` (default).

Example:

```bash
python main.py \
    --model_name deepvslnet \
    --predictor glove \
    --film_mode inside_encoder:after_pos \
    ...
```

---

## Extension 2: Knowledge Distillation and Quantization

The final part of the project investigates model compression.

**Knowledge distillation.**

*Technique 1 – Teacher‑Student [(PyTorch reference)](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).*
The student implementation is in `NLQ/model/LightVSLNet.py`; its compressed building blocks are in `NLQ/model/light_layers.py`.  Training is launched via `main_distillation.py`, which mirrors the standard `main.py` interface (same flags and environment variables).  This setup follows the procedure in the official PyTorch tutorial and produces a lighter model with minimal loss in accuracy.

*Technique 2 – Counterclockwise Block‑by‑Block Knowledge Distillation.*
Implemented in utils/cbkd_helpers.py (pruning and KD utilities) and utils/cbkd_config.py (stage hyper‑parameters). The full pipeline is executed with cbkd_main.py, which prunes and retrains blocks of DeepVSLNet in successive stages, following the original paper.

## Ectension 3: Post‑training static quantization (PTQ).
The repository includes a prototype PTQ pipeline (utils/quantization_helpers.py). It performs module fusion, assigns per‑layer qconfigs, runs calibration on a held‑out subset, and converts the model to 8‑bit integers – all after full‑precision training is complete.

Entry points main_quantization.py (eager mode) and main_fxquantization.py (FX graph mode).

Quantized model stub NLQ/model/QuantizedDeepVSLNet.py.

We attempted post‑training static quantization, but encountered implementation issues with PyTorch’s quantization APIs and, given our limited experience, were unable to generate a working int8 checkpoint. The helper scripts therefore remain a proof of concept for future work.

## Project Report

A full description of the methodology, experiments and results is provided [here](https://github.com/AndreaLolli2912/Ego4D-NLQ/blob/main/ARDA_VSLNet__FiLM_Enhanced_Knowledge_Distillation_for_Efficient_Natural_Language_Query_Localization_in_Ego4D_Videos.pdf) ("ARDA VSLNet: FiLM‑Enhanced Knowledge Distillation for Efficient NLQ in Ego4D Videos")

# References

**FiLM: Visual Reasoning with a General Conditioning Layer**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.11671

**Counterclockwise block-by-block knowledge distillation for neural network compression**: Lan, X., Zeng, Y., Wei, X. et al. Counterclockwise block-by-block knowledge distillation for neural network compression. Sci Rep 15, 11369 (2025). https://doi.org/10.1038/s41598-025-91152-3
