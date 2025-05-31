# To train the model.
CUDA_VISIBLE_DEVICES=1 python main.py \
    --task nlq_official_v1 \
    --model_name vslnet \
    --predictor bert \
    --mode train \
    --video_feature_dim 2304 \
    --max_pos_len 128 \
    --epochs 200 \
    --fv official \
    --num_workers 64 \
    --model_dir checkpoints/ \
    --eval_gt_json "data/nlq_val.json" \
    --feature_map_weight 0.25 \
    --ce_loss_weight 0.75 \
    --distill_weight_loss 0.2


# To predict on test set.
# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task nlq_official_v1 \
#     --model_name
#     --predictor bert \
#     --mode test \
#     --video_feature_dim 2304 \
#     --max_pos_len 128 \
#     --fv official \
#     --model_dir checkpoints/


# To evaluate predictions using official evaluation script.
# PRED_FILE="checkpoints/vslnet_nlq_official_v1_official_128_bert/model"
# python utils/evaluate_ego4d_nlq.py \
#     --ground_truth_json data/nlq_test.json \
#     --model_prediction_json ${PRED_FILE}/vslnet_41184_test_result.json \
#     --thresholds 0.3 0.5 \
#     --topK 1 3 5
