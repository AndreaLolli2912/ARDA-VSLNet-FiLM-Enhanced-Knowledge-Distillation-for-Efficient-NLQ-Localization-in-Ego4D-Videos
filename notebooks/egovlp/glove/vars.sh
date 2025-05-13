
export NAME=egovlp_fp16
export TASK_NAME=nlq_official_v1_$NAME
export BASE_DIR=data/dataset/nlq_official_v1_$NAME
export FEATURE_BASE_DIR=data/features/nlq_official_v1_$NAME
export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
export MODEL_BASE_DIR=/content/nlq_official_v1/checkpoints/
export GLOVE_DICTIONARY=data/features/glove.840B.300d.txt
export ANNOTATION_PREPARED=data/dataset/nlq_official_v1_$NAME
export VIDEO_PREPARED=$FEATURE_BASE_DIR/official
export GDRIVE_PREPARED_ANNOTATION=/content/drive/MyDrive/prepared_features/$TASK_NAME/
export GDRIVE_PREPARED_VIDEO=/content/drive/MyDrive/prepared_features/official/
export DATASET_PATH=data/dataset

cd episodic-memory/NLQ/VSLNet
