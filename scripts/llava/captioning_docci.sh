DEVICE=0
EXP_NAME=DOCCI # results will be saved as $SAVE_FOLDER/$EXP_NAME.jsonl
ALPHA=1.1 # paramerter for attention re-weighting
BETA=0.1 # parameter for EMA
TAU=1.5 # Threshold
LAYER=20 # layer used for token selection

MODEL_PATH=liuhaotian/llava-v1.5-7b
DATASET_TYPE=docci # Dataset type: options are coco, iiw, or docci
# Path to input images
IMAGE_FOLDER="/home/mingi/experiments/LLaVA/data/eval/DOCCI/images"
# Path to annotation file
ANNOTATION_FILE="/home/mingi/experiments/LLaVA/data/eval/DOCCI/docci_descriptions.jsonl"
# Directory where results will be saved
SAVE_FOLDER="/home/mingi/experiments/SPARC/results"
SEED=42 # Random seed for evaluation data sampling (used only for coco_val and docci)

set -e
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval \
    --experiment_name $EXP_NAME \
    --model-path $MODEL_PATH \
    --dataset_type $DATASET_TYPE \
    --alpha $ALPHA \
    --beta $BETA \
    --image-folder $IMAGE_FOLDER \
    --annotation-file $ANNOTATION_FILE \
    --save_path $SAVE_FOLDER \
    --start_layer 0 \
    --end_layer 31 \
    --selected_layer $LAYER \
    --tau $TAU \
    --seed $SEED