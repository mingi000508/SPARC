EXPERIMENT_NAME=COCO

# Directory where results will be saved
ANSWER_FOLDER="/home/mingi/experiments/SPARC/results"
ANNOTATION_FOLDER=/home/mingi/data/coco/annotations
SAVE_FOLDER="/home/mingi/experiments/SPARC/results"

python chair.py \
    --cap_file $ANSWER_FOLDER/${EXPERIMENT_NAME}.jsonl \
    --image_id_key image_id --caption_key \
    caption --coco_path $ANNOTATION_FOLDER \
        --save_path $SAVE_FOLDER/${EXPERIMENT_NAME}_chair.jsonl \
    > $SAVE_FOLDER/${EXPERIMENT_NAME}.log