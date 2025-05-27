EXP_NAME=IIW-400 # Name of the answer file to evaluate (located at $ANSWER_FOLDER/$EXP_NAME.jsonl)
# Path to the annotation file
ANNOTATION_FILE="/home/mingi/experiments/LLaVA/data/eval/imageinwords/IIW-400/data.jsonl"
# Directory containing the answer file to evaluate
ANSWER_FOLDER="/home/mingi/experiments/SPARC/results"
# Directory where evaluation results will be saved
SAVE_FOLDER="/home/mingi/experiments/SPARC/results"
# Your OpenAI API key for evaluation
OPENAI_API_KEY="OPENAI_API_KEY"

set -e
CUDA_VISIBLE_DEVICES=$DEVICE python -m clair \
    --annotation_file $ANNOTATION_FILE \
    --answer_folder $ANSWER_FOLDER \
    --save_folder $SAVE_FOLDER \
    --openai_api_key $OPENAI_API_KEY \
    --experiment_name $EXP_NAME \
    --data_type iiw