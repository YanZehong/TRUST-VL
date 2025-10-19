#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="trustvl-13b-task"
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_factify \
        --model-path ./checkpoints/trustvl-13b-task \
        --question-file ./data/eval/Factify2_3000.jsonl \
        --image-folder ./data \
        --answers-file ./outputs/eval/Factify2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file="./outputs/eval/Factify2/$SPLIT/$CKPT/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./outputs/eval/Factify2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ./llava/eval/eval_results.py \
    --judge_file $output_file &

