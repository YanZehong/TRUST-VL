#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="trustvl-13b-task"
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_newsclippings \
        --model-path ./checkpoints/trustvl-13b-task \
        --question-file ./data/eval/NewsCLIPpings_7264.jsonl \
        --image-folder ./data/origin \
        --answers-file ./outputs/eval/NewsCLIPpings/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --enable-direct-evidence \
        --enable-inverse-evidence &
done

wait

output_file="./outputs/eval/NewsCLIPpings/$SPLIT/$CKPT/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./outputs/eval/NewsCLIPpings/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ./llava/eval/eval_results.py \
    --judge_file $output_file &

