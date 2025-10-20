#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="trustvl-13b-task"
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_out_of_domain \
        --model-path ./checkpoints/trustvl-13b-task \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file="./outputs/eval/MOCHEG/$SPLIT/$CKPT.jsonl"

python ./llava/eval/eval_results.py \
    --judge_file $output_file &

wait

output_file="./outputs/eval/Fakeddit/$SPLIT/$CKPT.jsonl"

python ./llava/eval/eval_results.py \
    --judge_file $output_file &

wait

output_file="./outputs/eval/VERITE/$SPLIT/$CKPT.jsonl"

python ./llava/eval/eval_results.py \
    --judge_file $output_file &
