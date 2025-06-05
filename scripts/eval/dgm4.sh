#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="veritas-7b-ablation-mistral-1000"
SPLIT="test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_dgm4 \
        --model-path /home/zehong/LLaVA/checkpoints/veritas-7b-newstune-198k-ablation_v1.6-mistral \
        --question-file /home/zehong/LLaVA/data/DGM4/metadata/test.json \
        --image-folder /home/zehong/LLaVA/data \
        --answers-file /home/zehong/LLaVA/outputs/eval/dgm4/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file="/home/zehong/LLaVA/outputs/eval/dgm4/$SPLIT/$CKPT/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/zehong/LLaVA/outputs/eval/dgm4/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /home/zehong/LLaVA/llava/eval/eval_results.py \
    --judge_file $output_file \
    --label_reference 'mixed' &

