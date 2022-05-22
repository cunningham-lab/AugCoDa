#!/bin/bash

# count number of datasets
task_idxs=($(seq 0 1 11))
seeds=($(seq 0 1 19))

# Fast jobs can run in series
for i in "${task_idxs[@]}"; do
for s in "${seeds[@]}"; do
    sbatch code/cpu_run.sh "code/train_and_evaluate.py --data_idx=$i --seed=$s --method=fast"
done
done

for i in "${task_idxs[@]}"; do
for s in "${seeds[@]}"; do
    sbatch code/cpu_run.sh "code/train_and_evaluate.py --data_idx=$i --seed=$s --method=deepmicro"
done
done

for i in "${task_idxs[@]}"; do
for s in "${seeds[@]}"; do
    sbatch code/cpu_run.sh "code/train_and_evaluate.py --data_idx=$i --seed=$s --method=contrastive"
done
done


for i in "${task_idxs[@]}"; do
for s in "${seeds[@]}"; do
    sbatch code/cpu_run_large.sh "code/train_and_evaluate.py --data_idx=$i --seed=$s --method=maml"
done
done