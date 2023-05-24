#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --array=0-279

for p in ../examples/*/conll*/logs/*
do
  for pp in ${p}/*
  do
    args+=("--dir ${pp} --epochs 1000")
  done
done



set -x

python tri_training.py ${args[${SLURM_ARRAY_TASK_ID}]}