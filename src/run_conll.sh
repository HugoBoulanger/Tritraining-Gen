#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --array=0-49

for p in ../examples/deeplo_conll_avg/logs/*
do
  for pp in ${p}/*
  do
    args+=("--dir ${pp} --epochs 100")
  done
done

for p in ../examples/deeplo_conll_small/logs/*
do
  for pp in ${p}/*
  do
    args+=("--dir ${pp} --epochs 100")
  done
done

set -x

python tri_training.py ${args[${SLURM_ARRAY_TASK_ID}]}