#!/bin/bash
#SBATCH --job mscbio2066
#SBATCH --partition=dept_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

echo Running on `hostname`
echo workdir $SLURM_SUBMIT_DIR

module load cuda

python3 model_k-sep_round2-proteins_100tree.py