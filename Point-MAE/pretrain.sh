#!/bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --constraint "inet&80gb"

module load gcc/9.5.0
module load miniconda3

source activate /scratch/usr/nimpede1/.conda/envs/pointmae/

python main.py --config cfgs/pretrain.yaml --exp_name 03_npoints-1024_mr-40