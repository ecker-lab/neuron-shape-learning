#!/bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --constraint "inet&80gb"

export PREFERRED_SOFTWARE_STACK=nhr-lmod
source /sw/etc/profile/profile.sh

module load gcc/9.5.0
module load miniconda3

source activate /scratch/usr/nimpede1/.conda/envs/pointmae/

python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_test_new_classification --ckpts experiments/finetune/cfgs/04_finetune_npoints-1024_mr-60/ckpt-best.pth

# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-1024_mr-40 --ckpts experiments/finetune/cfgs/04_finetune_npoints-1024_mr-40/ckpt-best.pth
# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-1024_mr-60 --ckpts experiments/finetune/cfgs/04_finetune_npoints-1024_mr-60/ckpt-best.pth
# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-1024_mr-80 --ckpts experiments/finetune/cfgs/04_finetune_npoints-1024_mr-80/ckpt-best.pth
# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-8192_mr-40 --ckpts experiments/finetune/cfgs/04_finetune_npoints-8192_mr-40/ckpt-best.pth
# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-8192_mr-60 --ckpts experiments/finetune/cfgs/04_finetune_npoints-8192_mr-60/ckpt-best.pth
# python main_vis.py --config cfgs/cluster.yaml --classify --exp_name 04_finetune_npoints-8192_mr-80 --ckpts experiments/finetune/cfgs/04_finetune_npoints-8192_mr-80/ckpt-best.pth