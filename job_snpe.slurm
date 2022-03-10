#!/bin/bash

#SBATCH -p general
#SBATCH -t 48:00:00
#SBATCH --job-name=snpetest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=76gb

echo "$0"
printf "%s" "$(<$0)"
echo ""

#module purge
source activate uedge-sbi-env

srun python3 -u snpe_tanh_plots.py
