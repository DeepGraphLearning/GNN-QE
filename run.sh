#!/bin/bash
#SBATCH --workdir=/home/tacucumides
#SBATCH --ntasks=1
#SBATCH --job-name=gnn-qe
#SBATCH --nodelist=scylla
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tacucumides@uc.cl
#SBATCH --output=/home/tacucumides/GNN-QE/logs/%A.log
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --cpus=8
#SBATCH --partition=ialab-high
pwd; hostname; date
echo "Start training"
echo $(pwd)
cd /home/tacucumides
source miniconda3/etc/profile.d/conda.sh
conda activate gnn-qe
cd /home/tacucumides/GNN-QE/script
python run.py
