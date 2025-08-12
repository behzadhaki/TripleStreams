#!/bin/bash
#SBATCH -J MuteGenreLatentVAE
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH -o logs/%J.%N.out
#SBATCH -e logs/%J.%N.err

# Necessary to access existing modules on the cluster
source /etc/profile.d/lmod.sh
source /etc/profile.d/zz_hpcnow-arch.sh

# Load Anaconda Module, Activate Conda CLI and Activate Environment
module load Anaconda3/2020.02
# module load wandb/0.13.3-GCCcore-10.2.0

source activate GrooveTransformer

# Login to WANDB
export WANDB_API_KEY=a697f9d7a155dab1e52d29d29164a82ecbaf251c
python -m wandb login


# Run your codes here

cd GrooveTransformerV2
python train_BaseVAE.py --config="config_beta_0.5.yaml"

