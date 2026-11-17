#!/bin/bash
#SBATCH --job-name=train_ast
#SBATCH --time=4:00:00            # Max run time
#SBATCH --mem=128G                 # Robosuite workers are memory-heavy; tune with seff if needed
#SBATCH --cpus-per-task=64        # Match CPU count to vectorized env workers
#SBATCH --gres=gpu:1

# Load environment
# remember to change the path to your conda environment
module load scicomp-python-env
source activate /scratch/work/ouy1/conda_envs/geo_asr

# Keep BLAS thread pools from over-subscribing in each worker process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /scratch/work/ouy1/speech_recognition

export HF_HOME=/scratch/work/ouy1/.cache/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2

srun python -u train.py