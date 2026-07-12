#!/bin/bash

#SBATCH --job-name="additional-analysis_gpu"
#SBATCH --output=logs/additional-analysis_gpu_%j.out
#SBATCH --error=logs/additional-analysis_gpu_%j.err
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

python -m spacy download en_core_web_sm

python main.py --config config.toml additional-analysis