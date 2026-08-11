#!/bin/bash

#SBATCH --job-name="preprocessing_full"
#SBATCH --output=logs/preprocessing_full_%j.out
#SBATCH --error=logs/preprocessing_full_%j.err
#SBATCH --partition=epyc2
#SBATCH --mem=12G
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

python -m spacy download en_core_web_sm
python main.py --config config.toml preprocess