#!/bin/bash

#SBATCH --job-name="analysis_full"
#SBATCH --output=logs/analysis_full_%j.out
#SBATCH --error=logs/analysis_full_%j.err
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

python -m spacy download en_core_web_sm
python main.py --config config.toml analyze
python main.py --config config.toml visualize