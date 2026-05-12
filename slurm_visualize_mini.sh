#!/bin/bash

#SBATCH --job-name="visualize_mini"
#SBATCH --output=logs/visualize_mini_%j.out
#SBATCH --error=logs/visualize_mini_%j.err
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

python main.py --config config_mini.toml visualize