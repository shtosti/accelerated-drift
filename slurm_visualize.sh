#!/bin/bash

#SBATCH --job-name="visualize_full"
#SBATCH --output=logs/visualize_full_%j.out
#SBATCH --error=logs/visualize_full_%j.err
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

python main.py --config config.toml visualize