#!/bin/bash

#SBATCH --job-name="collection_full"
#SBATCH --output=logs/collection_full_%j.out
#SBATCH --error=logs/collection_full_%j.err
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=2
#SBATCH --time=07:00:00
#SBATCH --ntasks=1

set -euo pipefail


source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"


python main.py --config config.toml collect