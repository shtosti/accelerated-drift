#!/bin/bash

#SBATCH --job-name="syntax-readability"
#SBATCH --output=logs/syntax-readability_%j.out
#SBATCH --error=logs/syntax-readability_%j.err
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -euo pipefail

mkdir -p logs
source .venv/bin/activate

echo "--- Python runtime info ---"
which python
python -c "import sys; print(sys.executable)"

echo "--- Verifying abstract feature inputs ---"
for dataset in \
  arxiv_ai_abstracts \
  arxiv_qbio_abstracts \
  arxiv_stat_abstracts \
  medarxiv_abstracts
do
  feature_file="data/analysis/${dataset}/features.jsonl"
  if [[ ! -s "${feature_file}" ]]; then
    echo "Missing or empty input: ${feature_file}" >&2
    exit 1
  fi
  echo "OK: ${feature_file}"
done

echo "--- Running full syntax-readability analysis without dependency bigrams ---"
python scripts/analyze_syntax_readability_abstracts.py \
  --skip-bigrams \
  --bootstrap 50 \
  --permutation-repeats 5 \
  --output-name syntax_readability_abstracts_no_bigrams

echo "--- Output files ---"
find data/analysis/syntax_readability_abstracts_no_bigrams \
  data/visuals/syntax_readability_abstracts_no_bigrams \
  -maxdepth 2 -type f -print
