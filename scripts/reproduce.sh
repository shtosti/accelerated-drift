#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config.toml}"
RECREATE_ENV=0
SKIP_SYNC=0
DRY_RUN=0
STAGES=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/reproduce.sh [options] [all|stage...]

Examples:
  ./scripts/reproduce.sh all
  ./scripts/reproduce.sh --config config_mini.toml preprocess analyze visualize
  ./scripts/reproduce.sh --dry-run all

Options:
  --config PATH    Config file to use. Default: config.toml
  --recreate-env   Remove .venv before syncing dependencies
  --skip-sync      Skip dependency sync and spaCy model check
  --dry-run        Print pipeline commands without executing them
  -h, --help       Show this help

Stages:
  collect preprocess analyze visualize additional-analysis topic-compare
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --recreate-env)
      RECREATE_ENV=1
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      STAGES+=("$1")
      shift
      ;;
  esac
done

if [[ "${#STAGES[@]}" -eq 0 ]]; then
  STAGES=("all")
fi

if [[ " ${STAGES[*]} " == *" all "* ]]; then
  STAGES=("collect" "preprocess" "analyze" "visualize")
fi

uv_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv "$@"
  else
    python -m uv "$@"
  fi
}

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing uv with Python..."
    python -m pip install --user uv
  fi

  if [[ "$RECREATE_ENV" -eq 1 && -d .venv ]]; then
    echo "Removing existing .venv..."
    rm -rf .venv
  fi

  echo "Syncing Python environment from uv.lock..."
  uv_cmd sync --locked

  echo "Checking spaCy English model..."
  if ! uv_cmd run python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('en_core_web_sm') else 1)"; then
    uv_cmd run python -m spacy download en_core_web_sm
  fi
fi

for stage in "${STAGES[@]}"; do
  case "$stage" in
    collect|preprocess|analyze|visualize|additional-analysis|topic-compare)
      echo
      echo "=== $stage ==="
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "uv run python main.py --config $CONFIG $stage"
      else
        uv_cmd run python main.py --config "$CONFIG" "$stage"
      fi
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo
echo "Done."
