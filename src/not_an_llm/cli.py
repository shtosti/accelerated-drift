from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from not_an_llm.analysis.features import default_hypotheses
from not_an_llm.config import load_config
from not_an_llm.pipelines.collect import run_collection



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="not-an-llm pipeline")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to root configuration TOML file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("collect", help="Download Semantic Scholar papers to JSONL.")
    subparsers.add_parser("show-hypotheses", help="Print default feature-shift hypotheses.")

    return parser



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "collect":
        output_path = run_collection(config)
        print(f"Saved paper records to {output_path}")
        return

    if args.command == "show-hypotheses":
        payload = [asdict(item) for item in default_hypotheses()]
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
