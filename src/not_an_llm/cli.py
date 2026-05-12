from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from not_an_llm.analysis.features import default_hypotheses
from not_an_llm.config import load_config
from not_an_llm.pipelines.analyze import run_analysis
from not_an_llm.pipelines.collect import run_collection
from not_an_llm.pipelines.external_analyze import (
    run_configured_external_analysis,
    run_external_analysis,
)
from not_an_llm.pipelines.external_preprocess import (
    run_configured_external_preprocessing,
    run_hc3_preprocessing,
    run_mage_preprocessing,
)
from not_an_llm.pipelines.preprocess import run_preprocessing
from not_an_llm.pipelines.visualize import run_visualization



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="not-an-llm pipeline")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to root configuration TOML file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("collect", help="Download Semantic Scholar papers to JSONL.")
    subparsers.add_parser("preprocess", help="Preprocess raw title/abstract text and save JSONL.")
    subparsers.add_parser("analyze", help="Run feature and readability analysis with yearly trends.")
    subparsers.add_parser("visualize", help="Generate plots from previously computed analysis data.")
    subparsers.add_parser("show-hypotheses", help="Print default feature-shift hypotheses.")

    subparsers.add_parser(
        "external-preprocess",
        help="Build paired human/AI JSONL for the dataset selected in [external].",
    )
    subparsers.add_parser(
        "external-analyze",
        help="Analyze paired human/AI JSONL selected in [external] without temporal trends.",
    )

    ext_pre = subparsers.add_parser(
        "hc3-preprocess",
        help="Build paired human/AI JSONL from HC3 (separate from main pipelines).",
    )
    ext_pre.add_argument("--subset", default="all", help="HC3 subset, e.g. all or medicine.")
    ext_pre.add_argument("--split", default="train", help="Dataset split to load.")
    ext_pre.add_argument(
        "--output",
        default="data/external/hc3_pairs_all.jsonl",
        help="Output JSONL path for paired human/AI entries.",
    )

    ext_an = subparsers.add_parser(
        "hc3-analyze",
        help="Analyze paired HC3 JSONL without temporal trends (separate from main pipelines).",
    )
    ext_an.add_argument(
        "--input",
        default="data/external/hc3_pairs_all.jsonl",
        help="Input paired JSONL path produced by hc3-preprocess.",
    )
    ext_an.add_argument(
        "--feature-output",
        default="data/analysis/external/hc3_pair_features.jsonl",
        help="Output JSONL with per-text extracted features.",
    )
    ext_an.add_argument(
        "--comparison-csv",
        default="data/analysis/external/hc3_human_vs_ai_comparison.csv",
        help="Output CSV with feature-level human vs AI comparisons.",
    )
    ext_an.add_argument(
        "--comparison-plot",
        default="data/analysis/external/hc3_human_vs_ai_top_differences.png",
        help="Output plot for top feature differences.",
    )

    mage_pre = subparsers.add_parser(
        "mage-preprocess",
        help="Build paired human/AI JSONL from MAGE (separate from main pipelines).",
    )
    mage_pre.add_argument("--split", default="validation", help="Dataset split to load.")
    mage_pre.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of human/AI pairs to write (0 means use all balanced pairs).",
    )
    mage_pre.add_argument("--seed", type=int, default=42, help="Random seed for pairing shuffle.")
    mage_pre.add_argument(
        "--output",
        default="data/external/mage_pairs_validation.jsonl",
        help="Output JSONL path for paired human/AI entries.",
    )

    mage_an = subparsers.add_parser(
        "mage-analyze",
        help="Analyze paired MAGE JSONL without temporal trends (separate from main pipelines).",
    )
    mage_an.add_argument(
        "--input",
        default="data/external/mage_pairs_validation.jsonl",
        help="Input paired JSONL path produced by mage-preprocess.",
    )
    mage_an.add_argument(
        "--feature-output",
        default="data/analysis/external/mage_pair_features_validation.jsonl",
        help="Output JSONL with per-text extracted features.",
    )
    mage_an.add_argument(
        "--comparison-csv",
        default="data/analysis/external/mage_human_vs_ai_comparison_validation.csv",
        help="Output CSV with feature-level human vs AI comparisons.",
    )
    mage_an.add_argument(
        "--comparison-plot",
        default="data/analysis/external/mage_human_vs_ai_top_differences_validation.png",
        help="Output plot for top feature differences.",
    )

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

    if args.command == "preprocess":
        output_path = run_preprocessing(config)
        print(f"Saved preprocessed records to {output_path}")
        return

    if args.command == "analyze":
        artifacts = run_analysis(config)
        print(f"Saved feature dataset to {artifacts.feature_dataset_jsonl}")
        print(f"Saved yearly trends to {artifacts.trends_csv}")
        print(f"Saved monthly trends to {artifacts.monthly_trends_csv}")
        print(f"Saved trend plots to {config.analysis.trends_plot_dir} ({len(artifacts.trends_plot_paths)} files)")
        return

    if args.command == "visualize":
        artifacts = run_visualization(config)
        print(f"Saved trend plots to {config.analysis.trends_plot_dir} ({len(artifacts.trends_plot_paths)} files)")
        return

    if args.command == "show-hypotheses":
        payload = [asdict(item) for item in default_hypotheses()]
        print(json.dumps(payload, indent=2))
        return

    if args.command == "external-preprocess":
        artifacts = run_configured_external_preprocessing(config)
        print(f"Saved paired external records to {artifacts.output_jsonl} ({artifacts.records_written} pairs)")
        return

    if args.command == "external-analyze":
        artifacts = run_configured_external_analysis(config)
        print(f"Saved external feature dataset to {artifacts.feature_dataset_jsonl}")
        print(f"Saved human-vs-ai comparison table to {artifacts.comparison_csv}")
        if artifacts.comparison_plot:
            print(f"Saved top-differences plot to {artifacts.comparison_plot}")
        return

    if args.command == "hc3-preprocess":
        artifacts = run_hc3_preprocessing(
            subset=args.subset,
            split=args.split,
            output_jsonl=args.output,
        )
        print(f"Saved paired HC3 records to {artifacts.output_jsonl} ({artifacts.records_written} pairs)")
        return

    if args.command == "hc3-analyze":
        artifacts = run_external_analysis(
            config,
            input_jsonl=args.input,
            feature_dataset_jsonl=args.feature_output,
            comparison_csv=args.comparison_csv,
            comparison_plot=args.comparison_plot,
        )
        print(f"Saved external feature dataset to {artifacts.feature_dataset_jsonl}")
        print(f"Saved human-vs-ai comparison table to {artifacts.comparison_csv}")
        print(f"Saved top-differences plot to {artifacts.comparison_plot}")
        return

    if args.command == "mage-preprocess":
        artifacts = run_mage_preprocessing(
            split=args.split,
            output_jsonl=args.output,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
        print(f"Saved paired MAGE records to {artifacts.output_jsonl} ({artifacts.records_written} pairs)")
        return

    if args.command == "mage-analyze":
        artifacts = run_external_analysis(
            config,
            input_jsonl=args.input,
            feature_dataset_jsonl=args.feature_output,
            comparison_csv=args.comparison_csv,
            comparison_plot=args.comparison_plot,
        )
        print(f"Saved external feature dataset to {artifacts.feature_dataset_jsonl}")
        print(f"Saved human-vs-ai comparison table to {artifacts.comparison_csv}")
        print(f"Saved top-differences plot to {artifacts.comparison_plot}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
