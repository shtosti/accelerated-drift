from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.interrupted_time_series import compute_interrupted_time_series
from not_an_llm.analysis.visual_style import (
    DARK_GREY,
    GREEN,
    HATCHES,
    ORCHID,
    WHITE,
    sign_color,
)


DEFAULT_DATASETS = (
    "arxiv_ai_abstracts",
    "arxiv_qbio_abstracts",
    "arxiv_stat_abstracts",
    "medarxiv_abstracts",
)

ARI_COLUMN = "automated_readability_index_monthly_mean"
FKGL_COLUMN = "flesch_kincaid_grade_monthly_mean"
WORDS_PER_SENTENCE_COLUMN = "avg_words_per_sentence_monthly_mean"
SYLLABLES_PER_WORD_COLUMN = "avg_syllables_per_word_monthly_mean"

ARI_TOTAL = "ari_total"
ARI_SENTENCE_COMPONENT = "ari_sentence_length_component"
ARI_CHARACTER_COMPONENT = "ari_implied_character_word_component"
ARI_INTERCEPT = -21.43
ARI_SENTENCE_COEFFICIENT = 0.5

FKGL_TOTAL = "fkgl_total"
FKGL_SENTENCE_COMPONENT = "fkgl_sentence_length_component"
FKGL_SYLLABLE_COMPONENT = "fkgl_syllables_word_component"
FKGL_RECONCILIATION_COMPONENT = "fkgl_formula_reconciliation_component"
FKGL_INTERCEPT = -15.59
FKGL_SENTENCE_COEFFICIENT = 0.39
FKGL_SYLLABLE_COEFFICIENT = 11.8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose abstract ARI and FKGL slope changes into their formula components."
    )
    parser.add_argument("--analysis-dir", default="data/analysis")
    parser.add_argument("--visuals-dir", default="data/visuals")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Abstract-only analysis folders to include.",
    )
    parser.add_argument("--output-name", default="readability_abstract_decomposition")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    table_dir = analysis_dir / args.output_name
    visual_dir = Path(args.visuals_dir) / args.output_name
    table_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    component_rows: list[pd.DataFrame] = []
    ari_summary_rows: list[dict[str, object]] = []
    fkgl_summary_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        monthly_path = analysis_dir / dataset / "trends_by_month.csv"
        if not monthly_path.exists():
            print(f"Skipping {dataset}: missing {monthly_path}")
            continue
        monthly = pd.read_csv(monthly_path)
        components = build_readability_components(monthly)
        stats = fit_component_its(components)
        if stats.empty:
            print(f"Skipping {dataset}: insufficient monthly observations for ITS")
            continue
        stats.insert(0, "dataset", dataset)
        component_rows.append(stats)
        ari_summary_rows.append(summarize_ari_decomposition(dataset, stats))
        fkgl_summary_rows.append(summarize_fkgl_decomposition(dataset, stats))

    if not ari_summary_rows:
        raise SystemExit("No abstract datasets produced a valid readability decomposition.")

    component_stats = pd.concat(component_rows, ignore_index=True)
    ari_summary = pd.DataFrame(ari_summary_rows)
    fkgl_summary = pd.DataFrame(fkgl_summary_rows)

    # Remove the obsolete ARI-only filename from early versions of this script.
    (table_dir / "ari_component_its.csv").unlink(missing_ok=True)
    component_stats.to_csv(table_dir / "readability_component_its.csv", index=False)
    ari_summary.to_csv(table_dir / "ari_slope_change_decomposition.csv", index=False)
    fkgl_summary.to_csv(table_dir / "fkgl_slope_change_decomposition.csv", index=False)

    save_decomposition_plot(
        ari_summary,
        visual_dir / "ari_slope_change_decomposition.png",
        metric_label="ARI",
        sentence_column="sentence_length_contribution_per_year",
        second_column="implied_character_word_contribution_per_year",
        second_label="implied chars/word",
    )
    save_decomposition_plot(
        fkgl_summary,
        visual_dir / "fkgl_slope_change_decomposition.png",
        metric_label="FKGL",
        sentence_column="sentence_length_contribution_per_year",
        second_column="syllables_word_contribution_per_year",
        second_label="syllables/word",
        residual_column="formula_reconciliation_contribution_per_year",
        residual_label="formula residual",
    )

    print(f"Saved ARI/FKGL decomposition tables to {table_dir}")
    print(f"Saved ARI/FKGL decomposition figures to {visual_dir}")
    print("\nARI decomposition")
    print(ari_summary.to_string(index=False))
    print("\nFKGL decomposition")
    print(fkgl_summary.to_string(index=False))


def build_ari_components(monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"month_ts", ARI_COLUMN, WORDS_PER_SENTENCE_COLUMN}
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"Monthly trends are missing required columns: {', '.join(sorted(missing))}")

    result = monthly[[column for column in ("month_ts", "paper_count") if column in monthly.columns]].copy()
    ari = pd.to_numeric(monthly[ARI_COLUMN], errors="coerce")
    words_per_sentence = pd.to_numeric(monthly[WORDS_PER_SENTENCE_COLUMN], errors="coerce")
    result[f"{ARI_TOTAL}_monthly_mean"] = ari
    result[f"{ARI_SENTENCE_COMPONENT}_monthly_mean"] = ARI_SENTENCE_COEFFICIENT * words_per_sentence
    # Characters/word are not retained in monthly tables, so recover the term
    # implied by the ARI identity. This can absorb small tokenization differences.
    result[f"{ARI_CHARACTER_COMPONENT}_monthly_mean"] = (
        ari - result[f"{ARI_SENTENCE_COMPONENT}_monthly_mean"] - ARI_INTERCEPT
    )
    return result


def build_readability_components(monthly: pd.DataFrame) -> pd.DataFrame:
    required = {
        "month_ts",
        ARI_COLUMN,
        FKGL_COLUMN,
        WORDS_PER_SENTENCE_COLUMN,
        SYLLABLES_PER_WORD_COLUMN,
    }
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"Monthly trends are missing required columns: {', '.join(sorted(missing))}")

    result = build_ari_components(monthly)
    fkgl = pd.to_numeric(monthly[FKGL_COLUMN], errors="coerce")
    words_per_sentence = pd.to_numeric(monthly[WORDS_PER_SENTENCE_COLUMN], errors="coerce")
    syllables_per_word = pd.to_numeric(monthly[SYLLABLES_PER_WORD_COLUMN], errors="coerce")
    result[f"{FKGL_TOTAL}_monthly_mean"] = fkgl
    result[f"{FKGL_SENTENCE_COMPONENT}_monthly_mean"] = FKGL_SENTENCE_COEFFICIENT * words_per_sentence
    result[f"{FKGL_SYLLABLE_COMPONENT}_monthly_mean"] = FKGL_SYLLABLE_COEFFICIENT * syllables_per_word
    # textstat and the pipeline can tokenize words/sentences differently. Keep
    # that discrepancy separate instead of assigning it to syllabic complexity.
    result[f"{FKGL_RECONCILIATION_COMPONENT}_monthly_mean"] = (
        fkgl
        - result[f"{FKGL_SENTENCE_COMPONENT}_monthly_mean"]
        - result[f"{FKGL_SYLLABLE_COMPONENT}_monthly_mean"]
        - FKGL_INTERCEPT
    )
    return result


def fit_component_its(components: pd.DataFrame) -> pd.DataFrame:
    features = [
        ARI_TOTAL,
        ARI_SENTENCE_COMPONENT,
        ARI_CHARACTER_COMPONENT,
        FKGL_TOTAL,
        FKGL_SENTENCE_COMPONENT,
        FKGL_SYLLABLE_COMPONENT,
        FKGL_RECONCILIATION_COMPONENT,
    ]
    stats = compute_interrupted_time_series(components, features)
    return stats[stats["feature"].isin(features)].copy() if not stats.empty else stats


def summarize_ari_decomposition(dataset: str, stats: pd.DataFrame) -> dict[str, object]:
    slopes = stats.set_index("feature")["slope_change_per_year"]
    required = {ARI_TOTAL, ARI_SENTENCE_COMPONENT, ARI_CHARACTER_COMPONENT}
    if not required.issubset(slopes.index):
        raise ValueError(f"ITS output for {dataset} is missing an ARI component")

    total = float(slopes[ARI_TOTAL])
    sentence = float(slopes[ARI_SENTENCE_COMPONENT])
    character = float(slopes[ARI_CHARACTER_COMPONENT])
    absolute_total = abs(sentence) + abs(character)
    return {
        "dataset": dataset,
        "ari_slope_change_per_year": total,
        "sentence_length_contribution_per_year": sentence,
        "implied_character_word_contribution_per_year": character,
        "sentence_length_absolute_share": abs(sentence) / absolute_total if absolute_total else np.nan,
        "implied_character_word_absolute_share": abs(character) / absolute_total if absolute_total else np.nan,
        "dominant_component": (
            "sent. length" if abs(sentence) > abs(character) else "implied chars/word"
        ),
        "reconstruction_error": total - sentence - character,
    }


# Backward-compatible name used by the original ARI-only tests and notebooks.
summarize_decomposition = summarize_ari_decomposition


def summarize_fkgl_decomposition(dataset: str, stats: pd.DataFrame) -> dict[str, object]:
    slopes = stats.set_index("feature")["slope_change_per_year"]
    required = {
        FKGL_TOTAL,
        FKGL_SENTENCE_COMPONENT,
        FKGL_SYLLABLE_COMPONENT,
        FKGL_RECONCILIATION_COMPONENT,
    }
    if not required.issubset(slopes.index):
        raise ValueError(f"ITS output for {dataset} is missing an FKGL component")

    total = float(slopes[FKGL_TOTAL])
    sentence = float(slopes[FKGL_SENTENCE_COMPONENT])
    syllable = float(slopes[FKGL_SYLLABLE_COMPONENT])
    reconciliation = float(slopes[FKGL_RECONCILIATION_COMPONENT])
    component_changes = {
        "sent. length": sentence,
        "syllables/word": syllable,
        "formula residual": reconciliation,
    }
    absolute_total = sum(abs(value) for value in component_changes.values())
    return {
        "dataset": dataset,
        "fkgl_slope_change_per_year": total,
        "sentence_length_contribution_per_year": sentence,
        "syllables_word_contribution_per_year": syllable,
        "formula_reconciliation_contribution_per_year": reconciliation,
        "sentence_length_absolute_share": abs(sentence) / absolute_total if absolute_total else np.nan,
        "syllables_word_absolute_share": abs(syllable) / absolute_total if absolute_total else np.nan,
        "formula_reconciliation_absolute_share": (
            abs(reconciliation) / absolute_total if absolute_total else np.nan
        ),
        "dominant_component": max(component_changes, key=lambda key: abs(component_changes[key])),
        "reconstruction_error": total - sentence - syllable - reconciliation,
    }


def save_decomposition_plot(
    summary: pd.DataFrame,
    path: Path,
    *,
    metric_label: str,
    sentence_column: str,
    second_column: str,
    second_label: str,
    residual_column: str | None = None,
    residual_label: str | None = None,
) -> None:
    # Increase PLOT_WIDTH_IN if labels
    # become crowded, or ROW_SPACING for more vertical separation.
    PLOT_WIDTH_IN = 1.5
    PLOT_HEIGHT_IN = 1.5
    ROW_SPACING = 0.65
    AXIS_FONT_SIZE = 5.5
    LEGEND_WIDTH_IN = 2
    LEGEND_HEIGHT_IN = 0.3
    LEGEND_COLUMNS = 2

    plot = summary.copy()
    labels = plot["dataset"].map(
        {
            "arxiv_ai_abstracts": "AI",
            "arxiv_qbio_abstracts": "q-bio",
            "arxiv_stat_abstracts": "stat",
            "medarxiv_abstracts": "medRxiv",
        }
    ).fillna(plot["dataset"])
    y = np.arange(len(plot)) * ROW_SPACING
    component_count = 3 if residual_column is not None else 2
    height = 0.25 if component_count == 3 else 0.3
    offsets = np.linspace(
        -height * (component_count - 1) / 2,
        height * (component_count - 1) / 2,
        component_count,
    )
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_IN, PLOT_HEIGHT_IN))
    sentence_values = plot[sentence_column].to_numpy(dtype=float)
    second_values = plot[second_column].to_numpy(dtype=float)
    ax.barh(
        y + offsets[0], sentence_values, height=height,
        color=[sign_color(value) for value in sentence_values], hatch=HATCHES[0],
        edgecolor=DARK_GREY, linewidth=0.4, label="sent. length",
    )
    ax.barh(
        y + offsets[1], second_values, height=height,
        color=[sign_color(value) for value in second_values],
        hatch=HATCHES[1], edgecolor=DARK_GREY, linewidth=0.4, label=second_label,
    )
    if residual_column is not None:
        residual_values = plot[residual_column].to_numpy(dtype=float)
        ax.barh(
            y + offsets[2], residual_values, height=height,
            color=[sign_color(value) for value in residual_values],
            hatch=HATCHES[3], edgecolor=DARK_GREY, linewidth=0.4, label=residual_label,
        )
    ax.axvline(0, color=DARK_GREY, linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"contribution to {metric_label} $\Delta$/year", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXIS_FONT_SIZE, length=2, pad=0.8)
    ax.grid(axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(fig)

    handles = [
        Patch(facecolor=WHITE, edgecolor=DARK_GREY, hatch=HATCHES[0], label="sent. length"),
        Patch(
            facecolor=WHITE, edgecolor=DARK_GREY, hatch=HATCHES[1],
            label=second_label,
        ),
    ]
    if residual_column is not None:
        handles.append(
            Patch(
                facecolor=WHITE, edgecolor=DARK_GREY, hatch=HATCHES[3],
                label=residual_label,
            )
        )
    handles.extend(
        [
            Patch(facecolor=GREEN, edgecolor=DARK_GREY, label="pos. contribution"),
            Patch(facecolor=ORCHID, edgecolor=DARK_GREY, label="neg. contribution"),
        ]
    )
    legend_path = path.with_name(f"{path.stem}_legend{path.suffix}")
    legend_fig, legend_ax = plt.subplots(figsize=(LEGEND_WIDTH_IN, LEGEND_HEIGHT_IN))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=LEGEND_COLUMNS,
        frameon=False,
        fontsize=5.5,
        handlelength=1.1,
        handleheight=0.8,
        columnspacing=0.8,
        labelspacing=0.35,
    )
    legend_fig.savefig(
        legend_path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01,
    )
    plt.close(legend_fig)


if __name__ == "__main__":
    main()
