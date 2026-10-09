"""Build paper-ready full ITS and counterfactual-excess appendix tables.

This script is intentionally read-only with respect to primary analysis outputs.
It joins the four abstract-corpus CSVs and writes derived tables under
``data/analysis/manuscript``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from not_an_llm.analysis.label_map import pretty_feature_label


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "analysis" / "manuscript"
CORPORA = {
    "arxiv_ai_abstracts": "arXiv AI",
    "arxiv_qbio_abstracts": "arXiv q-bio",
    "arxiv_stat_abstracts": "arXiv statistics",
    "medarxiv_abstracts": "medRxiv",
}
CORPUS_COLUMNS = {
    "arXiv AI": "AI",
    "arXiv q-bio": "q-bio",
    "arXiv statistics": "stat",
    "medRxiv": "medRxiv",
}
FAMILY_LABELS = {
    "adjectives": "Adjectives",
    "causal_markers": "Causal markers",
    "certainty_hedging": "Certainty and hedging",
    "contrast_markers": "Contrast markers",
    "emphasis_markers": "Emphasis markers",
    "marker_words": "Marker words",
    "phrases": "Phrases",
    "punctuation": "Punctuation",
    "readability": "Readability",
    "sequential_markers": "Sequential markers",
    "summary_markers": "Summary markers",
    "syntax": "Syntax",
    "verbs": "Verbs",
}
GROUP_AGGREGATES = {
    "marker_words_total_per_1k_words",
    "marker_verbs_total_per_1k_words",
    "marker_adjectives_total_per_1k_words",
    "marker_phrases_total_per_1k_words",
    "sequential_markers_total_per_1k_words",
    "causal_markers_total_per_1k_words",
    "contrast_markers_total_per_1k_words",
    "emphasis_markers_total_per_1k_words",
    "summary_markers_total_per_1k_words",
}
EXCLUDED_INFERENCE_FEATURES = GROUP_AGGREGATES | {"sentence_depth_std"}
TABLE_LABEL_OVERRIDES = {
    "clause_depth": "max tree depth",
    "clause_depth_std": r"tree depth $\sigma$",
}
PLACEBO_FEATURES = (
    "word_across_per_1k_words",
    "em_dash_per_1k_words",
    "avg_syllables_per_word",
    "clause_depth",
    "sequential_marker_moreover_per_1k_words",
)
PLACEBO_YEARS = (2018, 2019, 2020, 2021)


def _table_feature_label(feature: str) -> str:
    return TABLE_LABEL_OVERRIDES.get(feature, pretty_feature_label(feature))


def _read(corpus: str, filename: str) -> pd.DataFrame:
    path = ROOT / "data" / "analysis" / corpus / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _without_aggregates_and_recomputed_q(
    frame: pd.DataFrame, p_column: str, q_column: str
) -> pd.DataFrame:
    """Remove group totals before recomputing within-family BH q-values."""
    frame = frame.loc[~frame["feature"].isin(EXCLUDED_INFERENCE_FEATURES)].copy()
    frame[q_column] = pd.NA
    for _, indices in frame.groupby("family", sort=False).groups.items():
        valid = frame.loc[indices, p_column].dropna()
        if valid.empty:
            continue
        frame.loc[valid.index, q_column] = multipletests(
            valid.astype(float), method="fdr_bh"
        )[1]
    frame[q_column] = pd.to_numeric(frame[q_column])
    return frame


def _fmt_number(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _stars(q: object) -> str:
    if pd.isna(q):
        return ""
    q = float(q)
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""


def _effect_cell(row: pd.Series, stem: str, q_col: str) -> str:
    estimate = row[f"standardized_{stem}"]
    low = row[f"standardized_{stem}_ci_low"]
    high = row[f"standardized_{stem}_ci_high"]
    return f"{_fmt_number(estimate)} [{_fmt_number(low)}, {_fmt_number(high)}]{_stars(row[q_col])}"


def build_long_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for corpus, corpus_label in CORPORA.items():
        its = _without_aggregates_and_recomputed_q(
            _read(corpus, "its_stats.csv"), "slope_change_p", "slope_change_q"
        )
        one = _without_aggregates_and_recomputed_q(
            _read(corpus, "first_post_year_counterfactual_excess.csv"),
            "first_post_year_excess_p",
            "first_post_year_excess_q",
        )
        two = _without_aggregates_and_recomputed_q(
            _read(corpus, "first_two_year_counterfactual_excess.csv"),
            "first_two_year_excess_p",
            "first_two_year_excess_q",
        )
        merged = its.merge(one, on=["feature", "family"], how="outer", suffixes=("", "_one"))
        merged = merged.merge(two, on=["feature", "family"], how="outer", suffixes=("", "_two"))
        out = pd.DataFrame(
            {
                "corpus": corpus_label,
                "feature": merged["feature"],
                "family": merged["family"],
                "its_standardized_slope_change_per_year": merged["standardized_slope_change_per_year"],
                "its_ci_low": merged["standardized_slope_change_per_year_ci_low"],
                "its_ci_high": merged["standardized_slope_change_per_year_ci_high"],
                "its_q": merged["slope_change_q"],
                "excess_2023_standardized": merged["standardized_first_post_year_excess"],
                "excess_2023_ci_low": merged["standardized_first_post_year_excess_ci_low"],
                "excess_2023_ci_high": merged["standardized_first_post_year_excess_ci_high"],
                "excess_2023_q": merged["first_post_year_excess_q"],
                "excess_2023_2024_standardized": merged["standardized_first_two_year_excess"],
                "excess_2023_2024_ci_low": merged["standardized_first_two_year_excess_ci_low"],
                "excess_2023_2024_ci_high": merged["standardized_first_two_year_excess_ci_high"],
                "excess_2023_2024_q": merged["first_two_year_excess_q"],
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True).sort_values(["family", "feature", "corpus"])


def build_wide_table(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for (family, feature), group in long.groupby(["family", "feature"], sort=True):
        row: dict[str, str] = {
            "family": family,
            "feature": feature,
            "label": _table_feature_label(str(feature)),
        }
        for _, values in group.iterrows():
            prefix = str(values["corpus"])
            row[f"{prefix} ITS"] = (
                f"{_fmt_number(values['its_standardized_slope_change_per_year'])} "
                f"[{_fmt_number(values['its_ci_low'])}, {_fmt_number(values['its_ci_high'])}]"
                f"{_stars(values['its_q'])}"
            )
            row[f"{prefix} excess 2023"] = (
                f"{_fmt_number(values['excess_2023_standardized'])} "
                f"[{_fmt_number(values['excess_2023_ci_low'])}, {_fmt_number(values['excess_2023_ci_high'])}]"
                f"{_stars(values['excess_2023_q'])}"
            )
            row[f"{prefix} excess 2023-2024"] = (
                f"{_fmt_number(values['excess_2023_2024_standardized'])} "
                f"[{_fmt_number(values['excess_2023_2024_ci_low'])}, {_fmt_number(values['excess_2023_2024_ci_high'])}]"
                f"{_stars(values['excess_2023_2024_q'])}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _latex_stars(q: object) -> str:
    stars = _stars(q)
    return rf"$^{{{stars}}}$" if stars else ""


def _latex_cell(row: pd.Series, estimand: str) -> str:
    if estimand == "its":
        value, low, high, q = (
            row["its_standardized_slope_change_per_year"], row["its_ci_low"],
            row["its_ci_high"], row["its_q"],
        )
    elif estimand == "excess_2023":
        value, low, high, q = (
            row["excess_2023_standardized"], row["excess_2023_ci_low"],
            row["excess_2023_ci_high"], row["excess_2023_q"],
        )
    else:
        value, low, high, q = (
            row["excess_2023_2024_standardized"], row["excess_2023_2024_ci_low"],
            row["excess_2023_2024_ci_high"], row["excess_2023_2024_q"],
        )
    if pd.isna(value):
        return "--"
    return (
        _fmt_number(value) + _latex_stars(q)
        + " [" + _fmt_number(low) + ", " + _fmt_number(high) + "]"
    )


def write_latex(long: pd.DataFrame, path: Path, estimand: str) -> None:
    descriptions = {
        "its": (
            "Abstract-level interrupted-time-series results. Cells report the annualized post-intervention change in slope ($\\beta_3$), standardized by the pre-intervention monthly standard deviation, with 95\% confidence intervals. Positive values indicate acceleration relative to the pre-intervention slope; negative values indicate deceleration or reversal.",
            r"tab:full-feature-its",
        ),
        "excess_2023": (
            "Abstract-level counterfactual-excess results for 2023. Cells report the paper-count-weighted mean difference between observed monthly values and projections from the pre-intervention trend, standardized by the pre-intervention monthly standard deviation, with 95\% confidence intervals.",
            r"tab:full-feature-excess-2023",
        ),
        "excess_2023_2024": (
            "Abstract-level counterfactual-excess results for 2023--2024. Cells report the paper-count-weighted mean difference between observed monthly values and projections from the pre-intervention trend, standardized by the pre-intervention monthly standard deviation, with 95\% confidence intervals.",
            r"tab:full-feature-excess-2023-2024",
        ),
    }
    caption, label = descriptions[estimand]
    lines = [
        r"\clearpage",
        r"\onecolumn",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{@{\extracolsep{\fill}}lllll@{}}",
        r"\toprule",
        r"Feature & AI & q-bio & stat & medRxiv \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Feature & AI & q-bio & stat & medRxiv \\",
        r"\midrule",
        r"\endhead",
    ]
    previous_family: str | None = None
    for (family, feature), group in long.groupby(["family", "feature"], sort=True):
        if family != previous_family:
            if previous_family is not None:
                lines.append(r"\addlinespace[2pt]")
            family_label = FAMILY_LABELS.get(str(family), str(family).replace("_", " ").title())
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{family_label}}}}} \\")
        by_corpus = {str(row["corpus"]): row for _, row in group.iterrows()}
        cells = [_latex_cell(by_corpus[corpus], estimand) for corpus in CORPUS_COLUMNS]
        lines.append(f"{_table_feature_label(str(feature))} & " + " & ".join(cells) + r" \\")
        previous_family = str(family)
    lines.extend([
        r"\bottomrule",
        rf"\caption{{{caption} $^*q<.05$, $^{{**}}q<.01$, $^{{***}}q<.001$ after within-family Benjamini--Hochberg correction.}}\label{{{label}}}\\",
        r"\end{longtable}",
        r"\endgroup",
        r"\clearpage",
        r"\twocolumn",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_operationalization_table(path: Path) -> None:
    """Write a compact implementation-grounded feature definition table."""
    rows = [
        (
            "Single-word lexical and discourse markers",
            "Exact whole-token match in lowercased lemmatized text",
            r"$1000c/(N_{\alpha}+1)$",
            "Document value; monthly mean",
            "Marker words; verbs; adjectives; sequential, causal, contrast, emphasis, and summary markers",
        ),
        (
            "Multiword phrases",
            "Exact whole-phrase match in lowercased normalized surface text",
            r"$1000c/(N_{\alpha}+1)$",
            "Document value; monthly mean",
            "Phrases",
        ),
        (
            "Colon, semicolon, em dash",
            r"Surface-form count after dash normalization (em dash mapped to \texttt{--})",
            r"$1000c/(N_{\alpha}+1)$",
            "Document value; monthly mean",
            "Punctuation",
        ),
        (
            "Certainty and hedge terms",
            "Exact-token lexicon count in lemmatized text",
            r"$c/(N_{\alpha}+1)$",
            "Document ratio; monthly mean",
            "Certainty/hedging",
        ),
        (
            "Words/sentence; syllables/word",
            r"Alphabetic spaCy tokens per sentence; \texttt{textstat} syllable count per alphabetic token",
            "Words or syllables",
            "Document mean; monthly mean",
            "Readability",
        ),
        (
            "FRE, FKGL, Dale--Chall, SMOG, ARI, Gunning Fog",
            r"Corresponding \texttt{textstat} implementation applied to normalized document text",
            "Native score",
            "Document score; monthly mean",
            "Readability",
        ),
        (
            "Maximum dependency-tree depth",
            "Largest root-to-terminal-token path over sentence dependency trees; root and terminal included",
            "Tokens in path",
            "Document maximum; monthly mean",
            "Syntax",
        ),
        (
            r"Tree-depth $\sigma$",
            "Sample SD of sentence-level dependency-tree depths; zero for fewer than two sentences",
            "Tokens",
            "Document SD; monthly mean",
            "Syntax",
        ),
        (
            r"Dependency length and dependency-length $\sigma$",
            "Absolute token--head index distance over non-root arcs; punctuation retained. Variation is the sample SD of sentence means",
            "Tokens",
            "Document mean or SD; monthly mean",
            "Syntax",
        ),
        (
            "Dependency-label entropy",
            r"Shannon entropy (natural logarithm) of dependency labels; \texttt{punct} excluded and \texttt{ROOT} retained",
            "Nats",
            "Document entropy; monthly mean",
            "Syntax",
        ),
        (
            r"Coordination count and coordination/sentence $\sigma$",
            r"Count of \texttt{cc} tokens; variation is the sample SD of sentence-level \texttt{cc} counts",
            "Count",
            "Document count or SD; monthly mean",
            "Syntax",
        ),
        (
            "List of three",
            r"\texttt{and}/\texttt{or} coordination containing exactly three conjuncts; each coordination head counted once",
            r"$1000c/(N_{\alpha}+1)$",
            "Document frequency; monthly mean",
            "Syntax",
        ),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}>{\raggedright\arraybackslash}p{0.20\textwidth}@{}}",
        r"\toprule",
        r"Feature(s) & Definition or matching rule & Unit & Aggregation & Inferential family \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Operational definitions of the reported feature families. $c$ denotes the document-level count and $N_{\alpha}$ the number of alphabetic spaCy tokens. Monthly values are arithmetic means over documents; monthly paper counts are subsequently used as weights in the inferential models. Group aggregates and the duplicated sentence-depth-variation column are excluded from inference.}",
        r"\label{tab:feature-operationalization}",
        r"\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_placebo_comparison() -> pd.DataFrame:
    """Join the Figure 1 features across placebo and actual interventions."""
    frames: list[pd.DataFrame] = []
    for corpus, corpus_label in CORPORA.items():
        actual = _read(corpus, "its_stats.csv")
        actual = actual.loc[actual["feature"].isin(PLACEBO_FEATURES)].copy()
        actual["cutoff"] = "Nov. 2022"
        placebo = _read(corpus, "its_placebo_stats.csv")
        placebo = placebo.loc[placebo["feature"].isin(PLACEBO_FEATURES)].copy()
        placebo["cutoff"] = placebo["placebo_year"].astype("Int64").astype(str)
        joined = pd.concat([placebo, actual], ignore_index=True)
        joined.insert(0, "corpus", corpus_label)
        frames.append(joined[[
            "corpus", "feature", "family", "cutoff", "intervention_date",
            "n_pre_months", "n_post_months",
            "standardized_slope_change_per_year",
            "standardized_slope_change_per_year_ci_low",
            "standardized_slope_change_per_year_ci_high",
        ]])
    return pd.concat(frames, ignore_index=True)


def write_placebo_latex(frame: pd.DataFrame, path: Path) -> None:
    cutoffs = [str(year) for year in PLACEBO_YEARS] + ["Nov. 2022"]
    compact_corpus_labels = {
        "arXiv AI": "AI",
        "arXiv q-bio": "q-bio",
        "arXiv statistics": "stat",
        "medRxiv": "medRxiv",
    }
    compact_feature_labels = {
        "word_across_per_1k_words": r"$\it{across}$",
        "em_dash_per_1k_words": "em dash",
        "avg_syllables_per_word": "syll./word",
        "clause_depth": "max tree depth",
        "sequential_marker_moreover_per_1k_words": r"$\it{moreover}$",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Corpus & Feature & 2018 & 2019 & 2020 & 2021 & Actual \\",
        r"\midrule",
    ]
    for corpus_label in CORPORA.values():
        corpus_rows = frame.loc[frame["corpus"] == corpus_label]
        for index, feature in enumerate(PLACEBO_FEATURES):
            feature_rows = corpus_rows.loc[corpus_rows["feature"] == feature]
            by_cutoff = {str(row["cutoff"]): row for _, row in feature_rows.iterrows()}
            cells = []
            for cutoff in cutoffs:
                row = by_cutoff.get(cutoff)
                if row is None or pd.isna(row["standardized_slope_change_per_year"]):
                    cells.append("--")
                else:
                    value = _fmt_number(row["standardized_slope_change_per_year"])
                    cells.append(rf"\textbf{{{value}}}" if cutoff == "Nov. 2022" else value)
            corpus_cell = compact_corpus_labels[corpus_label] if index == 0 else ""
            lines.append(
                f"{corpus_cell} & {compact_feature_labels[feature]} & "
                + " & ".join(cells) + r" \\"
            )
        if corpus_label != list(CORPORA.values())[-1]:
            lines.append(r"\addlinespace[3pt]")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Temporal placebo checks for selected main-result features. Cells report standardized annual ITS slope changes for January placebo interventions and the prespecified November 2022 intervention (Actual, bold); dashes indicate unavailable estimates.}",
        r"\label{tab:placebo-interventions}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placebo_section(path: Path) -> None:
    lines = [
        r"\subsection{Temporal Placebo Checks}",
        r"\label{app:placebo-checks}",
        (
            "We refitted the interrupted-time-series models using placebo interventions in January "
            "2018--2021 (Table~\\ref{tab:placebo-interventions}); medRxiv supports only the 2021 "
            "placebo because of its shorter history. For a compact illustration, we show one feature "
            "from each prespecified group in the main results table, selecting the feature with the "
            "largest mean absolute primary effect across corpora. Selection did not use the placebo "
            "estimates, and the displayed subset is not a summary of all placebo tests. Earlier cutoffs "
            "often produced changes in the same "
            "direction, indicating pre-existing drift. However, the prespecified November 2022 "
            "intervention yielded the largest absolute effect in 19 of 20 feature--corpus comparisons. "
            "The results therefore suggest that these trajectories intensified after ChatGPT's release, "
            "rather than originating entirely at that point."
        ),
        r"\input{data/analysis/manuscript/placebo_intervention_comparison_table.tex}",
    ]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long = build_long_table()
    wide = build_wide_table(long)
    long.to_csv(OUTPUT_DIR / "full_feature_results_long.csv", index=False)
    wide.to_csv(OUTPUT_DIR / "full_feature_results_wide.csv", index=False)
    write_latex(long, OUTPUT_DIR / "full_feature_results_longtable.tex", "its")
    write_latex(long, OUTPUT_DIR / "full_feature_excess_2023_longtable.tex", "excess_2023")
    write_latex(
        long,
        OUTPUT_DIR / "full_feature_excess_2023_2024_longtable.tex",
        "excess_2023_2024",
    )
    write_feature_operationalization_table(
        OUTPUT_DIR / "feature_operationalization_table.tex"
    )
    placebo = build_placebo_comparison()
    placebo.to_csv(OUTPUT_DIR / "placebo_intervention_comparison.csv", index=False)
    write_placebo_latex(
        placebo, OUTPUT_DIR / "placebo_intervention_comparison_table.tex"
    )
    write_placebo_section(OUTPUT_DIR / "placebo_appendix_section.tex")
    print(
        f"Wrote {len(long)} corpus-feature source rows and three "
        f"{long['feature'].nunique()}-row manuscript tables "
        f"(group totals and one duplicate depth feature excluded)."
    )


if __name__ == "__main__":
    main()
