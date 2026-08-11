from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.interrupted_time_series import compute_interrupted_time_series
from not_an_llm.analysis.visual_style import (
    BLUE,
    CATEGORICAL_COLORS,
    DARK_GREY,
    GREEN,
    MARKERS,
    ORCHID,
    WHITE,
)


DEFAULT_DATASETS = (
    "arxiv_ai_abstracts",
    "arxiv_qbio_abstracts",
    "arxiv_stat_abstracts",
    "medarxiv_abstracts",
)

READABILITY_METRICS = (
    "automated_readability_index",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "dale_chall",
    "flesch_reading_ease",
)
READABILITY_DIRECTIONS = {metric: -1.0 if metric == "flesch_reading_ease" else 1.0 for metric in READABILITY_METRICS}

BASE_SYNTAX_FEATURES = (
    "clause_depth",
    "clause_depth_std",
    "dependency_entropy",
    "dependency_length",
    "dependency_length_std",
    "coordination_count",
    "coordination_per_sentence_std",
    "sentence_depth_std",
    "list_of_three_per_1k_words",
)
FORMULA_CONTROLS = ("avg_words_per_sentence", "avg_syllables_per_word")
FAMILY_MARKERS = {"syntax": MARKERS[0], "dependency role": MARKERS[1], "dependency bigram": MARKERS[2]}
FAMILY_COLORS = {
    "syntax": CATEGORICAL_COLORS[0],
    "dependency role": CATEGORICAL_COLORS[2],
    "dependency bigram": CATEGORICAL_COLORS[3],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Associate abstract syntax with readability and align it with syntax ITS changes."
    )
    parser.add_argument("--analysis-dir", default="data/analysis")
    parser.add_argument("--visuals-dir", default="data/visuals")
    parser.add_argument("--output-name", default="syntax_readability_abstracts")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--top-bigrams", type=int, default=40)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--max-documents-per-dataset",
        type=int,
        default=0,
        help="Optional deterministic cap for development; 0 uses all documents.",
    )
    parser.add_argument(
        "--skip-bigrams",
        action="store_true",
        help="Run with stored syntax and dependency roles only; do not parse edge bigrams.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    table_dir = analysis_dir / args.output_name
    visual_dir = Path(args.visuals_dir) / args.output_name
    cache_dir = table_dir / "cache"
    table_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [dataset for dataset in args.datasets if not (analysis_dir / dataset / "features.jsonl").exists()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing document-level feature files for: " + joined + ". "
            "Run the ordinary abstract analyze step first; this standalone script does not modify the pipeline."
        )

    documents = load_document_features(
        analysis_dir,
        args.datasets,
        chunk_size=args.chunk_size,
        max_documents_per_dataset=args.max_documents_per_dataset,
    )
    if not args.skip_bigrams:
        bigram_maps = load_or_build_bigram_cache(
            documents,
            cache_dir,
            spacy_model=args.spacy_model,
        )
    else:
        bigram_maps = {}

    model_frame, syntax_columns = build_model_frame(
        documents,
        bigram_maps=bigram_maps,
        top_bigrams=args.top_bigrams,
    )
    bigram_columns = [
        column for column in syntax_columns if column.startswith("dep_bigram__")
    ]
    if not bigram_columns:
        raise SystemExit(
            "No dependency-edge bigram predictors are available. "
            "Run without --skip-bigrams and verify the bigram cache."
        )
    save_document_model_frame(model_frame, table_dir)

    model_specs = [
        ("total_association", "ari_complexity", ()),
        ("total_association", "fkgl_complexity", ()),
        ("total_association", "readability_composite", ()),
        ("beyond_formula", "readability_composite", FORMULA_CONTROLS),
    ]
    coefficient_parts = []
    for model_spec, outcome, extra_controls in model_specs:
        result = fit_syntax_association(
            model_frame,
            bigram_columns,
            outcome=outcome,
            model_spec=model_spec,
            extra_controls=list(extra_controls),
            bootstrap=args.bootstrap,
            permutation_repeats=args.permutation_repeats,
        )
        coefficient_parts.append(result)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    coefficients.to_csv(table_dir / "syntax_readability_coefficients.csv", index=False)

    syntax_its = compute_syntax_its(model_frame, bigram_columns)
    if not syntax_its.empty:
        syntax_its["family"] = syntax_its["feature"].map(feature_family)
    syntax_its.to_csv(table_dir / "syntax_its_changes.csv", index=False)
    alignment = build_alignment_table(coefficients, syntax_its)
    alignment.to_csv(table_dir / "syntax_readability_its_alignment.csv", index=False)

    save_coefficient_plot(
        coefficients,
        visual_dir / "syntax_readability_coefficients.png",
        outcome="readability_composite",
        model_spec="total_association",
    )
    save_alignment_plot(
        alignment,
        visual_dir / "syntax_readability_its_alignment.png",
        outcome="readability_composite",
        model_spec="total_association",
    )

    print(f"Saved syntax-readability tables to {table_dir}")
    print(f"Saved syntax-readability figures to {visual_dir}")
    if alignment.empty:
        print("No alignment scores were available; inspect the coefficient and ITS tables for missing estimates.")
    else:
        print(
            alignment.query("outcome == 'readability_composite' and model_spec == 'total_association'")
            .sort_values("abs_alignment_score", ascending=False)
            .head(15)
            .to_string(index=False)
        )


def load_document_features(
    analysis_dir: Path,
    datasets: list[str],
    *,
    chunk_size: int,
    max_documents_per_dataset: int,
) -> pd.DataFrame:
    wanted = {
        "paperId", "publicationDate", "year", "month", "text_clean", "word_count",
        "dependency_distribution", *READABILITY_METRICS, *BASE_SYNTAX_FEATURES, *FORMULA_CONTROLS,
        "topic_id",
    }
    parts = []
    for dataset in datasets:
        path = analysis_dir / dataset / "features.jsonl"
        dataset_parts = []
        offset = 0
        for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
            available = [column for column in chunk.columns if column in wanted]
            frame = chunk[available].copy()
            frame.insert(0, "dataset", dataset)
            fallback = [f"{dataset}:{offset + index}" for index in range(len(frame))]
            if "paperId" in frame.columns:
                ids = frame["paperId"].astype("string")
                frame["document_key"] = [
                    f"{dataset}:{paper_id}" if pd.notna(paper_id) and str(paper_id).strip() else fallback[index]
                    for index, paper_id in enumerate(ids)
                ]
            else:
                frame["document_key"] = fallback
            offset += len(frame)
            dataset_parts.append(frame)
        dataset_frame = pd.concat(dataset_parts, ignore_index=True)
        if max_documents_per_dataset > 0 and len(dataset_frame) > max_documents_per_dataset:
            dataset_frame = stratified_year_sample(dataset_frame, max_documents_per_dataset)
        parts.append(dataset_frame)
    documents = pd.concat(parts, ignore_index=True)
    documents["month_ts"] = resolve_month_timestamp(documents)
    return documents.dropna(subset=["month_ts"]).reset_index(drop=True)


def stratified_year_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if "year" not in frame.columns or frame["year"].isna().all():
        return frame.sample(n=n, random_state=42)
    fractions = frame.groupby("year", dropna=False).size() / len(frame)
    sampled = []
    for year, group in frame.groupby("year", dropna=False):
        target = max(1, int(round(n * fractions.loc[year])))
        sampled.append(group.sample(n=min(target, len(group)), random_state=42))
    result = pd.concat(sampled, ignore_index=True)
    return result.sample(n=min(n, len(result)), random_state=42).reset_index(drop=True)


def resolve_month_timestamp(frame: pd.DataFrame) -> pd.Series:
    if "publicationDate" in frame.columns:
        dates = pd.to_datetime(frame["publicationDate"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if "year" in frame.columns:
        year = pd.to_numeric(frame["year"], errors="coerce")
        month_source = frame["month"] if "month" in frame.columns else pd.Series(1, index=frame.index)
        month = pd.to_numeric(month_source, errors="coerce").fillna(1)
        fallback = pd.to_datetime(
            {"year": year, "month": month.clip(1, 12), "day": 1}, errors="coerce"
        )
        dates = dates.fillna(fallback)
    return dates.dt.to_period("M").dt.to_timestamp()


def save_document_model_frame(frame: pd.DataFrame, table_dir: Path) -> Path:
    """Prefer compact Parquet, with a dependency-free compressed CSV fallback."""
    persisted = frame.drop(columns=["text_clean", "dependency_distribution"], errors="ignore")
    parquet_path = table_dir / "syntax_readability_documents.parquet"
    try:
        persisted.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = table_dir / "syntax_readability_documents.csv.gz"
        persisted.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def load_or_build_bigram_cache(
    documents: pd.DataFrame,
    cache_dir: Path,
    *,
    spacy_model: str,
) -> dict[str, dict[str, float]]:
    cache_path = cache_dir / "document_dependency_bigrams.jsonl.gz"
    expected_keys = set(documents["document_key"].astype(str))
    cached: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        cached = read_bigram_cache(cache_path)
        if expected_keys.issubset(cached):
            return {key: cached[key] for key in expected_keys}

    # Retain valid records from an interrupted run and parse only documents
    # that are still missing. Appending creates another valid gzip member;
    # Python's gzip reader transparently reads concatenated members.
    result = {key: value for key, value in cached.items() if key in expected_keys}
    missing_keys = expected_keys.difference(result)

    import spacy

    nlp = spacy.load(spacy_model, disable=["ner", "textcat"])
    nlp.max_length = 2_000_000
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_documents = documents[
        documents["document_key"].astype(str).isin(missing_keys)
    ]
    texts = missing_documents["text_clean"].fillna("").astype(str).tolist()
    keys = missing_documents["document_key"].astype(str).tolist()
    mode = "at" if cache_path.exists() else "wt"
    with gzip.open(cache_path, mode, encoding="utf-8") as handle:
        for key, doc in zip(keys, nlp.pipe(texts, batch_size=128, n_process=1), strict=False):
            counts: Counter[str] = Counter()
            total = 0
            for token in doc:
                if token.dep_ == "punct" or token.head == token or token.head.dep_ == "punct":
                    continue
                total += 1
                counts[f"{token.head.dep_}->{token.dep_}"] += 1
            proportions = {bigram: count / total for bigram, count in counts.items()} if total else {}
            result[key] = proportions
            handle.write(json.dumps({"document_key": key, "bigrams": proportions}) + "\n")
    return result


def read_bigram_cache(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result[str(record["document_key"])] = {
                str(key): float(value) for key, value in record.get("bigrams", {}).items()
            }
    return result


def build_model_frame(
    documents: pd.DataFrame,
    *,
    bigram_maps: dict[str, dict[str, float]],
    top_bigrams: int,
) -> tuple[pd.DataFrame, list[str]]:
    frame = documents.copy()
    frame = add_readability_outcomes(frame)
    syntax_columns = [column for column in BASE_SYNTAX_FEATURES if column in frame.columns]

    role_maps = frame.get("dependency_distribution", pd.Series([{}] * len(frame), index=frame.index))
    role_totals: Counter[str] = Counter()
    normalized_roles = []
    for value in role_maps:
        distribution = value if isinstance(value, dict) else {}
        total = sum(float(count) for count in distribution.values())
        normalized = {str(role): float(count) / total for role, count in distribution.items()} if total else {}
        normalized_roles.append(normalized)
        role_totals.update({role: float(count) for role, count in distribution.items()})
    for role, _ in role_totals.most_common():
        column = f"dep_role__{role}"
        frame[column] = [mapping.get(role, 0.0) for mapping in normalized_roles]
        syntax_columns.append(column)

    if bigram_maps:
        totals: Counter[str] = Counter()
        for mapping in bigram_maps.values():
            totals.update(mapping)
        selected = [bigram for bigram, _ in totals.most_common(top_bigrams)]
        keys = frame["document_key"].astype(str)
        for bigram in selected:
            column = f"dep_bigram__{bigram}"
            frame[column] = [bigram_maps.get(key, {}).get(bigram, 0.0) for key in keys]
            syntax_columns.append(column)

    for column in [*syntax_columns, *FORMULA_CONTROLS, "word_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, syntax_columns


def add_readability_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aligned_columns = []
    for metric in READABILITY_METRICS:
        if metric not in result.columns:
            continue
        values = pd.to_numeric(result[metric], errors="coerce")
        z = pd.Series(np.nan, index=result.index, dtype=float)
        for dataset, indices in result.groupby("dataset").groups.items():
            dataset_values = values.loc[indices]
            years = pd.to_numeric(result.loc[indices, "year"], errors="coerce")
            pre = dataset_values.loc[years <= 2022].dropna()
            center = float(pre.mean()) if not pre.empty else float(dataset_values.mean())
            scale = float(pre.std(ddof=1)) if len(pre) > 1 else float(dataset_values.std(ddof=1))
            if np.isfinite(scale) and scale > 0:
                z.loc[indices] = (dataset_values - center) / scale * READABILITY_DIRECTIONS[metric]
        column = f"aligned_z__{metric}"
        result[column] = z
        aligned_columns.append(column)
    result["readability_composite"] = result[aligned_columns].mean(axis=1, skipna=True)
    result["ari_complexity"] = result.get("aligned_z__automated_readability_index")
    result["fkgl_complexity"] = result.get("aligned_z__flesch_kincaid_grade")
    return result


def feature_family(feature: str) -> str:
    if feature.startswith("dep_bigram__"):
        return "dependency bigram"
    if feature.startswith("dep_role__"):
        return "dependency role"
    return "syntax"


def fit_syntax_association(
    frame: pd.DataFrame,
    syntax_columns: list[str],
    *,
    outcome: str,
    model_spec: str,
    extra_controls: list[str],
    bootstrap: int,
    permutation_repeats: int,
) -> pd.DataFrame:
    columns = [outcome, "dataset", "year", "topic_id", *syntax_columns, "word_count", *extra_controls]
    columns = list(dict.fromkeys(column for column in columns if column in frame.columns))
    data = frame[columns].dropna(subset=[outcome, "dataset", "year"]).copy()
    syntax = data[syntax_columns].apply(pd.to_numeric, errors="coerce")
    syntax = syntax.fillna(syntax.median()).fillna(0.0)
    syntax_scaler = StandardScaler()
    syntax_x = syntax_scaler.fit_transform(syntax)

    continuous_controls = [column for column in ["word_count", *extra_controls] if column in data.columns]
    control_parts = []
    if continuous_controls:
        controls = data[continuous_controls].apply(pd.to_numeric, errors="coerce")
        controls = controls.fillna(controls.median()).fillna(0.0)
        if "word_count" in controls:
            controls["word_count"] = np.log1p(controls["word_count"].clip(lower=0))
        control_parts.append(StandardScaler().fit_transform(controls))
    categorical_columns = [column for column in ("dataset", "year", "topic_id") if column in data.columns]
    categorical = pd.get_dummies(
        data[categorical_columns].astype(str),
        drop_first=True,
        dtype=float,
    )
    if not categorical.empty:
        control_parts.append(categorical.to_numpy())
    x = np.column_stack([syntax_x, *control_parts]) if control_parts else syntax_x
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(dtype=float)
    groups = data["year"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 3:
        raise ValueError("At least three distinct years are required for grouped cross-validation")
    splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        cv=splits,
        max_iter=20_000,
        n_jobs=1,
        random_state=42,
    ).fit(x, y)
    cv_predictions = np.full(len(y), np.nan)
    permutation = np.zeros(len(syntax_columns), dtype=float)
    rng = np.random.default_rng(42)
    for train, test in splits:
        fold_model = ElasticNet(
            alpha=float(model.alpha_), l1_ratio=float(model.l1_ratio_), max_iter=20_000, random_state=42
        ).fit(x[train], y[train])
        baseline = mean_squared_error(y[test], fold_model.predict(x[test]))
        cv_predictions[test] = fold_model.predict(x[test])
        for feature_index in range(len(syntax_columns)):
            increases = []
            for _ in range(permutation_repeats):
                permuted = x[test].copy()
                permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
                increases.append(mean_squared_error(y[test], fold_model.predict(permuted)) - baseline)
            permutation[feature_index] += float(np.mean(increases)) / len(splits)

    boot_coefficients = np.empty((max(bootstrap, 1), len(syntax_columns)), dtype=float)
    for bootstrap_index in range(max(bootstrap, 1)):
        sample = rng.integers(0, len(y), len(y))
        boot_model = ElasticNet(
            alpha=float(model.alpha_), l1_ratio=float(model.l1_ratio_), max_iter=20_000, random_state=bootstrap_index
        ).fit(x[sample], y[sample])
        boot_coefficients[bootstrap_index] = boot_model.coef_[: len(syntax_columns)]

    rows = []
    for index, feature in enumerate(syntax_columns):
        boot = boot_coefficients[:, index]
        rows.append(
            {
                "model_spec": model_spec,
                "outcome": outcome,
                "feature": feature,
                "family": feature_family(feature),
                "standardized_coefficient": float(model.coef_[index]),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_selection_frequency": float(np.mean(np.abs(boot) > 1e-10)),
                "grouped_permutation_mse_increase": float(permutation[index]),
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
                "grouped_cv_r2": float(r2_score(y, cv_predictions)),
                "n_documents": len(data),
            }
        )
    return pd.DataFrame(rows)


def compute_syntax_its(frame: pd.DataFrame, syntax_columns: list[str]) -> pd.DataFrame:
    parts = []
    for dataset, group in frame.groupby("dataset"):
        monthly = group.groupby("month_ts", as_index=False)[syntax_columns].mean()
        counts = group.groupby("month_ts", as_index=False).size().rename(columns={"size": "paper_count"})
        monthly = monthly.merge(counts, on="month_ts", how="left")
        monthly = monthly.rename(columns={column: f"{column}_monthly_mean" for column in syntax_columns})
        stats = compute_interrupted_time_series(monthly, syntax_columns)
        if stats.empty:
            continue
        stats.insert(0, "dataset", dataset)
        parts.append(stats)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_alignment_table(coefficients: pd.DataFrame, syntax_its: pd.DataFrame) -> pd.DataFrame:
    if syntax_its.empty:
        return pd.DataFrame()
    its_summary = syntax_its.groupby("feature", as_index=False).agg(
        mean_standardized_its_change=("standardized_slope_change_per_year", "mean"),
        mean_abs_standardized_its_change=("standardized_slope_change_per_year", lambda values: values.abs().mean()),
        its_valid_corpora=("standardized_slope_change_per_year", "count"),
        its_positive_share=("standardized_slope_change_per_year", lambda values: float((values > 0).mean())),
    )
    result = coefficients.merge(its_summary, on="feature", how="inner")
    result["alignment_score"] = (
        result["standardized_coefficient"] * result["mean_standardized_its_change"]
    )
    result["abs_alignment_score"] = result["alignment_score"].abs()
    result["complexity_aligned"] = result["alignment_score"] > 0
    return result.sort_values("abs_alignment_score", ascending=False).reset_index(drop=True)


def display_feature(feature: str) -> str:
    return (
        feature.replace("dep_bigram__", "")
        .replace("dep_role__", "")
        .replace("_per_1k_words", "")
        .replace("_", " ")
    )


def save_coefficient_plot(
    coefficients: pd.DataFrame,
    path: Path,
    *,
    outcome: str,
    model_spec: str,
    top_n: int = 20,
) -> None:
    data = coefficients[(coefficients["outcome"] == outcome) & (coefficients["model_spec"] == model_spec)].copy()
    data = data.sort_values("standardized_coefficient", key=lambda values: values.abs()).tail(top_n)
    if data.empty:
        return
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(4.6, max(2.6, 0.23 * len(data) + 0.8)))
    for index, row in enumerate(data.itertuples(index=False)):
        color = GREEN if row.standardized_coefficient >= 0 else ORCHID
        lower_error = max(0.0, row.standardized_coefficient - row.bootstrap_ci_low)
        upper_error = max(0.0, row.bootstrap_ci_high - row.standardized_coefficient)
        ax.errorbar(
            row.standardized_coefficient,
            index,
            xerr=[[lower_error], [upper_error]],
            fmt=FAMILY_MARKERS[row.family],
            color=color,
            ecolor=color,
            markersize=4,
            capsize=2,
        )
    ax.axvline(0, color=DARK_GREY, linewidth=0.8)
    ax.set_yticks(y, [display_feature(feature) for feature in data["feature"]])
    ax.set_xlabel("Standardized association with readability complexity", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    save_family_legend(path.with_name(f"{path.stem}_legend{path.suffix}"), color_mode="sign")


def save_alignment_plot(
    alignment: pd.DataFrame,
    path: Path,
    *,
    outcome: str,
    model_spec: str,
    label_n: int = 12,
) -> None:
    data = alignment[(alignment["outcome"] == outcome) & (alignment["model_spec"] == model_spec)].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    scale = data["abs_alignment_score"].max() or 1.0
    for family, group in data.groupby("family"):
        ax.scatter(
            group["standardized_coefficient"],
            group["mean_standardized_its_change"],
            s=18 + 80 * group["abs_alignment_score"] / scale,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            edgecolor=DARK_GREY,
            linewidth=0.35,
            alpha=0.8,
        )
    for row in data.nlargest(label_n, "abs_alignment_score").itertuples(index=False):
        ax.annotate(
            display_feature(row.feature),
            (row.standardized_coefficient, row.mean_standardized_its_change),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
        )
    ax.axhline(0, color=DARK_GREY, linewidth=0.7)
    ax.axvline(0, color=DARK_GREY, linewidth=0.7)
    ax.set_xlabel("Association with readability complexity", fontsize=8)
    ax.set_ylabel("Mean standardized ITS slope change/year", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    save_family_legend(path.with_name(f"{path.stem}_legend{path.suffix}"), color_mode="family")


def save_family_legend(path: Path, *, color_mode: str) -> None:
    handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[family], color="none",
            markerfacecolor=(DARK_GREY if color_mode == "sign" else FAMILY_COLORS[family]),
            markeredgecolor=DARK_GREY,
            label=family, markersize=5,
        )
        for family in FAMILY_MARKERS
    ]
    if color_mode == "sign":
        handles.extend(
            [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, label="positive association"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=ORCHID, label="negative association"),
            ]
        )
    fig, ax = plt.subplots(figsize=(5.0, 0.5))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE, pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
