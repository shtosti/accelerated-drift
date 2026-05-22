python - <<'PY' > top20_its_table_rows.tex
import pandas as pd
import math

files = {
    "arXiv": "data/analysis/arxiv_its_stats.csv",
    "medRxiv": "data/analysis/medarxiv_its_stats.csv",
}

def fmt_q(q):
    if q < 0.001:
        return r"\textbf{< .001***}"
    if q < 0.01:
        return rf"\textbf{{{q:.3f}**}}".replace("0.", ".")
    if q < 0.05:
        return rf"\textbf{{{q:.3f}*}}".replace("0.", ".")
    return f"{q:.3f}".replace("0.", ".")

def fmt_feature(x):
    return x.replace("_", r"\_")

rows = []
for dataset, path in files.items():
    df = pd.read_csv(path)
    df["Dataset"] = dataset
    rows.append(df)

out = pd.concat(rows, ignore_index=True)
out = out[out["slope_change_q"] < 0.05].copy()
out["abs_delta_beta_sd"] = out["standardized_slope_change_per_year"].abs()
out = out.sort_values("abs_delta_beta_sd", ascending=False).head(20)

for _, r in out.iterrows():
    print(
        f"{r['Dataset']} & "
        f"{fmt_feature(r['family'])} & "
        f"{fmt_feature(r['feature'])} & "
        f"{r['pre_mean']:.3f} $\\rightarrow$ {r['post_mean']:.3f} & "
        f"\\textbf{{{r['standardized_slope_change_per_year']:.2f}}} & "
        f"[{r['standardized_slope_change_per_year_ci_low']:.2f}, "
        f"{r['standardized_slope_change_per_year_ci_high']:.2f}] & "
        f"{fmt_q(r['slope_change_q'])} \\\\"
    )
PY