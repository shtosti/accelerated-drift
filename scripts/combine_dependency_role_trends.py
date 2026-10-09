from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.trends import DEPENDENCY_ROLE_ORDER, dependency_role_color_map


PANELS = (
    ("arxiv_ai_abstracts", "AI"),
    ("arxiv_qbio_abstracts", "Quantitative biology"),
    ("arxiv_stat_abstracts", "Statistics"),
    ("medarxiv_abstracts", "medRxiv"),
)


def _trim_white(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    bbox = ImageChops.difference(rgb, background).getbbox()
    return rgb.crop(bbox) if bbox else rgb


def _remove_event_label(image: Image.Image) -> Image.Image:
    """Remove the legacy label above the axes while retaining the event line."""
    cleaned = image.convert("RGB").copy()
    width, height = cleaned.size
    draw = ImageDraw.Draw(cleaned)
    draw.rectangle((width * 0.54, 0, width, height * 0.052), fill="white")
    return cleaned


def main() -> None:
    visuals = ROOT / "data" / "visuals"
    output_dir = visuals / "corpus_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(5.65, 5.85))
    for ax, (dataset, label) in zip(axes.flat, PANELS, strict=True):
        source = visuals / dataset / "dependency_distribution_trends.png"
        if not source.exists():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            ax.imshow(_trim_white(_remove_event_label(image)))
        ax.set_title(label, color="black", fontsize=10, pad=2)
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.985, wspace=-0.12, hspace=0.12)

    path = output_dir / "dependency_role_trends_all_corpora.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.025)
    plt.close(fig)

    legend_target = output_dir / "dependency_role_trends_all_corpora_legend.png"
    roles = DEPENDENCY_ROLE_ORDER
    colors = dependency_role_color_map(roles)
    handles = [
        Line2D(
            [0], [0], color=colors[role], marker="o",
            linewidth=1.5, markersize=3.5, label=role,
        )
        for index, role in enumerate(roles)
    ]
    legend_fig = plt.figure(figsize=(2.15, 1.55))
    legend_fig.legend(
        handles=handles, loc="center", ncol=2, frameon=False, fontsize=7.5,
        handlelength=1.45, handletextpad=0.45, columnspacing=1.0, labelspacing=0.38,
    )
    legend_fig.savefig(
        legend_target, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.015,
    )
    plt.close(legend_fig)

    print(path)
    print(legend_target)


if __name__ == "__main__":
    main()
