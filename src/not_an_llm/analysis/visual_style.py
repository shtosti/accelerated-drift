"""Central, color-vision-friendly visual language for every analysis figure."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


# Project identity colors sampled from the previous manuscript figures.
# Green and orchid remain the default binary/sign encoding; blue, orange, and
# other colors are reserved for additional corpora or categories.
ORCHID = "#963E8D"
GREEN = "#54A066"
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
SKY_BLUE = "#56B4E9"
BROWN = "#8C6D31"
GREY = "#666666"
DARK_GREY = "#333333"
LIGHT_GREY = "#D9D9D9"
WHITE = "#FFFFFF"

# Hatch stroke thickness is measured in points. Lower values keep patterns
# legible in compact manuscript figures. Pattern density is controlled by
# repeating the hatch character (for example, "...." is denser than "..").
HATCH_LINEWIDTH = 0.2
mpl.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH

# Eight-category palette used by dependency panels and small categorical plots.
# Color must always be paired with a label, marker, hatch, or line style.
CATEGORICAL_COLORS = (
    BLUE,
    ORANGE,
    GREEN,
    ORCHID,
    VERMILION,
    SKY_BLUE,
    BROWN,
    GREY,
)

# Stable colors used by the manuscript's dependency-role trend plots. Keep this
# mapping explicit: cycling the general categorical palette changes a role's
# identity when roles are filtered or reordered.
DEPENDENCY_ROLE_COLORS = {
    "det": "#4E79A7",
    "pobj": "#F28E2B",
    "prep": "#E15759",
    "aux": "#76B7B2",
    "auxpass": "#59A14F",
    "advcl": "#EDC948",
    "conj": "#B07AA1",
    "compound": "#FF9DA7",
    "dobj": "#1F77B4",
    "amod": "#8C564B",
    "nummod": "#E377C2",
    "appos": "#7F7F7F",
}

# Longer topic palettes deliberately cycle color while marker shape continues to
# vary. This is safer than pretending 20 colors can all remain distinguishable.
TOPIC_COLORS = CATEGORICAL_COLORS
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "8")
HATCHES = ("", "....", "//////", "xxxx", "++++", "----", "\\\\\\\\", "****")

INCREASE_COLOR = GREEN
DECREASE_COLOR = ORCHID
INCREASE_HATCH = ""
DECREASE_HATCH = ".."

ORCHID_GREEN_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "accessible_orchid_white_green",
    [ORCHID, WHITE, GREEN],
)

CORPUS_LINESTYLES = {
    "arxiv_ai": "-",
    "arxiv_qbio": "--",
    "arxiv_stat": ":",
    "medarxiv": "-.",
}

DATASET_COLORS = {
    "arxiv_ai_abstracts": GREEN,
    "arxiv_qbio_abstracts": ORCHID,
    "arxiv_stat_abstracts": ORANGE,
    "medarxiv_abstracts": BLUE,
    "arxiv_ai_titles": GREEN,
    "arxiv_qbio_titles": ORCHID,
    "arxiv_stat_titles": ORANGE,
    "medarxiv_titles": BLUE,
}

DATASET_MARKERS = {
    dataset: MARKERS[index % len(MARKERS)]
    for index, dataset in enumerate(DATASET_COLORS)
}

DATASET_HATCHES = {
    "arxiv_ai_abstracts": "",
    "arxiv_qbio_abstracts": "..",
    "arxiv_stat_abstracts": "///",
    "medarxiv_abstracts": "xx",
    "arxiv_ai_titles": "++",
    "arxiv_qbio_titles": "**",
    "arxiv_stat_titles": "--",
    "medarxiv_titles": "\\\\",
}


def sign_color(value: float) -> str:
    return DECREASE_COLOR if value < 0 else INCREASE_COLOR


def sign_hatch(value: float) -> str:
    return DECREASE_HATCH if value < 0 else INCREASE_HATCH
