"""Central, color-vision-friendly visual language for every analysis figure."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


# Project identity colors. The green leans toward blue/teal so it remains more
# distinct from orchid under common red-green color-vision deficiencies.
ORCHID = "#A64CA6"
GREEN = "#00876C"
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
SKY_BLUE = "#56B4E9"
BROWN = "#8C6D31"
GREY = "#666666"
DARK_GREY = "#333333"
LIGHT_GREY = "#D9D9D9"
WHITE = "#FFFFFF"

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

# Longer topic palettes deliberately cycle color while marker shape continues to
# vary. This is safer than pretending 20 colors can all remain distinguishable.
TOPIC_COLORS = CATEGORICAL_COLORS
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "8")
HATCHES = ("", "///", "\\\\", "xx", "..", "++", "oo", "--")

INCREASE_COLOR = GREEN
DECREASE_COLOR = ORCHID
INCREASE_HATCH = ""
DECREASE_HATCH = "///"

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
    "arxiv_qbio_abstracts": BLUE,
    "arxiv_stat_abstracts": SKY_BLUE,
    "medarxiv_abstracts": GREY,
    "arxiv_ai_titles": ORCHID,
    "arxiv_qbio_titles": ORANGE,
    "arxiv_stat_titles": VERMILION,
    "medarxiv_titles": BROWN,
}

DATASET_MARKERS = {
    dataset: MARKERS[index % len(MARKERS)]
    for index, dataset in enumerate(DATASET_COLORS)
}

DATASET_HATCHES = {
    dataset: HATCHES[index % len(HATCHES)]
    for index, dataset in enumerate(DATASET_COLORS)
}


def sign_color(value: float) -> str:
    return DECREASE_COLOR if value < 0 else INCREASE_COLOR


def sign_hatch(value: float) -> str:
    return DECREASE_HATCH if value < 0 else INCREASE_HATCH
