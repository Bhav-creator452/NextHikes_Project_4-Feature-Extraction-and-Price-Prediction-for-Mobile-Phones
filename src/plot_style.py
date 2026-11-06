"""Shared chart style so every figure in the project looks consistent."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.data_loader import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Categorical order (fixed, never cycled): blue, orange, aqua, yellow, magenta, green, violet, red.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = PALETTE[0]
ACCENT = PALETTE[1]
NEUTRAL = "#8a8984"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

# Diverging (correlations): blue  <- neutral grey ->  red.
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])
# Sequential (magnitude): one hue, light -> dark.
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#2a78d6", "#0d3a73"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c8c7c1",
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "font.size": 10,
    })


def save_fig(fig, name: str) -> None:
    """Save to reports/figures/<name>.png (used by the README and the final report)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")


def rupee_axis(ax, axis: str = "y") -> None:
    fmt = plt.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
