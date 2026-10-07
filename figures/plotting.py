from math import isclose
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


MM = 1 / 25.4
SINGLE_COLUMN = 89 * MM
DOUBLE_COLUMN = 183 * MM
MAX_HEIGHT = 170 * MM

NATURE = {
  "black": "#000000",
  "orange": "#E69F00",
  "sky_blue": "#56B4E9",
  "bluish_green": "#009E73",
  "yellow": "#F0E442",
  "blue": "#0072B2",
  "vermillion": "#D55E00",
  "reddish_purple": "#CC79A7",
}

CONDITION_STYLES = {
  "no_chunk": dict(color=NATURE["black"], marker="o", ls="-"),
  "matched_chunk": dict(color=NATURE["blue"], marker="s", ls="-"),
  "scrambled_chunk": dict(color=NATURE["orange"], marker="^", ls="--"),
  "wrong_gamma": dict(color=NATURE["reddish_purple"], marker="D", ls=":"),
  "untrained_chunk": dict(color=NATURE["bluish_green"], marker="v", ls="-."),
}

STYLE = Path(__file__).with_name("style") / "nature_motor.mplstyle"
FONT = STYLE.parent / "fonts/EBGaramond-Regular.ttf"
OUTPUT = Path(__file__).with_name("output")

font_manager.fontManager.addfont(str(FONT))


def apply_style() -> None:
  plt.style.use(STYLE)


def panel_label(ax, label: str) -> None:
  ax.text(
    -0.14,
    1.06,
    label,
    transform=ax.transAxes,
    fontsize=8,
    fontweight="bold",
    fontstyle="normal",
    va="top",
    ha="left",
  )


def save_figure(fig, name: str, output: Path = OUTPUT) -> tuple[Path, Path]:
  width, height = fig.get_size_inches()
  if not (isclose(width, SINGLE_COLUMN) or isclose(width, DOUBLE_COLUMN)):
    raise ValueError("Figure width must be 89 mm or 183 mm")
  if height > MAX_HEIGHT:
    raise ValueError("Figure height must not exceed 170 mm")

  output.mkdir(parents=True, exist_ok=True)
  png = output / f"{name}.png"
  fig.savefig(png, dpi=600)
  return png, png
