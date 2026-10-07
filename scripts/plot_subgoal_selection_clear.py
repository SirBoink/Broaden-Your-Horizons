"""Display frozen A-E selection using original scores and discovery references.

No smoothing or rescaling. Export 600 dpi PNGs and their exact plotted samples;
retain the frozen catalog and unsmoothed recorded scores.
"""
from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.plotting import DOUBLE_COLUMN, MM, NATURE, apply_style, save_figure
import formulas as math

COLORS = [NATURE[key] for key in
          ("vermillion", "blue", "reddish_purple", "sky_blue", "orange")]
OUTPUTS = [ROOT / "figures/analysis", ROOT / "figures/output/correct_model"]


def save_plot(fig, name):
    png, _ = save_figure(fig, name, OUTPUTS[0])
    OUTPUTS[1].mkdir(parents=True, exist_ok=True)
    copyfile(png, OUTPUTS[1] / png.name)


def load_selection():
    audit = json.loads((ROOT / "data/mechanism/audit.json").read_text())
    with np.load(ROOT / "data/mechanism/references.npz") as data:
        refs = [data[f"band_{i}"].copy() for i in range(39)]
    with np.load(ROOT / "data/discovery/normalization/temporal_band_pilot.npz") as data:
        mean, std = data["feature_mean_raw"], data["feature_std"]

    sources, selected = {}, []
    for candidate, option in zip(audit["candidates"], audit["original_library"]):
        trial = candidate["trial"]
        if trial not in sources:
            path = ROOT / f"data/mechanism/integration/source_{trial}.npz"
            with np.load(path) as data:
                phi, hit = data["phi"], int(data["first_hit"])
            raw, _ = math.scores(phi, mean, std, np.asarray(audit["gamma_grid"]))
            sources[trial] = (raw[:, :hit], math.percentiles(raw[:, :hit], refs), hit)
        raw, percentile, hit = sources[trial]
        band, column = candidate["peak_band"], candidate["step"] - 1
        components, _ = label(percentile >= .90, structure=np.ones((3, 3), dtype=int))
        component = components == components[band, column]
        cells = np.argwhere(component)
        bands = np.unique(cells[:, 0])

        # Check plotted decisions against the frozen catalog, not their appearance.
        np.testing.assert_allclose(percentile[band, column], candidate["peak_percentile"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(raw[band, column], candidate["raw_change"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(bands, candidate["bands"])
        assert components[band, column] > 0 and candidate["nearest_target_m"] > .025
        assert any(set(range(int(b), int(b) + 3)) <= set(bands) for b in bands)
        assert tuple(max(cells, key=lambda v: (percentile[tuple(v)], -v[1], -v[0]))) == (band, column)

        selected.append(dict(candidate=candidate, raw=raw, percentile=percentile,
                             component=component, hit=hit,
                             time_ms=np.arange(1, hit + 1) * 10,
                             tau_ms=-10 / np.log(option["gamma"])))
    return selected


def draw_trace(ax, item, color):
    candidate = item["candidate"]
    band, column = candidate["peak_band"], candidate["step"] - 1
    time = item["time_ms"]
    values = item["percentile"][band] * 100
    ax.axhspan(90, 100, color="0.95", zorder=0)
    ax.axhline(90, color="0.45", ls="--", lw=.7, zorder=1)
    ax.plot(time, values, color=color, lw=1.3, marker="o", ms=2,
            markeredgewidth=0, zorder=3)
    active = item["component"][band]
    ax.fill_between(time, 90, values, where=active, color=color,
                    alpha=.18, interpolate=False, zorder=2)
    ax.axvline(candidate["step"] * 10, color=color, lw=.7, ls=":", alpha=.7)
    ax.scatter(time[column], values[column], marker="D", s=24,
               facecolor=color, edgecolor="black", lw=.5, zorder=5)
    ax.set(xlim=(0, item["hit"] * 10 + 5), ylim=(0, 105),
           yticks=[0, 50, 90, 100], xlabel="Time after target activation (ms)",
           ylabel="Within-band percentile (%)")
    ax.tick_params(labelsize=5.5)


def plot_peaks(selected):
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN, 125 * MM), layout="constrained")
    for i, (item, color) in enumerate(zip(selected, COLORS)):
        ax = axes.flat[i]
        candidate = item["candidate"]
        draw_trace(ax, item, color)
        ax.set_title(f"{candidate['id']}  |  T{candidate['target']}  |  "
                     f"{candidate['step'] * 10} ms ({candidate['peak_percentile'] * 100:.1f}%)\n"
                     f"{len(candidate['bands'])} bands; nearest target {candidate['nearest_target_m'] * 1000:.1f} mm\n"
                     f"Horizon {item['tau_ms']:.1f} ms; raw change {candidate['raw_change']:.3g}",
                     fontsize=6.2, loc="left", linespacing=1.5)

    info = axes.flat[5]
    info.axis("off")
    info.set_title("Why these five states?", loc="left", fontsize=7, fontweight="bold")
    info.text(0, .93,
              "1. Within-band percentile >=90%.\n\n"
              "2. Connected support across >=3\n    adjacent discount bands.\n\n"
              "3. Choose the region's maximum.\n\n"
              "4. Merge matching discoveries.\n\n"
              "5. Keep >25 mm from all targets.",
              fontsize=6.2, va="top", linespacing=1.2)
    info.legend(handles=[
        Line2D([], [], color="0.25", marker="o", ms=2, lw=1, label="Original 10 ms samples"),
        Line2D([], [], color="black", marker="D", ms=4, lw=0, label="Selected representative"),
        Patch(fc="0.8", alpha=.4, label="Selected region in this band")],
        loc="lower left", bbox_to_anchor=(-.04, -.03), fontsize=5.6)
    fig.suptitle("Original A-E selection: percentile, multiscale support, off-target position",
                 fontsize=8, fontweight="bold")
    save_plot(fig, "subgoal_selection_peaks_clear")
    plt.close(fig)


def plot_components(selected):
    fig, axes = plt.subplots(5, 2, figsize=(DOUBLE_COLUMN, 164 * MM),
                             gridspec_kw={"width_ratios": [1, 1.25]}, layout="constrained")
    for row, (item, color) in enumerate(zip(selected, COLORS)):
        candidate = item["candidate"]
        left, right = axes[row]
        draw_trace(left, item, color)
        left.set_title(f"{candidate['id']}: {candidate['step'] * 10} ms "
                       f"({candidate['peak_percentile'] * 100:.1f}%); "
                       f"{candidate['nearest_target_m'] * 1000:.1f} mm from nearest target",
                       loc="left", fontsize=6.5, fontweight="bold")
        time_edges = np.arange(item["hit"] + 1) * 10 + 5
        band_edges = np.arange(40) - .5
        im = right.pcolormesh(time_edges, band_edges, item["percentile"] * 100,
                              cmap="Greys", vmin=0, vmax=100, shading="flat", rasterized=True)
        overlay = np.ma.masked_where(~item["component"], np.ones_like(item["percentile"]))
        right.pcolormesh(time_edges, band_edges, overlay,
                        cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=.65,
                        shading="flat", rasterized=True)
        right.scatter(candidate["step"] * 10, candidate["peak_band"], marker="D",
                       s=24, facecolor=color, edgecolor="black", lw=.6, zorder=4)
        right.set(xlim=(0, item["hit"] * 10 + 5), ylim=(-.5, 38.5),
                  yticks=[0, 10, 20, 30, 38], ylabel="Discount-band index",
                  xlabel="Time after target activation (ms)")
        right.set_title(f"Selected component: {len(candidate['bands'])} bands; "
                        f"representative band {candidate['peak_band']}",
                        loc="left", fontsize=6.5)
        right.tick_params(labelsize=5.5)
        if row != 4:
            left.set_xlabel("")
            right.set_xlabel("")
    colorbar = fig.colorbar(im, ax=axes[:, 1], fraction=.025, pad=.02)
    colorbar.set_label("Original within-band percentile (%)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5)
    fig.suptitle("Selection evidence: colored region is the original connected >=90% component",
                 fontsize=7.5, fontweight="bold")
    save_plot(fig, "subgoal_selection_components_clear")
    plt.close(fig)


def main():
    apply_style()
    selected = load_selection()
    plot_peaks(selected)
    plot_components(selected)
    summary = dict(reference="data/mechanism/references.npz",
                   processing="Original 10 ms samples; no smoothing, rescaling or clipping",
                   display="Full target leg before first qualifying completion",
                   selection_rule="90th percentile; >=3 adjacent bands in connected region; "
                                  "component representative; discovery merge; >25 mm from all targets",
                   candidates=[dict(**item["candidate"],
                                    discount_horizon_ms=item["tau_ms"],
                                    target_completion_ms=item["hit"] * 10,
                                    displayed_samples=item["hit"]) for item in selected])
    arrays = {}
    for item in selected:
        prefix = item["candidate"]["id"]
        for key in ("raw", "percentile", "component", "time_ms"):
            arrays[f"{prefix}_{key}"] = item[key]
    for output in OUTPUTS:
        (output / "subgoal_selection_clear.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(output / "subgoal_selection_clear.npz", **arrays)
    print("Verified all five original peak values, component bands, representatives and exclusion distances.")
    print("Saved two 600 dpi figures and their exact plotted data; original PNG retained.")


if __name__ == "__main__":
    main()
