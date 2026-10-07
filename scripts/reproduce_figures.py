"""Regenerate scientific figures from the saved results, without training."""
import subprocess
import sys
from pathlib import Path

from rich.progress import track

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    modules = ("plot_handoff_traces", "plot_methods", "plot_discovery_and_dynamics", "plot_results", "plot_paper_subgoal_curvature")
    for module in track(modules, description="Reproducing figures"):
        subprocess.run([sys.executable, "-m", f"scripts.{module}"], cwd=ROOT, check=True)
