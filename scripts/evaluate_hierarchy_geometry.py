"""Resumable, frozen, matched base/DDQN/A-E-label-shuffle evaluation. No training."""

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("scripts/evaluate_hierarchy_geometry.py", {})
DEFAULTS = SETTINGS["defaults"].get("scripts/evaluate_hierarchy_geometry.py", {})
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from rich.progress import Progress
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
import torch

from analysis import mechanism_experiment as e

DEFAULT_OUT = ROOT / "data/transfer/evaluation_seeds"


def halfway_crossing(path, start, target, following):
    """First forward intersection with the geometric halfway perpendicular."""
    direction = target - start
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        return None, 0
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    side = normal @ (following - target)
    if abs(side) < 1e-6:
        return None, 0  # No meaningful inside/outside sign for collinear targets.
    normal *= np.sign(side)
    projected = (path - start) @ unit - length / 2
    indices = np.flatnonzero((projected[:-1] <= 0) & (projected[1:] > 0))
    if not len(indices):
        return None, 0
    index = indices[0]
    fraction = -projected[index] / (projected[index + 1] - projected[index])
    point = path[index] + fraction * (path[index + 1] - path[index])
    return float((point - start) @ normal * 1000), len(indices)


def geometry(xy, visit, targets, speed_floor=DEFAULTS["geometry"]["speed_floor"]):
    """Paper-inspired hC1 and spatially target-restricted per-trial peak curvatures."""
    hits = [np.flatnonzero(visit >= k) for k in (1, 2, 3)]
    if any(not len(hit) for hit in hits):
        return {"geometry_available": False, "reason": "First three targets not acquired"}
    first, second, third = [int(hit[0]) for hit in hits]
    hc, crossings = halfway_crossing(xy[:first + 1], xy[0], targets[0], targets[1])
    smooth = gaussian_filter1d(xy, sigma=2, axis=0)
    velocity = np.gradient(smooth, 0.01, axis=0)
    acceleration = np.gradient(velocity, 0.01, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    numerator = np.abs(velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0])
    curvature = np.full(len(xy), np.nan)
    np.divide(numerator, speed**3, out=curvature, where=speed >= speed_floor)
    peaks, valid_counts, slow_counts = [], [], []
    for target, start, end in [(targets[0], 0, second), (targets[1], first, third)]:
        inside = np.linalg.norm(xy[start:end + 1] - target, axis=1) <= 0.02
        values = curvature[start:end + 1][inside]
        valid = values[np.isfinite(values)]
        peaks.append(float(valid.max()) if len(valid) else None)
        valid_counts.append(len(valid))
        slow_counts.append(int(np.sum(~np.isfinite(values))))
    return {"geometry_available": hc is not None and all(p is not None for p in peaks),
            "reason": None if hc is not None and all(p is not None for p in peaks)
                      else "Missing signed halfway crossing or reliable within-target curvature",
            "hc1_mm": hc, "halfway_forward_crossings": crossings,
            "kappa1_m_inv": peaks[0], "kappa2_m_inv": peaks[1],
            "valid_curvature_samples": valid_counts, "excluded_slow_samples": slow_counts,
            "speed_floor_mps": speed_floor}


def self_check():
    path = np.array([[0., 0.], [.25, .10], [.75, .30], [1., 0.]])
    hc, count = halfway_crossing(path, np.zeros(2), np.array([1., 0.]), np.array([1., 1.]))
    assert np.isclose(hc, 200) and count == 1  # Geometric interpolation, not time midpoint.
    hc_reverse, _ = halfway_crossing(path, np.zeros(2), np.array([1., 0.]), np.array([1., -1.]))
    assert np.isclose(hc_reverse, -200)
    assert halfway_crossing(path[:2], np.zeros(2), np.array([1., 0.]), np.array([1., 1.])) == (None, 0)
    assert not geometry(np.zeros((10, 2)), np.zeros(10), np.zeros((3, 2)))["geometry_available"]
    theta = np.linspace(0, np.pi, 201)
    xy = np.column_stack([.1 * np.cos(theta), .1 * np.sin(theta)])
    visits = np.zeros(201, dtype=int)
    visits[50:] = 1
    visits[100:] = 2
    visits[150:] = 3
    metric = geometry(xy, visits, xy[[50, 100, 150]])
    assert metric["geometry_available"]
    assert np.isclose(metric["kappa1_m_inv"], 10, rtol=.01)
    assert np.isclose(metric["kappa2_m_inv"], 10, rtol=.01)


def contexts(panel, order, limit):
    rows = []
    for seed in e.math.SEEDS:
        for posture in range(2):
            rng = np.random.default_rng(np.random.SeedSequence([seed, posture, 270019]))
            origin = panel["postures_deg"][order * 2 + posture]
            rows.append({"context_seed": seed, "posture": posture,
                         "q": (np.asarray(origin) + rng.uniform(-.5, .5, 2)).tolist()})
    return rows[:limit] if limit is not None else rows


def evaluate(args):
    torch.set_num_threads(1)
    root = args.output
    panel_path = ROOT / "data/mechanism/panel.json"
    panel = e.read(panel_path)
    sequences = [s for group in panel["groups"].values() for s in group]
    assert all(tuple(s) != e.SOURCE_SEQUENCE for s in sequences)
    library = e.read(e.LIBRARY)["options"]
    assert [row["id"] for row in library] == [f"off_target_{sid}" for sid in "ABCDE"]
    assert [row["choice"] for row in library] == list(range(1, 6))
    count = len(sequences) if args.orders is None else min(args.orders, len(sequences))
    conditions = ["base", "ddqn"] + [f"shuffle_{i}" for i in range(1, args.shuffles + 1)]
    protocol = {"version": 1, "source_hashes": {str(p): e.sha(p) for p in
                [e.BASE, e.REACHER, e.LIBRARY, e.SELECTOR, panel_path, Path(__file__),
                 ROOT / "analysis/mechanism_experiment.py"]},
                "steps": args.steps, "orders": count, "contexts": args.contexts,
                "conditions": conditions, "physical_dt": args.physical_dt,
                "smoothing_sigma_samples": 2, "curvature_speed_floors_mps": [.01, .02, .05],
                "context_seeds_are_not_policy_training_seeds": True,
                "smoke": count < len(sequences) or args.contexts is not None or args.steps < 800,
                "shuffle": "Cyclic permutation of frozen option Q labels, leaving base Q and executor unchanged.",
                "interpretation": "Matched behavioral diagnostic, not a replication of the human paper or proof of Laplace-specific causation."}
    if (root / "protocol.json").exists() and e.read(root / "protocol.json") != protocol:
        raise ValueError("Protocol changed; use a new output directory")
    e.write(root / "protocol.json", protocol)
    policies, selector = e.models(), e.verified_selector()
    started = time.perf_counter()
    with Progress() as progress:
        overall = progress.add_task("Matched sequence/condition batches", total=count * len(conditions))
        current = progress.add_task("Environment steps", total=args.steps)
        for order in range(count):
            batch = contexts(panel, order, args.contexts)
            for condition in conditions:
                path = root / "rollouts" / f"order_{order:02d}_{condition}.npz"
                result_path = path.with_suffix(".json")
                if path.exists() and result_path.exists():
                    progress.advance(overall)
                    continue
                progress.reset(current, total=args.steps, description=f"Order {order:02d} {condition}")
                task, obs = e.make_task(sequences[order], args.physical_dt, [r["q"] for r in batch], args.steps)
                targets = e.array(task.targets)[sequences[order]]
                positions = [e.features(task)[:, :2]]
                visits = [e.array(task.sequence_index)]
                finished = [e.array(task.finished)]
                choices = []

                def observe(step, task, observation, executed):
                    positions.append(e.features(task)[:, :2])
                    visits.append(e.array(task.sequence_index))
                    finished.append(e.array(task.finished))
                    choices.append(executed.copy())
                    progress.update(current, completed=step)

                rotation = int(condition.split("_")[1]) if condition.startswith("shuffle") else 0

                def select(observation):
                    values = selector(observation)
                    indices = [0] + list(np.roll(np.arange(1, 6), rotation))
                    return values[:, indices]

                result = e.transfer_batch(task, obs, policies, (15., 1.5),
                          "base" if condition == "base" else "learned", library,
                          select, args.steps, observer=observe)
                e.save_arrays(path, xy=np.asarray(positions), visit=np.asarray(visits),
                              finished=np.asarray(finished), executed_option=np.asarray(choices), targets=targets)
                e.write(result_path, {"order": order, "condition": condition,
                                     "contexts": batch, "outcomes": result})
                progress.advance(overall)
    print(f"Evaluation completed in {time.perf_counter() - started:.1f}s")
    analyze(root)


def analyze(root):
    protocol = e.read(root / "protocol.json")
    rows = []
    for path in sorted((root / "rollouts").glob("*.npz")):
        if not path.with_suffix(".json").exists():
            continue
        metadata = e.read(path.with_suffix(".json"))
        with np.load(path) as data:
            for index, (context, outcome) in enumerate(zip(metadata["contexts"], metadata["outcomes"])):
                # Stop analysis at that row's first completion, not at batch completion.
                done = np.flatnonzero(data["finished"][:, index])
                end = int(done[0]) + 1 if len(done) else len(data["xy"])
                xy, visit = data["xy"][:end, index], data["visit"][:end, index]
                metrics = {str(floor): geometry(xy, visit, data["targets"], floor)
                           for floor in (.01, .02, .05)}
                rows.append({"order": metadata["order"], "condition": metadata["condition"],
                             **context, **outcome, **metrics["0.02"], "speed_floor_sensitivity": metrics})
    if not rows:
        raise ValueError("No finished rollout files")
    e.write(root / "trial_metrics.json", rows)
    conditions = protocol["conditions"]
    summary = {"smoke": protocol["smoke"], "physical_dt": protocol["physical_dt"],
               "conditions": {}, "paired_success": {},
               "interpretation": "Descriptive diagnostics only. Architecture is known; behavioral signature and transfer utility are separate claims."}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        available = [r for r in subset if r["geometry_available"]]
        summary["conditions"][condition] = {"trials": len(subset), "successful": sum(r["success"] for r in subset),
                                              "geometry_available": len(available), "within_order_correlations": [],
                                              "median_trial_peaks_by_speed_floor": {}}
        for floor in ("0.01", "0.02", "0.05"):
            metrics = [r["speed_floor_sensitivity"][floor] for r in subset
                       if r["speed_floor_sensitivity"][floor]["geometry_available"]]
            summary["conditions"][condition]["median_trial_peaks_by_speed_floor"][floor] = {
                "n": len(metrics), "kappa1_m_inv": float(np.median([m["kappa1_m_inv"] for m in metrics])) if metrics else None,
                "kappa2_m_inv": float(np.median([m["kappa2_m_inv"] for m in metrics])) if metrics else None}
        for order in sorted({r["order"] for r in available}):
            batch = [r for r in available if r["order"] == order]
            if len(batch) >= 3:
                x, y = [r["hc1_mm"] for r in batch], [r["kappa2_m_inv"] for r in batch]
                k1 = [r["kappa1_m_inv"] for r in batch]
                rho_hc = float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else None
                rho_k = float(spearmanr(k1, y).statistic) if np.std(k1) > 0 and np.std(y) > 0 else None
                summary["conditions"][condition]["within_order_correlations"].append(
                    {"order": order, "n": len(batch), "hc1_vs_k2_rho": rho_hc, "k1_vs_k2_rho": rho_k})
    key = lambda r: (r["order"], r["context_seed"], r["posture"])
    ddqn = {key(r): r for r in rows if r["condition"] == "ddqn"}
    for condition in conditions:
        if condition == "ddqn":
            continue
        control = {key(r): r for r in rows if r["condition"] == condition}
        pairs = [(ddqn[k], control[k]) for k in ddqn.keys() & control.keys()]
        if pairs:
            summary["paired_success"][condition] = {"n": len(pairs),
                "ddqn_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
                "control_only_success": sum(b["success"] and not a["success"] for a, b in pairs),
                "both_geometry_available": sum(a["geometry_available"] and b["geometry_available"] for a, b in pairs)}
    e.write(root / "summary.json", summary)
    font_manager.fontManager.addfont(str(ROOT / "figures/style/fonts/EBGaramond-Regular.ttf"))
    plt.style.use(str(ROOT / "figures/style/nature_motor.mplstyle"))
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 75 / 25.4), layout="constrained")
    colors = {"base": "#D55E00", "ddqn": "#009E73"}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition and r["geometry_available"]]
        color = colors.get(condition, "#7F7F7F")
        for ax, xkey in zip(axes, ("hc1_mm", "kappa1_m_inv")):
            ax.scatter([r[xkey] for r in subset], [r["kappa2_m_inv"] for r in subset],
                       s=9, alpha=.55, color=color, label=f"{condition} (n={len(subset)})")
    axes[0].set_xlabel("First-leg halfway crossing (mm)")
    axes[1].set_xlabel(r"First-target peak curvature ($\mathrm{m^{-1}}$)")
    for ax in axes:
        ax.set_ylabel(r"Second-target peak curvature ($\mathrm{m^{-1}}$)")
        ax.legend(frameon=False, fontsize=5)
    fig.suptitle("SMOKE CHECK — no scientific conclusion" if protocol["smoke"] else
                 "Matched hierarchy geometry diagnostic (pooled sequences)", fontsize=8)
    fig.savefig(root / "joint_geometry.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def smoke(args):
    self_check()
    torch.set_num_threads(1)
    policies, selector = e.models(), e.verified_selector()
    library = e.read(e.LIBRARY)["options"]
    results = []
    for record in (False, True):
        task, obs = e.make_task(e.SOURCE_SEQUENCE, .01, max_steps=30)
        captured = []
        callback = lambda step, task, obs, choice: captured.append((step, e.features(task), choice.copy()))
        result = e.transfer_batch(task, obs, policies, (15., 1.5), "learned", library,
                                  selector, steps=30, observer=callback if record else None)
        results.append((result, e.features(task)))
        if record:
            assert len(captured) == 30 and captured[-1][0] == 30
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])
    e.write(args.output / "checks.json", {"geometric_crossing": True, "circle_curvature": True,
             "failed_trial_retained": True, "observer_preserves_frozen_executor": True,
             "selector_artifact_hashes_verified": True})
    print("Geometry and frozen-executor observation checks passed. No training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "run", "analyze"], default="check")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--orders", type=int)
    parser.add_argument("--contexts", type=int)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--shuffles", type=int, default=4)
    parser.add_argument("--physical-dt", type=float, choices=[.01, .001], default=.01,
                        help="0.01 reproduces historical MotorNet; 0.001 uses existing RK4 for convergence audit")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.shuffles <= 4 or (args.orders is not None and args.orders < 1) or (args.contexts is not None and not 1 <= args.contexts <= 10):
        parser.error("Positive steps/orders; shuffles 1–4; contexts 1–10")
    {"check": lambda: smoke(args), "run": lambda: evaluate(args),
     "analyze": lambda: analyze(args.output)}[args.stage]()
