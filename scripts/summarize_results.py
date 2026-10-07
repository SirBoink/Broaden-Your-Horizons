"""Recompute presentation outcomes from complete saved trial panels."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from rich.progress import track

from config import PARAMETERS

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_interval(values):
    cfg = PARAMETERS["statistics"]
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    sample = rng.choice(values, (cfg["bootstrap_samples"], len(values))).mean(axis=1)
    tail = (1 - cfg["confidence_level"]) / 2
    return np.quantile(sample, [tail, 1 - tail]).tolist()


def summarize():
    rows = []
    paths = sorted((ROOT / "data/mechanism/historical_transfer").glob("*.json"))
    paths += sorted((ROOT / "data/single_discount/single_transfer").glob("*.json"))
    for path in track(paths, description="Reading complete transfer panels"):
        rows.extend(row for row in read(path)["rows"] if row["controller"] == "base")
    if len(rows) != 720:
        raise ValueError(f"Expected 720 historical rows; found {len(rows)}")
    result = {}
    for group, orders in (("full", range(24)), ("short_transitions", range(16, 24))):
        panel = {}
        rates = {}
        for method in ("base", "single", "learned"):
            selected = [r for r in rows if r["method"] == method and r["order"] in orders]
            if len(selected) != len(orders) * 10:
                raise ValueError(f"Incomplete {group}/{method} panel")
            rates[method] = [100 * np.mean([r["success"] for r in selected if r["order"] == o]) for o in orders]
            panel[method] = {"trials": len(selected), "successes": sum(r["success"] for r in selected),
                             "completion_pct": 100 * np.mean([r["success"] for r in selected]),
                             "restricted_completion_s": np.mean([r["restricted_completion_s"] for r in selected])}
        for method in ("base", "single"):
            differences = np.asarray(rates["learned"]) - rates[method]
            panel[f"learned_minus_{method}"] = {"gain_pp": float(differences.mean()),
                                               "order_bootstrap_95CI_pp": paired_interval(differences)}
        result[group] = panel
    policies = defaultdict(lambda: {"base": [], "ddqn": []})
    for path in sorted((ROOT / "data/handoffs/rollouts").glob("*.json")):
        job = read(path)
        for controller in ("base", "ddqn"):
            policies[job["policy_seed"]][controller].extend(job[controller])
    result["policy_replication"] = {
        str(seed): {controller: {"trials": len(rows), "successes": sum(r["success"] for r in rows)}
                    for controller, rows in controllers.items()}
        for seed, controllers in sorted(policies.items())}
    replay = read(ROOT / "data/controls/replay/replay_comparison.json")
    tolerance = PARAMETERS["statistics"]["local_progress_tie_tolerance_m"]
    result["local_replay"] = {}
    for dt in ("0.01", "0.001", "0.0005"):
        values = np.array([r["chosen_minus_base_progress_500ms_m"][dt] for r in replay])
        result["local_replay"][dt] = {"cases": len(values), "better": int(sum(values > tolerance)),
                                     "tied": int(sum(abs(values) <= tolerance)), "worse": int(sum(values < -tolerance))}
    result["demonstrations"] = {}
    for label in ("A", "B"):
        with (ROOT / f"data/transfer/demonstrations/{label}/episodes.csv").open() as stream:
            episodes = list(csv.DictReader(stream))
        options = sum(int(r["opt_bouts"]) for r in episodes)
        base = sum(int(r["prim_bouts"]) for r in episodes)
        commands = sum(int(r["opt_steps"]) for r in episodes)
        option_commands = sum(int(r["opt_steps"]) - int(r["opt_decisions"]) + int(r["opt_bouts"]) for r in episodes)
        result["demonstrations"][label] = {"episodes": len(episodes), "option_segments": options,
            "base_segments": base, "option_segment_share": options / (options + base),
            "option_commands": option_commands, "all_commands": commands, "option_command_share": option_commands / commands}
    return result


if __name__ == "__main__":
    output = ROOT / "results/summary.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summarize(), indent=2) + "\n", encoding="utf-8")
    print(output)
