"""Saved-data audit and small, outcome-blind numerical screen. No training."""

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("scripts/screen_landmark_evidence.py", {})
DEFAULTS = SETTINGS["defaults"].get("scripts/screen_landmark_evidence.py", {})
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def episode_key(row):
    return tuple(row[k] for k in ('policy_seed', 'order', 'context_seed', 'posture'))


def clustered_effect(groups):
    """Equal episode weight; bootstrap whole orders within one policy."""
    orders = defaultdict(list)
    for key, values in groups.items():
        orders[key[1]].append(float(np.mean(values)))
    totals = np.array([sum(v) for v in orders.values()])
    counts = np.array([len(v) for v in orders.values()])
    draws = np.random.default_rng(42).integers(len(orders), size=(10000, len(orders)))
    boot = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(mean=float(totals.sum()/counts.sum()),
                exploratory_order_bootstrap_95CI=np.quantile(boot, [.025, .975]).tolist(),
                episodes=int(counts.sum()), orders=len(orders))


def audit(source, output):
    contrasts = read(source/'branch_contrasts.json')
    endpoints = read(source/'actual_endpoints.json')
    assert len(endpoints) == len(list((source/'branches').glob('*.json')))
    results = []
    for seed in (7, 101, 202):
        for comparator in ('base', 'mirror', 'other_mean'):
            for metric in ('success', 'restricted_completion_s', 'progress_500ms_m', 'targets_completed'):
                rows = [r for r in contrasts if (r['policy_seed'], r['comparator'], r['metric']) == (seed, comparator, metric)]
                groups = defaultdict(list)
                for r in rows:
                    groups[episode_key(r)].append(r['benefit'])
                assert groups, (seed, comparator, metric)
                results.append(dict(policy_seed=seed, comparator=comparator, metric=metric,
                                    invocations=len(rows), **clustered_effect(groups)))
    distributions = []
    for seed in (7, 101, 202):
        for choice in range(1, 6):
            rows = [r for r in endpoints if r['policy_seed'] == seed and r['choice'] == choice]
            reasons = Counter(r['actual_endpoint']['reason'] if r['actual_endpoint'] else 'missing' for r in rows)
            for reason, count in reasons.items():
                selected = [r for r in rows if (r['actual_endpoint']['reason'] if r['actual_endpoint'] else 'missing') == reason]
                summary = {}
                if reason != 'missing':
                    for field in ('joint', 'cartesian', 'activation', 'force_N', 'muscle_joint_torque_Nm'):
                        values = np.array([r['actual_endpoint'][field] for r in selected])
                        groups = defaultdict(list)
                        for r, v in zip(selected, values):
                            groups[episode_key(r)].append(v)
                        means = np.array([np.mean(v, axis=0) for v in groups.values()])
                        summary[field] = dict(episode_weighted_mean=means.mean(axis=0).tolist(),
                                              invocation_quantiles_5_50_95=np.quantile(values, [.05, .5, .95], axis=0).tolist())
                distributions.append(dict(policy_seed=seed, option=chr(64+choice), reason=reason,
                                          invocations=count, episodes=len({episode_key(r) for r in selected}),
                                          policies=1, distributions=summary))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source/'protocol.json', source/'branch_contrasts.json', source/'actual_endpoints.json', Path(__file__))}
    write(output/'saved_data_audit.json', dict(contrasts=results, termination_distributions=distributions,
          total_invocations=len(endpoints), missing_terminations=sum(r['actual_endpoint'] is None for r in endpoints),
          mirror_exclusions=dict(Counter(r['mirror_exclusion'] for r in endpoints if r['mirror_exclusion'])),
          source_hashes=hashes,
          inference='Descriptive exploratory intervals; no hypothesis tests or multiplicity-adjusted discoveries. Three policies, shared selector/reacher.'))
    print(f'Saved-data audit complete: {len(results)} contrasts, {len(distributions)} endpoint groups.')


def replay(source, output, pilot, per_cell=DEFAULTS["replay"]["per_cell"]):
    from rich.progress import Progress
    from scripts import evaluate_landmark_handoffs as h
    h.torch.set_num_threads(1)
    files = sorted((source/'branches').glob('*.json'), key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    # This diagnostic sample is not a powered physical-robustness study.
    selected = []
    metadata = {p: read(p)['metadata'] for p in files}
    if per_cell:
        for seed in (7, 101, 202):
            for choice in range(1, 6):
                candidates = [p for p in files if metadata[p]['policy_seed'] == seed and metadata[p]['choice'] == choice]
                orders = set()
                for p in candidates:
                    if metadata[p]['order'] not in orders:
                        selected.append(p)
                        orders.add(metadata[p]['order'])
                    if len(orders) == per_cell:
                        break
                assert orders, f'No states: seed {seed}, option {choice}'
                if len(orders) < per_cell:
                    print(f'Limited coverage: seed {seed}, option {choice}: {len(orders)} distinct order(s) available')
    else:
        for choice in range(1, 6):
            selected.append(next(p for p in files if metadata[p]['choice'] == choice))
        for seed in (7, 101, 202):
            if not any(metadata[p]['policy_seed'] == seed for p in selected):
                selected.append(next(p for p in files if metadata[p]['policy_seed'] == seed))
    protocol = dict(selection='Smallest SHA256(filename); up to requested distinct orders per policy/option cell; report shortages' if per_cell else 'Smallest SHA256(filename), one per option then missing base policies; outcome-blind',
                    per_cell=per_cell,
                    files=[p.name for p in selected], physical_dt=[.01, .001, .0005], command_dt=.01,
                    horizon_steps=100, scope='One-second local numerical screen; not original eight-second completion or population inference',
                    source_protocol_sha256=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = output/'replay_protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Protocol changed; use a new output directory'
    write(path, protocol)
    inputs = [p for file in selected for p in (file, file.with_name(file.stem+'_start.npz'))]
    write(output/'replay_input_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    original, reacher = h.e.models()
    library = h.e.read(h.e.LIBRARY)['options']
    started = time.perf_counter()
    panel = selected[:1] if pilot else selected
    with Progress() as progress:
        job = progress.add_task('Locked numerical replay', total=len(panel)*3)
        for file in panel:
            metadata = read(file)['metadata']
            seed = metadata['policy_seed']
            base = original if seed == 7 else h.SAC.load(str(h.POLICIES[seed]), device='cpu')
            base.policy.set_training_mode(False)
            with np.load(file.with_name(file.stem+'_start.npz')) as arrays:
                state = {k: arrays[k].copy() for k in arrays.files}
            for dt in protocol['physical_dt']:
                dest = output/'replay'/f'{file.stem}_dt{dt}.json'
                if not dest.exists():
                    result, traces = h.branches(state, metadata['choice'], library, (base, reacher), dt, 100)
                    write(dest, dict(metadata=metadata, physical_dt=dt, **result))
                    h.e.save_arrays(dest.with_suffix('.npz'), **{k: np.asarray(v) for k, v in traces.items()})
                progress.advance(job)
    write(output/('pilot_runtime.json' if pilot else 'replay_runtime.json'), dict(seconds=time.perf_counter()-started, states=len(panel)))
    comparisons = []
    for file in panel:
        results = {}
        traces = {}
        for dt in protocol['physical_dt']:
            dest = output/'replay'/f'{file.stem}_dt{dt}.json'
            results[dt] = read(dest)
            with np.load(dest.with_suffix('.npz')) as arrays:
                traces[dt] = arrays['phi'].copy()
        differences = {}
        for dt, result in results.items():
            by = {r['branch']: r for r in result['rows']}
            chosen = by['chosen']['progress_500ms_m']
            differences[str(dt)] = None if chosen is None or by['base']['progress_500ms_m'] is None else chosen-by['base']['progress_500ms_m']
        # Compare common timestamps only; early completion changes trace length.
        length = min(len(traces[.001]), len(traces[.0005]))
        assert [r['branch'] for r in results[.001]['rows']] == [r['branch'] for r in results[.0005]['rows']]
        errors = np.linalg.norm(traces[.001][:length, :, :2]-traces[.0005][:length, :, :2], axis=-1)
        comparisons.append(dict(file=file.name, metadata=read(file)['metadata'],
                                chosen_minus_base_progress_500ms_m=differences,
                                RK4_1ms_vs_half_ms_max_position_error_m=float(errors.max()),
                                common_trace_steps=length,
                                RK4_branch_names=[r['branch'] for r in results[.001]['rows']]))
    write(output/('pilot_comparison.json' if pilot else 'replay_comparison.json'), comparisons)
    print(f'Replay complete: {len(panel)} states; all comparator branches retained.')
    if not pilot:
        from scripts.plot_landmark_evidence import generate
        generate(source, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'data/handoffs')
    parser.add_argument('--output', type=Path, default=ROOT/'results/landmark_screen')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--per-cell', type=int, default=0, help='States from distinct orders per policy/option cell; 2 gives 30 states')
    args = parser.parse_args()
    if args.replay:
        if not 0 <= args.per_cell <= 24:
            parser.error('--per-cell must be 0–24')
        replay(args.source, args.output, args.pilot, args.per_cell)
    else:
        # Minimal aggregation check: repeated invocations receive equal episode weight.
        check = clustered_effect({(7, 0, 0, 0): [0, 1], (7, 0, 1, 0): [0]})
        assert check['mean'] == .25
        audit(args.source, args.output)
