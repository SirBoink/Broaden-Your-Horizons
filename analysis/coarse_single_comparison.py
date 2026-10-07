"""Exploratory 10 ms Euler comparison with a calibration-selected single-discount control."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

from rich.progress import track

from . import mechanism_experiment as e


def check_deadline(args):
    if args.deadline and time.time() >= args.deadline:
        raise TimeoutError('Bounded check stopped. Saved files resume with the same command without --max-seconds.')


def calibrate(args, root, source):
    if (root/'single_discount.json').exists():
        return
    policies = e.models()
    audit = e.read(source/'audit.json')
    plans = e.episode_plans(args, 'calibration', e.read(source/'panel.json'))
    e.collect_plans(args, root, 'calibration', plans, .01, policies, (15., 1.5))
    heads = [g for g, library in enumerate(audit['single_libraries']) if len(library) == 5]
    directory = root/'head_calibration'
    directory.mkdir(exist_ok=True)
    for plan in track(plans, description='10 ms: select single discount on calibration only'):
        check_deadline(args)
        path = directory/(plan['id']+'.json')
        if path.exists():
            continue
        data = e.load_episode(root/'calibration'/(plan['id']+'.npz'))
        stalls = e.stall_indices(data)
        step = stalls[0] if stalls else 30
        values = {}
        for start in range(0, len(heads), args.batch_size):
            check_deadline(args)
            batch = heads[start:start+args.batch_size]
            cases = [(str(g), audit['single_libraries'][g], False) for g in batch]
            outcomes = e.branch_many(data, step, .01, policies, (15., 1.5), cases, measure_probes=False)
            values.update({str(g): result['progress_500ms_m'] for g, result in zip(batch, outcomes)})
        e.write(path, values)
    records = [e.read(directory/(plan['id']+'.json')) for plan in plans]
    results = [dict(head=g, gamma=float(e.math.discount_grid()[g]), cardinality=5,
                    progress_m=float(e.np.mean([r[str(g)] for r in records]))) for g in heads]
    selected = max(results, key=lambda r: (r['progress_m'], -r['head']))
    e.write(root/'single_discount.json', dict(selected=selected, all_heads=results,
        library=audit['single_libraries'][selected['head']], calibration_contexts=[p['id'] for p in plans],
        selection='Mean 500 ms original-target progress; calibration only; ties lower head'))


def evaluate(args, root, source):
    policies = e.models()
    library = e.read(root/'single_discount.json')['library']
    panel = e.read(source/'panel.json')
    sequences = [v for group in panel['groups'].values() for v in group]
    directory = root/'single_transfer'
    directory.mkdir(exist_ok=True)
    for order in track(range(args.worker, len(sequences), args.workers), description=f'10 ms single worker {args.worker}'):
        check_deadline(args)
        path = directory/f'order_{order:02d}.json'
        if path.exists():
            continue
        base_path = source/'historical_transfer'/f'order_{order:02d}_base_base.json'
        contexts = [{key: row[key] for key in ('seed', 'posture', 'q', 'order')}
                    for row in e.read(base_path)['rows']]
        rows = []
        for start in range(0, len(contexts), args.batch_size):
            batch = contexts[start:start+args.batch_size]
            task, obs = e.make_task(sequences[order], .01, [r['q'] for r in batch], max_steps=800)
            outcomes = e.transfer_batch(task, obs, policies, (15., 1.5), 'single', library)
            rows.extend(dict(**context, **outcome) for context, outcome in zip(batch, outcomes))
        e.write(path, dict(rows=rows, physical_dt=.01, solver='Unmodified MotorNet Euler, float32',
            base_case_hash=e.sha(base_path), selected_library_hash=e.sha(root/'single_discount.json')))


def summarize(root, source):
    paths = list((root/'single_transfer').glob('*.json'))
    panel = e.read(source/'panel.json')
    order_groups = [name for name, sequences in panel['groups'].items() for _ in sequences]
    if len(paths) != len(order_groups):
        raise RuntimeError('Incomplete coarse single-discount panel')
    rows = [r for path in (source/'historical_transfer').glob('*.json') for r in e.read(path)['rows']]
    rows += [r for path in paths for r in e.read(path)['rows']]
    keyed = {(r['method'], r['seed'], r['order'], r['posture']): r for r in rows}
    expected = {(method, seed, order, posture) for method in ('base', 'learned', 'single')
                for seed in e.math.SEEDS for order in range(len(order_groups)) for posture in range(2)}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise RuntimeError('Missing, duplicate or unexpected coarse cases')
    groups = {}
    for name in ('all', *panel['groups']):
        selected = rows if name == 'all' else [r for r in rows if order_groups[r['order']] == name]
        methods = {}
        for method in ('base', 'single', 'learned'):
            values = [r for r in selected if r['method'] == method]
            methods[method] = dict(n=len(values), successes=sum(r['success'] for r in values),
                completion=float(e.np.mean([r['success'] for r in values])),
                restricted_completion_s=float(e.np.mean([r['restricted_completion_s'] for r in values])))
        contrasts = {}
        for other in ('base', 'single'):
            means = [float(e.np.mean([int(r['success'])-int(keyed[(other, seed, r['order'], r['posture'])]['success'])
                     for r in selected if r['method'] == 'learned' and r['seed'] == seed])) for seed in e.math.SEEDS]
            contrasts[other] = e.math.interval(means)
        groups[name] = dict(methods=methods, learned_minus_control=contrasts)
    e.write(root/'summary.json', dict(comparison_complete=True, exploratory=True, physical_dt=.01,
        numerically_converged=False, selected_single=e.read(root/'single_discount.json')['selected'], groups=groups,
        scope='Frozen learned A--E versus baseline and calibration-selected single; historical Euler only',
        limitations='Learned selector and fixed-rule single have different decision rules; no human-controller fit; seed intervals describe the fixed sequence panel'))
    print(f'Comparison: {root / "summary.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=e.ROOT/'data/mechanism')
    parser.add_argument('--output', type=Path, default=e.ROOT/'results/single_discount')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--max-seconds', type=int, help='Bounded check; omit for an operator-run resumption')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    e.torch.set_num_threads(1)
    root, source = args.output.resolve(), args.source.resolve()
    if root == source:
        parser.error('Use a separate output directory')
    source_protocol = e.read(source/'protocol.json')
    args.legacy_root = Path(source_protocol['legacy_root'])
    args.batch_size, args.limit, args.device = 8, None, 'cpu'
    original = copy.copy(args)
    original.output = source
    e.manifest(original)
    protocol = dict(exploratory=True, physical_dt=.01, solver='Unmodified MotorNet Euler, float32',
        runner_hash=e.sha(Path(__file__)), source_protocol_hash=e.sha(source/'protocol.json'),
        source_artifacts={str(p.relative_to(source)): e.sha(p) for p in
            [source/'audit.json', source/'panel.json', *sorted((source/'historical_transfer').glob('*.json'))]},
        batch_size=8, single_selection='Same independent calibration contexts and criterion as the 1 ms study')
    if (root/'protocol.json').exists() and e.read(root/'protocol.json') != protocol:
        raise ValueError('Comparison inputs changed; use a new output directory')
    if not (root/'protocol.json').exists():
        e.write(root/'protocol.json', protocol)
    if args.worker is not None:
        evaluate(args, root, source)
        return
    if args.max_seconds:
        args.deadline = time.time()+args.max_seconds
    calibrate(args, root, source)
    check_deadline(args)
    children = []
    logs = root/'worker_logs'
    logs.mkdir(exist_ok=True)
    try:
        for worker in range(args.workers):
            stream = (logs/f'worker_{worker}.log').open('wb')
            process = subprocess.Popen([sys.executable, '-B', '-m', 'analysis.coarse_single_comparison',
                '--source', str(source), '--output', str(root), '--workers', str(args.workers),
                '--worker', str(worker), '--deadline', str(args.deadline)], cwd=e.ROOT,
                stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            children.append((process, stream))
        from rich.progress import Progress
        total = sum(map(len, e.read(source/'panel.json')['groups'].values()))
        with Progress() as progress:
            task_id = progress.add_task('10 ms: complete paired single-discount panel', total=total)
            while True:
                check_deadline(args)
                codes = [p.poll() for p, _ in children]
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError(f'Worker exited unsuccessfully; inspect {logs}')
                progress.update(task_id, completed=len(list((root/'single_transfer').glob('*.json'))))
                if all(code == 0 for code in codes):
                    break
                time.sleep(1)
        summarize(root, source)
    finally:
        for process, stream in children:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


if __name__ == '__main__':
    main()
