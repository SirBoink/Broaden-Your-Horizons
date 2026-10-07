"""Resume independent mechanism stages concurrently, preserving the frozen experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn


def shard_jobs(rows, worker, workers, limit=None):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid transfer worker index or count')
    return (rows if limit is None else rows[:limit])[worker::workers]


def transfer_paths(root, limit=None):
    import json
    panel = json.loads((root/'panel.json').read_text(encoding='utf-8'))
    orders = sum(map(len, panel['groups'].values()))
    paths = [root/'transfer'/f'order_{order:02d}_{controller}_{method}.json'
             for order in range(orders) for controller in ('base', 'reacher', 'pd')
             for method in ('base', 'learned', 'direct', 'multiscale', 'single')
             if method != 'learned' or controller == 'base']
    return paths if limit is None else paths[:limit]


def neural_detail(root, log):
    fresh = len(list((root/'neural/fresh_physical_state').glob('*.npz')))
    evaluations = len(list((root/'neural').glob('evaluation_*.json')))
    if evaluations or fresh:
        return f'{fresh} fresh episodes; {evaluations} evaluations'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-8192))
            epochs = re.findall(r'Seed (\d+), epoch (\d+)', stream.read().decode('utf-8', errors='replace'))
        if epochs:
            seed, epoch = epochs[-1]
            return f'seed {seed}, epoch {epoch}'
    return 'loading frozen data'


def run(args, experiment, root):
    paths = transfer_paths(root, args.limit)
    if args.plan:
        print(f'Transfer: {sum(p.exists() for p in paths)}/{len(paths)} saved; {args.workers} CPU workers')
        print(f'Neural: one {args.device} worker; human: one CPU worker; report after all finish')
        print('Stop the existing --stage all process with Ctrl+C before starting this launcher.')
        return

    logs = root/'parallel_logs'
    logs.mkdir(exist_ok=True)
    experiment.write(root/'parallel_run.json', dict(launcher_hash=experiment.sha(Path(__file__)),
        protocol_hash=experiment.sha(root/'protocol.json'), transfer_workers=args.workers,
        device=args.device, started_unix_s=time.time(), completed=False))
    common = ['--output', str(root), '--legacy-root', str(args.legacy_root),
              '--batch-size', str(args.batch_size), '--device', args.device, '--workers', str(args.workers)]
    if args.limit is not None:
        common += ['--limit', str(args.limit)]
    names = [(f'transfer_{i}', 'transfer', i) for i in range(args.workers)]
    names += [('neural', 'neural', 0), ('human', 'human', 0)]
    children = []
    print('Resuming saved jobs. Each stage has one owner; transfer jobs have disjoint owners.', flush=True)
    try:
        for name, stage, index in names:
            stream = (logs/f'{name}.log').open('wb')
            try:
                process = subprocess.Popen([sys.executable, '-B', '-m', 'analysis.mechanism_parallel',
                    '--worker', stage, '--worker-index', str(index), *common], cwd=experiment.ROOT,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except BaseException:
                stream.close()
                raise
            children.append((name, process, stream))
        fit_seeds = (101, 202, 303, 404) if args.limit is None else (101,)
        human_models = 60 if args.limit is None else 2
        with Progress(TextColumn('{task.description}'), BarColumn(), MofNCompleteColumn(),
                      TimeElapsedColumn(), TextColumn('{task.fields[detail]}')) as progress:
            transfer = progress.add_task(f'Transfer ({args.workers} workers)', total=len(paths), detail='')
            neural = progress.add_task('Neural (fits + analysis)', total=len(fit_seeds)+1, detail='')
            human = progress.add_task('Human (models + analysis)', total=human_models+1, detail='')
            while True:
                codes = [(name, process.poll()) for name, process, _ in children]
                for name, code in codes:
                    if code is not None and code != 0:
                        raise RuntimeError(f'{name} exited with code {code}; inspect {logs / (name+".log")}')
                finished = {name for name, code in codes if code == 0}
                progress.update(transfer, completed=sum(p.exists() for p in paths), detail='frozen 1 ms physics')
                fits = sum((root/f'neural/seed_{seed}/complete.json').exists() for seed in fit_seeds)
                progress.update(neural, completed=len(fit_seeds)+1 if 'neural' in finished else fits,
                                detail='done' if 'neural' in finished else neural_detail(root, logs/'neural.log'))
                models = sum(len(list((root/'human').glob(f'{family}_*.npz'))) for family in ('flat', 'hierarchical'))
                progress.update(human, completed=human_models+1 if 'human' in finished else models,
                                detail='done' if 'human' in finished else 'published models and participants')
                if len(finished) == len(children):
                    break
                time.sleep(1)
        if not all(p.exists() for p in paths):
            raise RuntimeError('Transfer workers exited without completing the prescribed panel')
        metadata = experiment.read(root/'parallel_run.json')
        metadata.update(completed=True, finished_unix_s=time.time())
        experiment.write(root/'parallel_run.json', metadata)
        print('All workers finished. Building the final report.', flush=True)
        experiment.stage_report(args, root)
    finally:
        for _, process, stream in children:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


def main():
    from . import mechanism_experiment as experiment
    experiment.torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=experiment.ROOT/'data/mechanism')
    parser.add_argument('--legacy-root', type=Path, default=experiment.LEGACY)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--limit', type=int, help='Separate smoke protocol only')
    parser.add_argument('--plan', action='store_true', help='Check prerequisites and print the schedule without running')
    parser.add_argument('--worker', choices=('transfer', 'neural', 'human'), help=argparse.SUPPRESS)
    parser.add_argument('--worker-index', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and smoke limit must be positive')
    for file in ('protocol.json', 'audit.json', 'calibration_complete.json', 'single_discount.json'):
        if not (args.output/file).is_file():
            parser.error(f'Finish calibration before using this resume launcher: missing {file}')
    root = experiment.manifest(args)
    if args.worker == 'transfer':
        # Only this child changes the scheduling hook; each job still runs the frozen implementation.
        experiment.limited = lambda stage_args, rows: shard_jobs(rows, args.worker_index, args.workers, stage_args.limit)
        experiment.stage_transfer(args, root)
    elif args.worker:
        from . import mechanism_validation
        getattr(mechanism_validation, 'stage_'+args.worker)(args, root)
    else:
        plans = experiment.episode_plans(args, 'confirmation', experiment.read(root/'panel.json'))
        if any(not (root/'mechanism'/(plan['id']+'.json')).exists()
               or not (root/'confirmation'/(plan['id']+'.npz')).exists() for plan in plans):
            parser.error('Finish confirmation collection and mechanism analysis before using this resume launcher')
        run(args, experiment, root)


if __name__ == '__main__':
    main()
