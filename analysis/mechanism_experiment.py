"""Resumable multiscale SF / A--E mechanism experiment. No training on import.

Use --stage preflight first. Long stages are explicitly requested by the operator.
Historical discovery, fixed-rule utility and learned-selector utility stay separate.
"""
from __future__ import annotations

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("analysis/mechanism_experiment.py", {})
DEFAULTS = SETTINGS["defaults"].get("analysis/mechanism_experiment.py", {})

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.progress import track
from scipy.optimize import lsq_linear
import torch
import motornet as mn
from stable_baselines3 import SAC

import formulas as math
from .mechanism_physics import RK4Arm26
from SAC.changed.env import StraightReachEnv
from SAC.high_level import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT
BASE = ROOT / 'checkpoints/base_policy/best_model.zip'
REACHER = ROOT / 'checkpoints/option_reacher/best_model.zip'
LIBRARY = ROOT / 'data/discovery/options/off_target_band_options.json'
SELECTOR = ROOT / 'checkpoints/selector/accepted.pt'
SOURCE_SEQUENCE = tuple(PARAMS["SOURCE_SEQUENCE"])
FIGURES = ROOT / 'figures/mechanism'
PROTOCOL = SETTINGS["protocol"].copy()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def save_arrays(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()


def sources(args):
    old = args.legacy_root
    return dict(base=BASE, reacher=REACHER, library=LIBRARY, selector=SELECTOR,
                reaches=old / 'data/discovery/isolated_reaches/reaches.npz',
                bands=old / 'data/discovery/isolated_reaches/reach_bands.npz',
                candidates=old / 'data/discovery/isolated_reaches/relaxed_off_target_candidates.json',
                normalization=old / 'data/discovery/normalization/temporal_band_pilot.npz',
                panel=ROOT / 'config/heldout_panel.json')


def manifest(args):
    paths = sources(args)
    absent = [str(p) for p in paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError('Missing required research artifacts: ' + ', '.join(absent))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = dict(protocol=PROTOCOL, legacy_root=str(args.legacy_root.resolve()),
                source_hashes={k: sha(p) for k, p in paths.items()},
                analysis_hashes={str(p.relative_to(ROOT)):sha(p) for p in [
                    (ROOT/'formulas.py' if name == 'math' else ROOT/'analysis'/f'mechanism_{name}.py') for name in ('math','physics','experiment','validation','checks')]
                    + [ROOT/p for p in ('SAC/changed/env.py','SAC/core/env.py','SAC/core/ppo_env.py','SAC/options.py','SAC/high_level.py')]},
                environment=dict(python=platform.python_version(),packages={name:(torch.__version__ if name == "torch" else mn.__version__ if name == "motornet" else version(name)) for name in
                    ('numpy','scipy','torch','motornet','gymnasium','stable-baselines3')}),
                batch_size=args.batch_size, smoke_limit=args.limit)
    path = root / 'protocol.json'
    if path.exists() and read(path) != data:
        raise ValueError('Protocol or source artifacts changed. Use a new output directory.')
    if not path.exists():
        write(path, data)
        write(root / 'panel.json', read(paths['panel']))
    return root


def limited(args, rows):
    return rows if args.limit is None else rows[:args.limit]


def make_task(sequence=SOURCE_SEQUENCE, physical_dt=DEFAULTS["make_task"]["physical_dt"], postures=None, max_steps=DEFAULTS["make_task"]["max_steps"]):
    if physical_dt <= 0 or not np.isclose(.01 / physical_dt, round(.01 / physical_dt)):
        raise ValueError('Physical step must divide 10ms commands exactly')
    arm = mn.effector.RigidTendonArm26 if physical_dt == .01 else RK4Arm26
    dtype = torch.float32 if physical_dt == .01 else torch.float64
    task = StraightReachEnv(arm(
        muscle=mn.muscle.RigidTendonHillMuscle(), timestep=.01,
        n_ministeps=int(round(.01 / physical_dt))), sequence,
        differentiable=False, action_noise=0., obs_noise=0., max_episode_steps=max_steps,
        random_start=False, start_pos=[45., 90.])
    postures = np.asarray([[45., 90.]] if postures is None else postures, dtype=np.float32)
    if postures.ndim != 2 or postures.shape[1] != 2 or not np.isfinite(postures).all():
        raise ValueError('Postures must be finite N-by-2 angles in degrees')
    task.reset(seed=270000, options={'batch_size': len(postures), 'deterministic': True})
    joint = torch.tensor(np.column_stack((np.deg2rad(postures), np.zeros_like(postures))), dtype=dtype)
    if torch.any(joint[:, :2] < task.effector.pos_lower_bound) or torch.any(joint[:, :2] > task.effector.pos_upper_bound):
        raise ValueError('Initial posture violates joint bounds')
    task.effector.reset(options={'batch_size': len(postures), 'joint_state': joint})
    if isinstance(task.effector, RK4Arm26):
        task.effector.synchronize()
    zero = torch.zeros((len(postures), 6))
    task.obs_buffer['vision'] = [task.get_vision().clone() for _ in task.obs_buffer['vision']]
    task.obs_buffer['proprioception'] = [task.get_proprioception().clone() for _ in task.obs_buffer['proprioception']]
    task.obs_buffer['action'] = [zero.clone() for _ in task.obs_buffer['action']]
    observation = task.get_obs(action=zero, deterministic=True)
    assert np.shape(observation) == (len(postures), 30)
    return task, np.asarray(observation, dtype=np.float32)


def features(task):
    return np.column_stack((array(task.states['cartesian']), array(task.states['muscle'][:, 0, :])))


def physical_state(task):
    """Original SF conditioning state: [q,dq,activation,goal,length,velocity], 24D."""
    muscle=array(task.states['muscle'])
    return np.column_stack((array(task.states['joint']),muscle[:,0],array(task.goal),muscle[:,1],muscle[:,2]))


def muscle_forces(effector, activation):
    """Evaluate forces without advancing time, through the public muscle API."""
    activation = torch.as_tensor(activation, dtype=torch.float32).reshape(-1, 1, 6)
    state = effector.muscle.integrate(0., torch.zeros_like(activation), activation, effector.states['geometry'])
    return state[:, effector.force_index, :]


def pd_command(task, gains, goal=None):
    goal = array(task.goal) if goal is None else np.asarray(goal)
    skeleton = task.effector.skeleton
    x, y = goal.T
    cos = (x*x + y*y - skeleton.L1**2 - skeleton.L2**2) / (2*skeleton.L1*skeleton.L2)
    if np.any(np.abs(cos) > 1 + 1e-6):
        raise ValueError('Feedback goal outside arm workspace')
    elbow = np.arccos(np.clip(cos, -1, 1))
    shoulder = np.arctan2(y, x) - np.arctan2(skeleton.L2*np.sin(elbow), skeleton.L1+skeleton.L2*np.cos(elbow))
    q = array(task.states['joint'])
    desired = gains[0] * (np.column_stack((shoulder, elbow)) - q[:, :2]) - gains[1] * q[:, 2:]
    lower = muscle_forces(task.effector, np.full((len(q), 6), .001))
    upper = muscle_forces(task.effector, np.ones((len(q), 6)))
    moments = array(task.states['geometry'])[:, 2:, :]
    columns = -moments * array(upper - lower)[:, None, :]
    passive = np.sum(-moments * array(lower)[:, None, :], axis=-1)
    commands = []
    for matrix, demand, offset in zip(columns, desired, passive):
        fitted = lsq_linear(np.vstack((matrix, .01*np.eye(6))),
                            np.r_[demand-offset, np.zeros(6)], bounds=(0., 1.))
        if not fitted.success:
            raise RuntimeError('Bounded muscle allocation failed: ' + fitted.message)
        commands.append(.001+.999*fitted.x)
    return np.asarray(commands, dtype=np.float32)


def models():
    base, reacher = SAC.load(str(BASE), device='cpu'), SAC.load(str(REACHER), device='cpu')
    if base.observation_space.shape != (30,) or reacher.observation_space.shape != (22,):
        raise ValueError('Controller observation dimensions mismatch')
    for model in (base, reacher):
        if model.action_space.shape != (6,):
            raise ValueError('Controllers must output six actions')
        model.policy.set_training_mode(False)
    return base, reacher


def commands(controller, task, observation, policies, gains, goal=None):
    if controller == 'pd':
        return pd_command(task, gains, goal)
    if controller == 'reacher' or goal is not None:
        goal = array(task.goal) if goal is None else np.asarray(goal)
        obs = np.column_stack((goal, observation[:, 10:]))
        action, _ = policies[1].predict(obs, deterministic=True)
    else:
        action, _ = policies[0].predict(observation, deterministic=True)
    return (np.clip(np.asarray(action, dtype=np.float32), -1, 1) + 1) / 2


def capture(task, observation):
    result = {f'state_{k}': array(v) for k, v in task.states.items()}
    result.update(phi=features(task), observation=observation.copy(), goal=array(task.goal),
                  visit=array(task.sequence_index), finished=array(task.finished), hold=array(task.hold_count))
    result.update({f'buffer_{k}': np.stack([array(v) for v in values], axis=1)
                   for k, values in task.obs_buffer.items() if len(values)})
    return result


def record(task, observation, controller, policies, gains, steps, redirect=None):
    rows, applied = [], []
    hits = np.full(len(observation), -1, dtype=int)
    for step in range(steps + 1):
        if redirect is not None and step == 30:
            task._sequence[:] = redirect
            task.sequence_index.zero_()
            task.finished.zero_()
            task.hold_count.zero_()
            task.goal = task.targets[redirect].repeat(len(observation), 1)
            # Change only current goal/identity in the cached observation. No buffer advance.
            observation[:, :2] = array(task.goal)
            observation[:, 2:10] = np.eye(8, dtype=np.float32)[redirect]
        rows.append(capture(task, observation))
        if step == steps:
            break
        excitation = commands(controller, task, observation, policies, gains)
        applied.append(excitation.copy())
        observation, _, _, _, info = task.step(excitation, deterministic=True)
        observation = np.asarray(observation, dtype=np.float32)
        completed = np.asarray(info['completed'], dtype=bool).reshape(-1)
        hits[(hits < 0) & completed] = step + 1
    result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    if not np.isfinite(result['phi']).all():
        raise FloatingPointError('Nonfinite physical features during recording')
    result.update(excitation=np.asarray(applied), first_hit=hits,
                  sequence=array(task._sequence), time_s=np.arange(steps+1)*.01)
    return result


def restore(data, k, physical_dt):
    sequence = data['sequence']
    if 'metadata' in data:
        metadata = json.loads(str(data['metadata']))
        if metadata.get('redirect') is not None and k < 30:
            sequence = metadata['sequence']
    task, _ = make_task(tuple(map(int, sequence)), physical_dt)
    dtype = torch.float32 if physical_dt == .01 else torch.float64
    task.effector.states = {key[6:]: torch.tensor(data[key][k:k+1], dtype=dtype)
                            for key in data if key.startswith('state_')}
    if isinstance(task.effector, RK4Arm26):
        task.effector.synchronize()
    task.sequence_index = torch.tensor(data['visit'][k:k+1], dtype=torch.long)
    task.finished = torch.tensor(data['finished'][k:k+1], dtype=torch.bool)
    task.hold_count = torch.tensor(data['hold'][k:k+1], dtype=torch.long)
    task.goal = torch.tensor(data['goal'][k:k+1], dtype=torch.float32)
    task.elapsed_steps, task.elapsed = k, k*.01
    for key in task.obs_buffer:
        name = 'buffer_' + key
        if name in data:
            task.obs_buffer[key] = [torch.tensor(data[name][k:k+1, j], dtype=torch.float32)
                                    for j in range(data[name].shape[1])]
    return task, data['observation'][k:k+1].copy()


def pulse_probe(task, goal):
    return pulse_batch(task.effector,np.asarray(goal).reshape(1,2))[0]


def pulse_batch(source, goals):
    """13 prescribed branches per state, vectorized across independent snapshots."""
    effector=copy.deepcopy(source)
    neutral=array(effector.states['muscle'])[:,0]
    n=len(neutral)
    u=np.repeat(neutral[:,None,:],13,axis=1)
    for muscle in range(6):
        u[:,1+2*muscle,muscle]=np.minimum(1.,neutral[:,muscle]+.05)
        u[:,2+2*muscle,muscle]=np.maximum(.001,neutral[:,muscle]-.05)
    effector.states = {key: value.repeat_interleave(13,dim=0)
                       for key, value in effector.states.items()}
    x0=array(source.states['fingertip'])
    goals=np.asarray(goals).reshape(n,2)
    direction=goals-x0
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    results=[[] for _ in range(n)]
    for step in range(1, 31):
        effector.step(u.reshape(-1,6) if step<=5 else np.repeat(neutral,13,axis=0))
        if step in (5, 15, 30):
            xy=array(effector.states['fingertip']).reshape(n,13,2)
            for row in range(n):
                index=np.arange(6)
                delta=u[row,1+2*index,index]-u[row,2+2*index,index]
                jacobian=((xy[row,1::2]-xy[row,2::2])/delta[:,None]).T
                singular=np.linalg.svd(jacobian,compute_uv=False)
                results[row].append(dict(horizon_s=step*.01, response_matrix=jacobian,
                                minimum_gain_m=float(singular[-1]),
                                directional_gain_m=float(np.linalg.norm(direction[row]@jacobian)),
                                best_probe_progress_m=float(np.max(np.linalg.norm(goals[row]-x0[row])-np.linalg.norm(goals[row]-xy[row],axis=1)))))
    return results


def stall_indices(data, maximum_steps=DEFAULTS["stall_indices"]["maximum_steps"]):
    distance = np.linalg.norm(data['phi'][:, :2]-data['goal'], axis=1)
    target = np.argmax(data['observation'][:, 2:10], axis=1)
    selected, seen = [], set()
    for k in range(10, min(len(distance), maximum_steps+1)):
        visit = (int(target[k]), int(data['visit'][k]))
        if (not bool(data['finished'][k]) and visit not in seen and np.all(target[k-10:k+1] == target[k])
                and distance[k] > .04 and distance[k-10]-distance[k] < .001):
            selected.append(k)
            seen.add(visit)
    return selected


def nearest_option(xy, library):
    available = [(np.linalg.norm(xy-np.asarray(item['xy_m'])), index)
                 for index, item in enumerate(library) if np.linalg.norm(xy-np.asarray(item['xy_m'])) > .02]
    return min(available)[1] if available else None


def branch(data, k, physical_dt, policies, gains, library=None, direct=DEFAULTS["branch"]["direct"], measure_probes=DEFAULTS["branch"]["measure_probes"]):
    return branch_many(data,k,physical_dt,policies,gains,[('branch',library,direct)],measure_probes)[0]


def branch_many(data,k,physical_dt,policies,gains,cases,measure_probes=DEFAULTS["branch_many"]["measure_probes"]):
    """Paired interventions share a native batch, with independent task/option clocks."""
    task, obs = restore(data, k, physical_dt)
    original_goal = array(task.goal)[0]
    original_visit = int(task.sequence_index[0])
    d0 = np.linalg.norm(features(task)[0, :2]-original_goal)
    controller = json.loads(str(data['metadata'])).get('controller','base') if 'metadata' in data else 'base'
    n=len(cases)
    chosen=[nearest_option(features(task)[0,:2],library) if library else None for _,library,_ in cases]
    destinations=np.asarray([library[c]['xy_m'] if c is not None else original_goal
                             for c,(_,library,_) in zip(chosen,cases)])
    direct=np.array([v[2] for v in cases],dtype=bool)
    available=np.array([v is not None for v in chosen])|direct
    task.effector.states={key:value.repeat((n,)+(1,)*(value.ndim-1)) for key,value in task.states.items()}
    for key in ('sequence_index','finished','hold_count'):
        setattr(task,key,getattr(task,key).repeat(n))
    task.goal=task.goal.repeat(n,1)
    task.obs_buffer={key:[v.repeat(n,1) for v in values] for key,values in task.obs_buffer.items()}
    obs=np.repeat(obs,n,axis=0)
    acquired=np.full(n,np.nan);progress=None;intervention_steps=np.zeros(n,dtype=int);post_probe=[None]*n
    for offset in range(100):
        available &= direct | (np.linalg.norm(features(task)[:,:2]-destinations,axis=1)>.02)
        use=np.flatnonzero(np.isnan(acquired)&available&(offset<30))
        excitation=commands(controller,task,obs,policies,gains)
        if len(use):
            action,_=policies[1].predict(np.column_stack((destinations[use],obs[use,10:])),deterministic=True)
            excitation[use]=(np.clip(action,-1,1)+1)/2
            intervention_steps[use]+=1
        obs, _, terminated, truncated, _ = task.step(excitation, deterministic=True)
        obs = np.asarray(obs, dtype=np.float32)
        hit=array(task.finished).astype(bool)|(array(task.sequence_index)!=original_visit)
        acquired[np.isnan(acquired)&hit]=(offset+1)*.01
        if offset == 29 and measure_probes:
            post_probe=pulse_batch(task.effector,np.tile(original_goal,(n,1)))
        if offset == 49:
            # First acquisition is absorbing for this original-target endpoint.
            progress=np.where(np.isfinite(acquired),d0,d0-np.linalg.norm(features(task)[:,:2]-original_goal,axis=1))
        # Continue genuine physics after acquisition to measure a common-time endpoint.
        if truncated:
            break
    return [dict(method=case[0],acquired=bool(np.isfinite(acquired[row])),
                acquisition_s=float(acquired[row]) if np.isfinite(acquired[row]) else None,
                restricted_acquisition_s=float(acquired[row]) if np.isfinite(acquired[row]) else 1.,
                progress_500ms_m=float(progress[row]) if progress is not None else None,
                intervention_steps=int(intervention_steps[row]),option_index=chosen[row],
                post_intervention_probes=post_probe[row],
                progress_definition='500ms original-target progress; acquisition treated as absorbing') for row,case in enumerate(cases)]


def load_episode(path):
    with np.load(path) as archive:
        return {k: archive[k].copy() for k in archive.files}


def stats(args):
    with np.load(sources(args)['normalization']) as z:
        return z['feature_mean_raw'].copy(), z['feature_std'].copy()


def stage_audit(args, root):
    paths = sources(args)
    data = load_episode(paths['reaches'])
    if str(data['checkpoint_sha256'])!=sha(BASE) or str(data['normalization_sha256'])!=sha(paths['normalization']):
        raise ValueError('Original trajectories were generated by different controller or normalization artifacts')
    mean, std = stats(args)
    task, _ = make_task(physical_dt=.01)
    goals = array(task.targets)
    source = np.flatnonzero((data['cohort'] == 'cycles') & (data['repeat'] < 3))
    raw = []
    single = []
    for trial in track(source, description='Reproduce original 40-discount discovery'):
        multi, heads = math.scores(data['phi'][trial], mean, std)
        raw.append(multi[:, :int(data['hit'][trial])])
        single.append(heads)
    refs = math.reference(np.concatenate(raw, axis=1))
    selected = []
    for trial, values in zip(source, raw):
        for event in math.events(math.percentiles(values, refs), data['phi'][trial],
                                 int(data['hit'][trial]), goals, data['target'][trial], trial):
            if not any(math.same_event(event, old) for old in selected):
                selected.append(event)
    selected = sorted([e for e in selected if e['nearest_target_m'] > .025], key=lambda e: (e['target'], e['step']))
    expected = read(paths['candidates'])
    for i, event in enumerate(selected):
        event['id'] = chr(65+i)
        event['raw_change'] = float(raw[list(source).index(event['trial'])][event['peak_band'], event['step']-1])
    if [(v['trial'], v['step'], v['peak_band']) for v in selected] != [(v['trial'], v['step'], v['peak_band']) for v in expected]:
        raise AssertionError('Original A--E discovery did not reproduce; do not proceed')
    library = read(LIBRARY)['options']
    if len(library)!=5 or [v['choice'] for v in library]!=list(range(1,6)):
        raise ValueError('The A--E selector requires exactly five ordered options')
    for candidate, option in zip(selected, library):
        if not np.allclose(candidate['xy_m'], option['xy_m'], atol=1e-6):
            raise AssertionError('Learned-selector library differs from reproduced A--E')
    alternatives = [math.single_candidates([v[g] for v in single], data['phi'][source], data['hit'][source], goals)
                    for g in range(40)]
    write(root / 'audit.json', dict(candidates=selected, original_library=library,
          single_libraries=alternatives, source_discovery_trials=source.tolist(),
          learned_model_source='later final_model checkpoint, evaluated separately',
          gamma_grid=math.discount_grid(), feature_names=['x','y','vx','vy']+[f'activation_{i+1}' for i in range(6)],
          controller_seed_metadata={name:SAC.load(str(path),device='cpu').seed for name,path in [('base',BASE),('reacher',REACHER)]}))
    save_arrays(root / 'references.npz', **{f'band_{i}': ref for i, ref in enumerate(refs)})


def episode_plans(args, cohort, panel):
    seeds = (270001, 270002) if cohort == 'calibration' else math.SEEDS
    sequences = [tuple(v) for values in panel['groups'].values() for v in values]
    rows = []
    for controller in (('base',) if cohort=='calibration' else ('base','reacher','pd')):
        for task_name in (('isolated',) if cohort=='calibration' else ('isolated','sequence','redirect')):
            for trial in range(8):
                target = SOURCE_SEQUENCE[trial]
                sequence = (target,)*2000 if task_name != 'sequence' else sequences[trial]
                for seed in seeds:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, trial, 610]))
                    rows.append(dict(id=f'{controller}_{task_name}_{seed}_{trial}', controller=controller,
                                     task=task_name, seed=int(seed), trial=trial, sequence=sequence,
                                     posture=(np.array([45.,90.])+rng.uniform(-3,3,2)).tolist(),
                                     redirect=SOURCE_SEQUENCE[(trial+1)%8] if task_name=='redirect' else None))
    return limited(args, rows)


def collect_plans(args, root, cohort, plans, physical_dt, policies, gains):
    directory = root / cohort
    directory.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for plan in plans:
        if not (directory / (plan['id']+'.npz')).exists():
            groups[(plan['controller'],plan['task'],plan['trial'])].append(plan)
    batches = [values[i:i+args.batch_size] for values in groups.values() for i in range(0,len(values),args.batch_size)]
    for batch in track(batches, description=f'Collect {cohort} native MotorNet batches'):
        task, obs = make_task(batch[0]['sequence'], physical_dt, [p['posture'] for p in batch])
        data = record(task, obs, batch[0]['controller'], policies, gains, 1530, batch[0]['redirect'])
        for row, plan in enumerate(batch):
            arrays = {key:(value[:,row] if key not in ('sequence','time_s','first_hit') else
                           value[row] if key=='first_hit' else value) for key,value in data.items()}
            arrays['metadata'] = np.asarray(json.dumps(clean(plan)))
            save_arrays(directory/(plan['id']+'.npz'), **arrays)
    write(directory/'index.json', plans)


def original_reach(args, trial, policies):
    """Recreate an original cycle prefix before branching; no inferred muscle state."""
    source=load_episode(sources(args)['reaches'])
    if source['cohort'][trial]!='cycles':
        raise ValueError('Integration anchors must belong to original cycle trials')
    target=int(source['target'][trial]);offset=SOURCE_SEQUENCE.index(target)
    prefix=SOURCE_SEQUENCE*int(source['repeat'][trial])+SOURCE_SEQUENCE[:offset+1]
    task,obs=make_task(prefix+(target,)*1532,.01,[source['start_deg'][trial]],max_steps=10000)
    for _ in range(5000):
        if int(task.sequence_index[0])==len(prefix)-1:break
        obs,_,done,truncated,_=task.step(commands('base',task,obs,policies,(15.,1.5)),deterministic=True)
        obs=np.asarray(obs,dtype=np.float32)
        if done or truncated:raise RuntimeError('Original prefix ended before activation')
    else:raise RuntimeError('Original target did not activate within 5000 commands')
    data=record(task,obs,'base',policies,(15.,1.5),1530)
    row={key:(value[:,0] if key not in ('sequence','time_s','first_hit') else
              value[0] if key=='first_hit' else value) for key,value in data.items()}
    np.testing.assert_allclose(row['phi'],source['phi'][trial],atol=2e-6,rtol=0,
                               err_msg='Historical physical rollout did not reproduce')
    return row


def integration_check(args,root,policies):
    """Replay all unique A--E source trajectories from identical full snapshots."""
    anchors=sorted({c['trial'] for c in read(root/'audit.json')['candidates']})
    directory=root/'integration';directory.mkdir(exist_ok=True)
    mean,std=stats(args)
    for trial in track(anchors,description='Reproduce physical source trajectories for all A--E anchors'):
        path=directory/f'source_{trial}.npz'
        if not path.exists():save_arrays(path,**original_reach(args,trial,policies))
    def replay(dt):
        pending=[trial for trial in anchors if not (directory/f'trial_{trial}_dt_{dt}.npz').exists()]
        for start in range(0,len(pending),args.batch_size):
            trials=pending[start:start+args.batch_size]
            data=[load_episode(directory/f'source_{trial}.npz') for trial in trials]
            if dt==.01:
                phi=np.stack([row['phi'] for row in data],axis=1)
            else:
                task,_=restore(data[0],0,dt)
                task.effector.states={key[6:]:torch.tensor(np.concatenate([row[key][:1] for row in data]),dtype=torch.float64)
                                      for key in data[0] if key.startswith('state_')}
                task.effector.synchronize()
                rows=[features(task)]
                excitations=np.stack([row['excitation'] for row in data],axis=1)
                for excitation in track(excitations,description=f'Frozen commands: RK4 {dt*1000:g}ms, {len(trials)} anchors'):
                    task.effector.step(excitation);rows.append(features(task))
                phi=np.asarray(rows)
            for row,trial in enumerate(trials):
                score,_=math.scores(phi[:,row],mean,std)
                save_arrays(directory/f'trial_{trial}_dt_{dt}.npz',phi=phi[:,row],score=score[:,:180])
    def compare(a,b):
        rows=[]
        for trial in anchors:
            left=load_episode(directory/f'trial_{trial}_dt_{a}.npz')
            right=load_episode(directory/f'trial_{trial}_dt_{b}.npz')
            x,y=left['phi'],right['phi'];c,d=left['score'],right['score']
            meaningful=(np.nanmax(c,axis=1)>=PROTOCOL['meaningful_change'])&(np.nanmax(d,axis=1)>=PROTOCOL['meaningful_change'])
            shifts=np.abs(np.nanargmax(c,axis=1)-np.nanargmax(d,axis=1))
            rows.append(dict(trial=trial,position_error_m=float(np.linalg.norm(x[:181,:2]-y[:181,:2],axis=1).max()),
                full_trace_position_error_m=float(np.linalg.norm(x[:,:2]-y[:,:2],axis=1).max()),
                activation_error=float(np.abs(x[:181,4:]-y[:181,4:]).max()),
                score_error=float(np.nanmax(np.abs(c-d))),all_band_peak_shifts_samples=shifts,
                meaningful_bands=np.flatnonzero(meaningful),
                meaningful_peak_shift_samples=int(shifts[meaningful].max()) if meaningful.any() else 0))
        passed=all(r['position_error_m']<PROTOCOL['convergence_position_m']
                   and r['full_trace_position_error_m']<PROTOCOL['convergence_position_m']
                   and r['activation_error']<PROTOCOL['convergence_activation']
                   and r['score_error']<PROTOCOL['convergence_score'] and r['meaningful_peak_shift_samples']<=1 for r in rows)
        return dict(coarser=a,finer=b,passed=passed,rows=rows)
    replay(.01)
    grids=PROTOCOL['integration_grids'];replay(grids[0]);checks=[]
    for coarser,finer in zip(grids[:-1],grids[1:]):
        replay(finer);checks.append(compare(coarser,finer));chosen=finer
        if checks[-1]['passed']:
            chosen=coarser;break
    unresolved=not checks[-1]['passed']
    write(root/'integration.json',dict(physical_dt=chosen,unresolved=unresolved,comparisons=checks,
        solver=PROTOCOL['integration_method'],dtype=PROTOCOL['integration_dtype'],
        reference_dt=checks[-1]['finer'],
        historical_comparison=compare(.01,chosen),source_trials=anchors,
        scope='All A--E source trajectories; identical initial joints, velocities, activations and commands; synchronized algebraic force/geometry; full 15.3s position and SF tails',
        weak_change='All band shifts reported; timing tolerance applies to raw cosine changes >=0.01 in both integrations',
        historical_scope='Original coarse physics reproduces exactly; fine-grid convergence does not assert original event survival'))
    if unresolved:raise RuntimeError('Integration did not converge at prescribed tolerances; inspect integration.json before confirmation')


def stage_calibrate(args, root):
    if not (root/'audit.json').exists():
        raise ValueError('Run audit before calibration')
    policies = models()
    if not (root/'integration.json').exists() or read(root/'integration.json')['unresolved']:
        integration_check(args,root,policies)
    dt = read(root/'integration.json')['physical_dt']
    if not (root/'feedback_calibration.json').exists():
        rows=[]
        for kp,kd in track(limited(args,[(p,d) for p in (5.,15.,45.) for d in (.5,1.5,4.5)]), description='Calibrate independent feedback controller'):
            successes=[]; times=[]
            for target in limited(args,list(range(8))):
                task,obs=make_task((target,)*2000,dt,[[43.,88.],[47.,92.]])
                data=record(task,obs,'pd',policies,(kp,kd),180)
                successes.extend(data['first_hit']>=0)
                times.extend(np.where(data['first_hit']>=0,data['first_hit']*.01,1.8))
            rows.append(dict(gains=[kp,kd],success=float(np.mean(successes)),restricted_time_s=float(np.mean(times))))
        best=min(rows,key=lambda v:(-v['success'],v['restricted_time_s'],v['gains']))
        write(root/'feedback_calibration.json',dict(selected=best,all_gains=rows,smoke=args.limit is not None))
    gains=read(root/'feedback_calibration.json')['selected']['gains']
    plans=episode_plans(args,'calibration',read(root/'panel.json'))
    collect_plans(args,root,'calibration',plans,dt,policies,gains)
    if not (root/'single_discount.json').exists():
        audit=read(root/'audit.json'); progress=defaultdict(list)
        contexts=[p for p in plans if p['controller']=='base' and p['task']=='isolated']
        heads=[g for g,library in enumerate(audit['single_libraries']) if len(library)==5]
        for p in track(contexts,description='Select strongest single discount on calibration only'):
            data=load_episode(root/'calibration'/(p['id']+'.npz'))
            indices=stall_indices(data);k=indices[0] if indices else 30
            for start in track(range(0,len(heads),args.batch_size),description='Paired single-discount branch batches'):
                batch=heads[start:start+args.batch_size]
                cases=[(str(g),audit['single_libraries'][g],False) for g in batch]
                outcomes=branch_many(data,k,dt,policies,gains,cases,measure_probes=False)
                for g,result in zip(batch,outcomes):progress[g].append(result['progress_500ms_m'])
        results=[dict(head=g,gamma=float(math.discount_grid()[g]),cardinality=len(library),
                      progress_m=float(np.mean(progress[g])) if progress[g] else None)
                 for g,library in enumerate(audit['single_libraries'])]
        eligible=[r for r in results if r['cardinality']==5 and r['progress_m'] is not None]
        if not eligible:
            raise ValueError('No five-point single-discount library available; report coverage rather than fabricate points')
        selected=max(eligible,key=lambda r:(r['progress_m'],-r['head']))
        write(root/'single_discount.json',dict(selected=selected,all_heads=results,
              library=audit['single_libraries'][selected['head']],calibration_contexts=[p['id'] for p in contexts]))
    write(root/'calibration_complete.json',dict(physical_dt=dt,gains=gains,
          solver=PROTOCOL['integration_method'],smoke=args.limit is not None))


def configuration(root):
    if not (root/'calibration_complete.json').exists():
        raise ValueError('Run calibration before this stage')
    return read(root/'calibration_complete.json')


def stage_collect(args,root):
    cfg=configuration(root)
    plans=episode_plans(args,'confirmation',read(root/'panel.json'))
    collect_plans(args,root,'confirmation',plans,cfg['physical_dt'],models(),cfg['gains'])


def stage_mechanism(args,root):
    cfg=configuration(root); policies=models(); audit=read(root/'audit.json')
    mean,std=stats(args)
    with np.load(root/'references.npz') as z:
        refs=[z[f'band_{i}'].copy() for i in range(39)]
    multi_library=audit['original_library']; single=read(root/'single_discount.json')['library']
    directory=root/'mechanism'; directory.mkdir(exist_ok=True)
    for plan in track(read(root/'confirmation/index.json'),description='Measure predictive transitions and cloned-state recovery'):
        path=directory/(plan['id']+'.json')
        if path.exists():
            continue
        data=load_episode(root/'confirmation'/(plan['id']+'.npz'))
        multi,heads=math.scores(data['phi'],mean,std)
        pct=math.percentiles(multi,refs)
        # Confirmatory events use the original reference, independently of option success.
        event_stop=min(800,len(data['phi'])-1)
        detected=[]
        goals=array(make_task(physical_dt=cfg['physical_dt'])[0].targets)
        identities=np.argmax(data['observation'][:,2:10],axis=1)
        for visit,target in sorted(set(zip(data['visit'][:event_stop+1].tolist(),identities[:event_stop+1].tolist()))):
            mask=(data['visit'][1:event_stop+1]==visit)&(identities[1:event_stop+1]==target)
            masked=pct[:,:event_stop].copy();masked[:,~mask]=np.nan
            detected.extend(math.events(masked,data['phi'],event_stop,goals,target,plan['trial']))
        for event in detected:
            event['raw_change']=float(multi[event['peak_band'],event['step']-1])
            event['nearest_ae_m']=float(min(np.linalg.norm(np.asarray(event['xy_m'])-c['xy_m']) for c in multi_library))
        end=min(180,len(data['phi'])-1)
        # All predeclared A--E regions are queried, irrespective of outcome or score.
        selected=[]
        named_libraries=[dict(candidate,method='multiscale') for candidate in multi_library]+[
            dict(candidate,id=f'single_{i+1}',method='single') for i,candidate in enumerate(single)]
        for candidate in named_libraries:
            distance=np.linalg.norm(data['phi'][:end+1,:2]-candidate['xy_m'],axis=1)
            k=int(np.argmin(distance))
            if distance[k] <= .02:
                selected.append((k,candidate['id'],candidate['method']))
        probe_rows=[]; coverage=[]
        d=np.linalg.norm(data['phi'][:end+1,:2]-data['goal'][:end+1],axis=1)
        speed=np.linalg.norm(data['phi'][:end+1,2:4],axis=1)
        near=np.min(np.linalg.norm(data['phi'][:end+1,None,:2]-np.asarray([v['xy_m'] for v in named_libraries])[None],axis=-1),axis=1)<=.02
        probe_cache={}
        # Every candidate/match and temporal neighbor is known before outcomes are probed.
        pairs=[]
        for k,name,method in selected:
            target=np.argmax(data['observation'][k,2:10])
            candidates=np.flatnonzero((~near)&(abs(d-d[k])<=.02)&(abs(speed-speed[k])<=.01)
                  &(abs(np.arange(end+1)-k)/max(end,1)<=.15)
                  &(~data['finished'][:end+1].astype(bool))
                  &(np.argmax(data['observation'][:end+1,2:10],axis=1)==target))
            matched=int(candidates[np.argmin(abs(candidates-k))]) if len(candidates) else None
            pairs.append((k,name,method,matched))
        needed=sorted({neighbor for k,_,_,matched in pairs for step in ([k] if matched is None else [k,matched])
                       for neighbor in (step,max(0,step-3),min(len(data['phi'])-1,step+3))})
        for start in range(0,len(needed),args.batch_size):
            steps=needed[start:start+args.batch_size]
            effector=copy.deepcopy(make_task(physical_dt=cfg['physical_dt'])[0].effector)
            effector.states={key[6:]:torch.tensor(data[key][steps],dtype=effector.states['joint'].dtype) for key in data if key.startswith('state_')}
            probe_cache.update(zip(steps,pulse_batch(effector,data['goal'][steps])))
        def cached_probe(step):
            if step not in probe_cache:
                task,_=restore(data,step,cfg['physical_dt'])
                probe_cache[step]=pulse_probe(task,data['goal'][step])
            return probe_cache[step]
        for k,name,method,matched in pairs:
            coverage.append(dict(candidate=name,method=method,step=k,matched_step=matched))
            for step,kind in [(k,'candidate')]+([(matched,'matched')] if matched is not None else []):
                task,_=restore(data,step,cfg['physical_dt'])
                probes=cached_probe(step)
                left,right=max(0,step-3),min(len(data['phi'])-1,step+3)
                before=cached_probe(left)[-1]['response_matrix']
                after=cached_probe(right)[-1]['response_matrix']
                reorganization=float(np.linalg.norm(after-before)/((right-left)*.01))
                muscle=array(task.states['muscle'])[0]; geo=array(task.states['geometry'])[0]
                torque=-geo[2:]*muscle[task.effector.force_index][None]
                unsigned=float(abs(torque).sum())
                clearance=float(np.rad2deg(np.minimum(array(task.states['joint'])[0,:2]-array(task.effector.pos_lower_bound),array(task.effector.pos_upper_bound)-array(task.states['joint'])[0,:2])).min())
                probe_rows.append(dict(candidate=name,method=method,kind=kind,step=step,probes=probes,
                  response_reorganization_m_per_s=reorganization,
                  features=data['phi'][step],joint=data['state_joint'][step],muscle=muscle,
                  muscle_names=task.effector.muscle_name,torque=torque.sum(axis=1),
                  cancellation=1-float(abs(torque.sum(axis=1)).sum())/unsigned if unsigned>1e-12 else None,
                  clearance_deg=clearance,multiscale_percentiles=pct[:,max(step-1,0)],
                  single_changes=heads[:,max(step-1,0)]))
        stalls=stall_indices(data)
        branches=[]; association=[]
        for k in stalls:
            cases=[('base',None,False),('direct',None,True),('multiscale',multi_library,False),('single',single,False)]
            branches.extend(dict(step=k,**result) for result in branch_many(data,k,cfg['physical_dt'],policies,cfg['gains'],cases))
            history=data['phi'][max(0,k-10):k+1,:2]
            near_ae=bool(np.min(np.linalg.norm(history[:,None]-np.asarray([v['xy_m'] for v in multi_library])[None],axis=-1))<=.02)
            def near_event(step):
                return any(abs(e['step']-step)<=10 and np.linalg.norm(data['phi'][step,:2]-e['xy_m'])<=.02 for e in detected)
            near_stall=near_event(k)
            controls=[]
            for other in range(10,min(801,len(data['phi']))):
                if other in stalls or np.argmax(data['observation'][other,2:10])!=np.argmax(data['observation'][k,2:10]):
                    continue
                if abs(np.linalg.norm(data['phi'][other,:2]-data['goal'][other])-np.linalg.norm(data['phi'][k,:2]-data['goal'][k]))>.02:
                    continue
                if abs(np.linalg.norm(data['phi'][other,2:4])-np.linalg.norm(data['phi'][k,2:4]))>.01 or abs(other-k)>120:
                    continue
                # A matched nonstall must have positive progress in the same 100ms window.
                d_before=np.linalg.norm(data['phi'][other-10,:2]-data['goal'][other])
                d_after=np.linalg.norm(data['phi'][other,:2]-data['goal'][other])
                if d_before-d_after>=.001:
                    controls.append(other)
            other=min(controls,key=lambda v:abs(v-k)) if controls else None
            control_near=None if other is None else near_event(other)
            association.append(dict(step=k,boundary_near=near_stall,control_step=other,control_near=control_near,
                                    original_ae_region_near=near_ae))
        write(path,dict(metadata=plan,probe_rows=probe_rows,match_coverage=coverage,branches=branches,
                        association=association,stall_count=len(stalls),confirmation_events=detected,
                        reference_scope='original discovery frozen across controllers and tasks'))


def transfer_batch(task, obs, policies, gains, method, library, selector=None, steps=DEFAULTS["transfer_batch"]["steps"], controller=DEFAULTS["transfer_batch"]["controller"], observer=None, event_observer=None):
    """Per-row option clocks; logical task goals always remain the real targets."""
    n=len(obs)
    active=np.full(n,-1,dtype=int); remaining=np.zeros(n,dtype=int)
    acted=set(); histories=[]; option_steps=np.zeros(n,dtype=int)
    stalls=np.zeros(n,dtype=int); first_completion=np.full(n,-1,dtype=int)
    completed_targets=np.zeros(n,dtype=int)
    xy_library=np.asarray([v['xy_m'] for v in library]) if library else np.empty((0,2))
    for step in range(steps):
        xy=features(task)[:,:2]
        goal=array(task.goal); visit=array(task.sequence_index)
        finished=array(task.finished).astype(bool)
        histories.append((xy.copy(),goal.copy(),visit.copy()))
        if len(histories)>11:
            histories.pop(0)
        if method=='learned':
            ready=(remaining==0)&(~finished)
            if ready.any():
                with torch.no_grad():
                    q=selector(torch.from_numpy(obs[ready])).numpy()
                for values,row in zip(q,np.flatnonzero(ready)):
                    mask=np.r_[True,np.linalg.norm(xy_library-xy[row],axis=1)>.02]
                    choice=int(np.argmax(np.where(mask,values,-np.inf)))
                    if choice:
                        active[row]=choice-1
                        # Preserve the historical executor's effective timeout.
                        from SAC.options import TIMEOUT_STEPS
                        remaining[row]=min(int(library[choice-1].get('timeout_steps',200)),TIMEOUT_STEPS)
                        if event_observer is not None:
                            event_observer('start', step, row, choice, task, obs, None)
        elif method!='base' and len(histories)==11:
            for row in range(n):
                key=(row,int(visit[row]))
                if finished[row] or key in acted or remaining[row]:
                    continue
                first_xy,_,first_visit=histories[0]
                if first_visit[row]!=visit[row]:
                    continue
                distance=np.linalg.norm(goal[row]-xy[row])
                progress=np.linalg.norm(goal[row]-first_xy[row])-distance
                if distance>.04 and progress<.001:
                    acted.add(key); stalls[row]+=1
                    choice=nearest_option(xy[row],library) if library else None
                    active[row]=-1 if choice is None else choice
                    if method=='direct' or active[row]>=0:
                        remaining[row]=30
        excitation=commands(controller,task,obs,policies,gains)
        use=np.flatnonzero((remaining>0)&(~finished))
        if len(use):
            destinations=goal[use] if method=='direct' else xy_library[active[use]]
            action,_=policies[1].predict(np.column_stack((destinations,obs[use,10:])),deterministic=True)
            excitation[use]=(np.clip(action,-1,1)+1)/2
            option_steps[use]+=1
        obs,_,_,_,_=task.step(excitation,deterministic=True)
        obs=np.asarray(obs,dtype=np.float32)
        if observer is not None:
            executed=np.full(n,-1,dtype=int)
            executed[use]=active[use]
            observer(step+1,task,obs,executed)
        remaining=np.maximum(0,remaining-1)
        # Fixed-rule options terminate on original-target acquisition as well as waypoint entry.
        for row in use:
            waypoint_hit=method!='direct' and np.linalg.norm(features(task)[row,:2]-xy_library[active[row]])<=.02
            target_hit=int(task.sequence_index[row])!=int(visit[row]) or bool(task.finished[row])
            if waypoint_hit or (method!='learned' and target_hit):
                remaining[row]=0
            if event_observer is not None and remaining[row]==0:
                event_observer('end', step+1, row, int(active[row])+1, task, obs,
                               'waypoint' if waypoint_hit else 'target' if target_hit and method!='learned' else 'timeout')
        newly=array(task.finished).astype(bool)&(first_completion<0)
        first_completion[newly]=step+1
        completed_targets=np.maximum(completed_targets,array(task.sequence_index).astype(int)+array(task.finished).astype(int))
        if np.all(first_completion>=0):
            break
    return [dict(success=bool(k>=0),completion_s=float(k*.01) if k>=0 else None,
                 restricted_completion_s=float(k*.01) if k>=0 else steps*.01,
                 targets_completed=int(c),option_steps=int(o),stall_count=int(s),
                 command_budget=steps,method=method,controller=controller)
            for k,c,o,s in zip(first_completion,completed_targets,option_steps,stalls)]


def verified_selector():
    selector=load_checkpoint(SELECTOR,'cpu')
    if selector.config['state_dim']!=30 or selector.config['action_count']!=6:
        raise ValueError('Historical selector dimensions mismatch A--E library')
    expected=set(selector.config.get('artifact_hashes',{}).values())
    if not {sha(BASE),sha(REACHER),sha(LIBRARY)} <= expected:
        raise ValueError('Historical selector was trained with different artifacts')
    return selector


def stage_historical(args,root):
    stage_transfer(args,root,historical=True)


def stage_transfer(args,root,historical=DEFAULTS["stage_transfer"]["historical"]):
    cfg=dict(physical_dt=.01,gains=[15.,1.5]) if historical else configuration(root)
    policies=models(); selector=verified_selector()
    audit=read(root/'audit.json')
    libraries={'base':[],'learned':audit['original_library']}
    if not historical:
        libraries.update(direct=[],multiscale=audit['original_library'],single=read(root/'single_discount.json')['library'])
    panel=read(root/'panel.json')
    sequences=[v for values in panel['groups'].values() for v in values]
    positions=panel['postures_deg']
    directory=root/('historical_transfer' if historical else 'transfer');directory.mkdir(exist_ok=True)
    jobs=[(i,method,controller) for i in range(len(sequences)) for controller in (('base',) if historical else ('base','reacher','pd'))
          for method in libraries if method!='learned' or controller=='base']
    for order,method,controller in track(limited(args,jobs),description='Evaluate paired, frozen transfer panel'):
        path=directory/f'order_{order:02d}_{controller}_{method}.json'
        if path.exists():
            continue
        contexts=[]
        for seed_index,seed in enumerate(math.SEEDS):
            for posture in range(2):
                if len(positions)==10:
                    q=positions[seed_index*2+posture]
                else:
                    # Distinct prespecified replication panels when an older panel has only two starts.
                    rng=np.random.default_rng(np.random.SeedSequence([seed,posture,270019]))
                    origin=positions[order*2+posture] if len(positions)==2*len(sequences) else positions[posture]
                    q=(np.asarray(origin)+rng.uniform(-.5,.5,2)).tolist()
                contexts.append(dict(seed=seed,posture=posture,q=q,order=order))
        rows=[]
        for start in range(0,len(contexts),args.batch_size):
            batch=contexts[start:start+args.batch_size]
            task,obs=make_task(sequences[order],cfg['physical_dt'],[p['q'] for p in batch],max_steps=800)
            result=transfer_batch(task,obs,policies,cfg['gains'],method,libraries[method],selector,controller=controller)
            rows.extend([{**context,**outcome} for context,outcome in zip(batch,result)])
        write(path,dict(rows=rows,selector_hash=sha(SELECTOR) if method=='learned' else None,
              library_hash=sha(LIBRARY) if method in ('multiscale','learned') else None,
              physical_dt=cfg['physical_dt'],historical_discretization=historical,
              interpretation='frozen learned selection' if method=='learned' else 'common fixed-rule comparison'))


def stage_preflight(args,root):
    from .mechanism_checks import run_checks
    started=time.perf_counter()
    checks=run_checks(args.legacy_root)
    policies=models(); verified_selector()
    benchmark=[]
    for n in (1,8):
        task,obs=make_task((0,)*2000,postures=[[45.,90.]]*n)
        start=time.perf_counter()
        record(task,obs,'base',policies,(15.,1.5),20)
        seconds=time.perf_counter()-start
        benchmark.append(dict(batch_size=n,commands=20,seconds=seconds,
                              seconds_per_trajectory_command=seconds/(20*n)))
    write(root/'preflight.json',dict(checks=checks,python=platform.python_version(),
          torch=torch.__version__,motornet=getattr(mn,'__version__','unknown'),
          cuda=torch.cuda.is_available(),benchmark=benchmark,elapsed_s=time.perf_counter()-started))


def stage_report(args,root):
    evidence=dict(protocol=read(root/'protocol.json'),smoke=args.limit is not None,
                  completed_stages=[],transfer={},mechanism={},neural=None,human=None,
                  integration=read(root/'integration.json') if (root/'integration.json').exists() else None)
    for stage,path in [('audit','audit.json'),('calibration','calibration_complete.json'),
                       ('neural','neural/summary.json'),('human','human/summary.json')]:
        if (root/path).exists():
            evidence['completed_stages'].append(stage)
            if stage in ('neural','human'):
                evidence[stage]=read(root/path)
    transfer=[]
    for path in sorted((root/'transfer').glob('*.json')):
        transfer.extend(read(path)['rows'])
    for method in ('base','direct','multiscale','single','learned'):
        rows=[v for v in transfer if v['method']==method and v['controller']=='base']
        per_seed=[np.mean([r['success'] for r in rows if r['seed']==seed])
                  for seed in math.SEEDS if any(r['seed']==seed for r in rows)]
        evidence['transfer'][method]=dict(n=len(rows),completion=math.interval(per_seed))
    by_key={(r['method'],r['seed'],r['order'],r['posture']):r for r in transfer if r['controller']=='base'}
    contrasts={}
    for alternative in ('base','direct','single'):
        values=[]
        for seed in math.SEEDS:
            keys=[(r['order'],r['posture']) for r in transfer if r['controller']=='base' and r['method']=='multiscale' and r['seed']==seed
                  and (alternative,seed,r['order'],r['posture']) in by_key]
            if keys:
                values.append(np.mean([int(by_key[('multiscale',seed,*k)]['success'])-int(by_key[(alternative,seed,*k)]['success']) for k in keys]))
        contrasts[alternative]=math.interval(values)
    evidence['transfer']['paired_completion_differences']=contrasts
    historical=[r for p in sorted((root/'historical_transfer').glob('*.json')) for r in read(p)['rows']]
    evidence['historical_transfer']={}
    for method in ('base','learned'):
        values=[np.mean([r['success'] for r in historical if r['seed']==seed and r['method']==method])
                for seed in math.SEEDS if any(r['seed']==seed and r['method']==method for r in historical)]
        evidence['historical_transfer'][method]=math.interval(values)
    mechanisms=[read(p) for p in sorted((root/'mechanism').glob('*.json'))]
    for controller in ('base','reacher','pd'):
        for task_name in ('isolated','sequence','redirect'):
            rows=[v for v in mechanisms if v['metadata']['controller']==controller and v['metadata']['task']==task_name]
            values=[];coverage=0
            for seed in math.SEEDS:
                differences=[]
                for row in rows:
                    if row['metadata']['seed']!=seed:
                        continue
                    probes={(p['candidate'],p['kind']):p for p in row['probe_rows'] if p['method']=='multiscale'}
                    for key,p in probes.items():
                        other=probes.get((key[0],'matched'))
                        if key[1]=='candidate' and other is not None:
                            differences.append(p['probes'][-1]['directional_gain_m']-other['probes'][-1]['directional_gain_m'])
                if differences:
                    values.append(np.mean(differences));coverage+=len(differences)
            evidence['mechanism'][controller+'_'+task_name]=dict(matched_pairs=coverage,capacity_difference=math.interval(values))
            advantages=[]
            for seed in math.SEEDS:
                per_method=defaultdict(list)
                for row in rows:
                    if row['metadata']['seed']!=seed:
                        continue
                    paired={(p['candidate'],p['kind']):p for p in row['probe_rows']}
                    for (name,kind),p in paired.items():
                        match=paired.get((name,'matched'))
                        if kind=='candidate' and match is not None:
                            per_method[p['method']].append(p['response_reorganization_m_per_s']-match['response_reorganization_m_per_s'])
                if per_method['multiscale'] and per_method['single']:
                    advantages.append(np.mean(per_method['multiscale'])-np.mean(per_method['single']))
            evidence['mechanism'][controller+'_'+task_name]['multiscale_minus_single_reorganization']=math.interval(advantages)
    association=[]; rescue={}
    for seed in math.SEEDS:
        seed_rows=[r for r in mechanisms if r['metadata']['seed']==seed]
        pairs=[a for row in seed_rows for a in row['association'] if a['control_near'] is not None]
        if pairs:
            association.append(np.mean([int(p['boundary_near'])-int(p['control_near']) for p in pairs]))
    evidence['mechanism']['stall_association']=math.interval(association)
    for alternative in ('base','direct','single'):
        values=[];capacity=[];progress_values=[];latency_values=[]
        for seed in math.SEEDS:
            differences=[];gains=[];progress=[];latency=[]
            for row in mechanisms:
                if row['metadata']['seed']!=seed:
                    continue
                paired={(p['step'],p['method']):p for p in row['branches']}
                for (step,method),p in paired.items():
                    other=paired.get((step,alternative))
                    if method=='multiscale' and other is not None:
                        differences.append(int(p['acquired'])-int(other['acquired']))
                        if p['progress_500ms_m'] is not None and other['progress_500ms_m'] is not None:
                            progress.append(p['progress_500ms_m']-other['progress_500ms_m'])
                        latency.append(p['restricted_acquisition_s']-other['restricted_acquisition_s'])
                        if p['post_intervention_probes'] and other['post_intervention_probes']:
                            gains.append(p['post_intervention_probes'][-1]['directional_gain_m']-other['post_intervention_probes'][-1]['directional_gain_m'])
            if differences:
                values.append(np.mean(differences))
            if gains:
                capacity.append(np.mean(gains))
            if progress:progress_values.append(np.mean(progress))
            if latency:latency_values.append(np.mean(latency))
        rescue[alternative]=dict(acquisition_difference=math.interval(values),post_option_capacity_difference=math.interval(capacity),
                                  progress_500ms_difference_m=math.interval(progress_values),restricted_latency_difference_s=math.interval(latency_values))
    evidence['mechanism']['paired_rescue']=rescue
    evidence['transfer']['by_controller']={}
    for controller in ('base','reacher','pd'):
        evidence['transfer']['by_controller'][controller]={}
        for method in ('base','direct','single','multiscale'):
            rows=[r for r in transfer if r['controller']==controller and r['method']==method]
            evidence['transfer']['by_controller'][controller][method]=dict(n=len(rows),completion=math.interval([
                np.mean([r['success'] for r in rows if r['seed']==seed]) for seed in math.SEEDS if any(r['seed']==seed for r in rows)]))
    panel=read(root/'panel.json')
    expected_transfer=sum(map(len,panel['groups'].values()))*13
    expected_ids={p['id'] for p in episode_plans(args,'confirmation',panel)}
    collected=read(root/'confirmation/index.json') if (root/'confirmation/index.json').exists() else []
    calibration=read(root/'calibration_complete.json') if (root/'calibration_complete.json').exists() else {}
    evidence['simulation_complete']=(args.limit is None and len(list((root/'transfer').glob('*.json')))==expected_transfer
        and {p['id'] for p in collected}==expected_ids
        and all((root/'confirmation'/(name+'.npz')).exists() for name in expected_ids)
        and {p.stem for p in (root/'mechanism').glob('*.json')}==expected_ids
        and bool(evidence['integration']) and not evidence['integration']['unresolved']
        and bool(calibration) and not calibration.get('smoke') and not calibration.get('fixture_only'))
    evidence['confirmatory_complete']=bool(evidence['simulation_complete'] and evidence['neural'] and evidence['neural'].get('complete'))
    evidence['human_complete']=bool(evidence['human'] and evidence['human'].get('complete'))
    evidence['interpretation']=dict(body_constraint='Inspect physical probes and cloned-state contrasts; option success alone is insufficient',
        controller_generality='Three specified controller conditions, with training lineage disclosed',
        no_emg='EMG and neural activity were not recorded',
        historical_discovery='Exact Monte Carlo; learned predictor is separate evidence',
        statistics='Paired seed means, 95% t intervals; five seeds do not guarantee power')
    write(root/'evidence.json',evidence)
    # Reporting is allowed before completion, but figures show counts and no significance decoration.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figures.plotting import apply_style
    figure_directory=FIGURES/root.name
    apply_style();figure_directory.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.5,3.2),layout='constrained')
    for i,method in enumerate(('base','direct','single','multiscale','learned')):
        values=evidence['transfer'][method]['completion'].get('replication_values',[])
        if values:
            ax.scatter(np.full(len(values),i),values,s=18,label=method)
            ax.plot([i-.15,i+.15],[np.mean(values)]*2,color='black')
    ax.set(xticks=range(5),xticklabels=['Base','Direct','Single gamma','A--E fixed rule','A--E learned'],
           ylabel='Sequence completion fraction',ylim=(-.03,1.03),
           title='Five evaluation panels; frozen controllers' if evidence['confirmatory_complete'] else 'Incomplete evidence — counts in evidence.json')
    fig.savefig(figure_directory/'transfer.png',dpi=600);plt.close(fig)
    write(root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*')
         if p.is_file() and p.name!='artifact_hashes.json' and not p.name.endswith('.tmp')})
    print(f'Evidence: {root / "evidence.json"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=['preflight','audit','historical','calibrate','collect','mechanism','transfer','neural','human','report']
    parser.add_argument('--stage',required=True,choices=stages+['all'])
    parser.add_argument('--output',type=Path,default=ROOT/'results/mechanism')
    parser.add_argument('--legacy-root',type=Path,default=LEGACY)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--limit',type=int,help='Smoke-only limit; creates a separate, non-confirmatory protocol')
    args=parser.parse_args()
    if args.batch_size<1 or (args.limit is not None and args.limit<1):
        parser.error('Batch size and limit must be positive')
    torch.set_num_threads(1)
    root=manifest(args)
    requested=stages if args.stage=='all' else [args.stage]
    for stage in track(requested,description='Experiment stages'):
        print(f'Starting stage: {stage}',flush=True)
        if stage in ('neural','human'):
            from . import mechanism_validation
            getattr(mechanism_validation,'stage_'+stage)(args,root)
        else:
            globals()['stage_'+stage](args,root)


if __name__=='__main__':
    main()
