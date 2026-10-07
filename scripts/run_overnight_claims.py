"""Operator-launched, time-bounded evidence batch. Retain negative and partial results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def worker(args):
    from scripts import evaluate_landmark_handoffs as h
    from rich.progress import track
    from SAC import high_level
    import numpy as np
    h.torch.set_num_threads(1)
    output = args.output
    if args.worker in ('human_controls', 'sequence_discovery'):
        profile_checks(output, args.worker, args.smoke)
        return
    if args.worker == 'physiology':
        from scripts import run_five_claim_pilots as pilot
        # Reuse the intervention implementation; expanded panel is recorded in our protocol.
        pilot.PANEL = output/'replay'
        pilot.physiology(output/'state_interventions', args.smoke)
        return
    if args.worker == 'summary':
        summarize(output)
        return
    name, seed_text = args.worker.split(':')
    seed = int(seed_text)
    library_path = output/'libraries'/f'{name}.json'
    run = output/'trained'/f'{name}_{seed}'
    if not (run/'final.pt').exists():
        if run.exists():
            raise RuntimeError(f'Partial training retained: {run}. Use a new output directory to retrain; optimizer/replay cannot resume.')
        high_level.train(run, library_path, h.e.REACHER, base_checkpoint=h.e.BASE,
                         timesteps=1200 if args.smoke else 60000, seed=seed,
                         device='cpu', free_for_all=True, transfer_training=False,
                         deliberation_cost=.1)
    if not args.smoke:
        assert read(run/'summary.json')['updates'] > 0, 'No learning updates: do not interpret this fit'
    selector = high_level.load_checkpoint(run/'final.pt', 'cpu')
    library = read(library_path)['options']
    panel = read(ROOT/'data/mechanism/panel.json')
    sequences = [s for group in panel['groups'].values() for s in group]
    policies = h.e.models()
    for order in track(range(1 if args.smoke else len(sequences)), description=f'{name}, seed {seed}: paired orders'):
        rng = np.random.default_rng(20261007 + order)
        contexts = [dict(q=(np.array(panel['postures_deg'][order*2+i%2]) + rng.uniform(-.5, .5, 2)).tolist(),
                         context=i, rng_seed=20261007+order)
                    for i in range(1 if args.smoke else 4)]
        for method in ['base', 'learned']:
            path = output/'evaluation'/('base' if method == 'base' else f'{name}_{seed}')/f'order_{order:02d}.json'
            if path.exists():
                continue
            task, obs = h.e.make_task(sequences[order], .01, [c['q'] for c in contexts], 800)
            rows = h.e.transfer_batch(task, obs, policies, (15., 1.5), method, library,
                                      selector if method == 'learned' else None, 800)
            write(path, dict(order=order, sequence=sequences[order], contexts=contexts,
                             method=method, library=name if method == 'learned' else None,
                             selector_seed=seed if method == 'learned' else None, results=rows,
                             final_checkpoint_sha256=sha(run/'final.pt') if method == 'learned' else None))


def summarize(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'))
    plt.rcParams.update({'font.family': 'EB Garamond', 'font.size': 11})
    rows = []
    for folder in sorted((output/'evaluation').glob('*')):
        if folder.name == 'base':
            continue
        pairs = []
        for path in sorted(folder.glob('*.json')):
            base_path = output/'evaluation/base'/path.name
            if not base_path.exists():
                continue
            trial, base = read(path), read(base_path)
            assert trial['contexts'] == base['contexts']
            pairs.append(dict(order=trial['order'], n=len(trial['results']),
                              success_effect=np.mean([int(a['success'])-int(b['success']) for a,b in zip(trial['results'],base['results'])]),
                              time_effect_s=np.mean([a['restricted_completion_s']-b['restricted_completion_s'] for a,b in zip(trial['results'],base['results'])]),
                              target_effect=np.mean([a['targets_completed']-b['targets_completed'] for a,b in zip(trial['results'],base['results'])])))
        if not pairs:
            continue
        effects = np.array([p['success_effect'] for p in pairs])
        ci = np.quantile(np.random.default_rng(42).choice(effects, (10000,len(effects))).mean(1), [.025,.975])
        rows.append(dict(fit=folder.name, orders=len(pairs), trials=sum(p['n'] for p in pairs),
                         complete_panel=len(pairs)==24, success_effect=float(effects.mean()),
                         order_bootstrap_95CI=ci.tolist(), per_order=pairs))
    write(output/'SUMMARY.json', dict(scope='Exploratory; final checkpoints; all fits and failures retained. Training seeds are not base-policy replications.',
                                     effects=rows, source_hashes={str(p.relative_to(output)):sha(p) for p in (output/'evaluation').rglob('*.json')}))
    if rows:
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for i,row in enumerate(rows):
            mean = 100*row['success_effect']; low,high = 100*np.array(row['order_bootstrap_95CI'])
            axes[0].errorbar(mean, i, xerr=[[max(0,mean-low)],[max(0,high-mean)]], fmt='o', color='black',capsize=3)
            axes[1].scatter([100*p['success_effect'] for p in row['per_order']], [i]*row['orders'],alpha=.35,s=20)
        for ax in axes:
            ax.axvline(0,color='black',lw=.8)
            ax.set(yticks=range(len(rows)),yticklabels=[f"{r['fit']} ({r['orders']}/24 orders)" for r in rows],xlabel='Completion difference vs base (percentage points)')
        axes[0].set_title('Mean paired effect; order-bootstrap 95% intervals')
        axes[1].set_title('Every evaluated order; four paired starts per order')
        fig.tight_layout()
        target=output/'plots/library_completion.png';target.parent.mkdir(exist_ok=True)
        fig.savefig(target,dpi=600,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,key,label in [(axes[0],'time_effect_s','Failure-capped time difference vs base (s)'),
                             (axes[1],'target_effect','Targets acquired difference vs base')]:
            for i,row in enumerate(rows):
                values=np.array([p[key] for p in row['per_order']]);mean=values.mean()
                low,high=np.quantile(np.random.default_rng(42).choice(values,(10000,len(values))).mean(1),[.025,.975])
                ax.errorbar(mean,i,xerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o',capsize=3,color='black')
            ax.axvline(0,color='black',lw=.8)
            ax.set(yticks=range(len(rows)),yticklabels=[f"{r['fit']} ({r['orders']}/24)" for r in rows],xlabel=label)
        fig.suptitle('Secondary outcomes: every fit; order-bootstrap 95% intervals')
        fig.tight_layout();fig.savefig(target.parent/'library_secondary_outcomes.png',dpi=600,facecolor='white');plt.close(fig)
        write(output/'plots/FIGURE_MANIFEST.json',dict(script_sha256=sha(Path(__file__)),png_sha256=sha(target),source='SUMMARY.json',source_sha256=sha(output/'SUMMARY.json')))
    plot_dir=output/'plots';plot_dir.mkdir(exist_ok=True)
    human_path=output/'human_controls.json'
    if human_path.exists():
        completed=[r for r in read(human_path)['results'] if r['status']=='complete']
        if completed:
            fig,ax=plt.subplots(figsize=(10,4))
            for i,r in enumerate(completed):
                mean=r['mean_correlation_advantage'];low,high=r['participant_bootstrap_95CI']
                ax.errorbar(i,mean,yerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o',capsize=3,color='black')
            ax.axhline(0,color='black',lw=.8)
            ax.set(xticks=range(len(completed)),xticklabels=[f"{r['candidate']} {r['passage']}\nn={r['participants']}" for r in completed],
                   ylabel='Correlation advantage vs matched controls',title='All estimable candidate/passage contrasts; participant-bootstrap 95% intervals')
            fig.tight_layout();fig.savefig(plot_dir/'human_profile_controls.png',dpi=600,facecolor='white');plt.close(fig)
    sequence_path=output/'alternate_sequence/RESULTS.json'
    if sequence_path.exists():
        comparisons=read(sequence_path)['comparisons']
        eligible=[r for r in comparisons if r['null_motor_distances']]
        if eligible:
            fig,ax=plt.subplots(figsize=(10,4))
            ax.boxplot([r['null_motor_distances'] for r in eligible],positions=range(len(eligible)),showfliers=True)
            ax.scatter(range(len(eligible)),[r['observed_motor_distance'] for r in eligible],marker='D',color='#B56A38',label='Discovered candidate')
            ax.set(xticks=range(len(eligible)),xticklabels=[f"S{i+1}\n{len(r['null_motor_distances'])} controls" for i,r in enumerate(eligible)],
                   ylabel='Nearest A–E motor-feature distance (standardized units)',title='Policy 1001: velocity and activation; phase/target-matched controls')
            ax.legend();fig.tight_layout();fig.savefig(plot_dir/'alternate_sequence_motor_controls.png',dpi=600,facecolor='white');plt.close(fig)
    interventions=[read(p) for p in (output/'state_interventions/physiology').glob('*.json')]
    completed=[r for r in interventions if r.get('status')=='complete']
    if completed:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for index,ax in enumerate(axes,1):
            for option in range(1,6):
                selected=[r for r in completed if r['metadata']['choice']==option]
                values=[1000*(r['progress_500ms_m'][0]-r['progress_500ms_m'][index]) for r in selected]
                ax.scatter([option]*len(values),values,s=15,alpha=.5,color='black')
            ax.axhline(0,color='black',lw=.8)
            ax.set(xticks=range(1,6),xticklabels=list('ABCDE'),ylabel='Sham minus replacement progress at 500 ms (mm)',
                   title='Activation replacement' if index==1 else 'Velocity replacement')
        fig.suptitle(f'Every completed state intervention; {len(completed)} invocation states; active feedback retained')
        fig.tight_layout();fig.savefig(plot_dir/'state_intervention_sensitivity.png',dpi=600,facecolor='white');plt.close(fig)
    write(plot_dir/'FIGURE_MANIFEST.json',dict(script_sha256=sha(Path(__file__)),
          figures={p.name:sha(p) for p in plot_dir.glob('*.png')},
          source_hashes={str(p.relative_to(output)):sha(p) for p in [output/'SUMMARY.json',human_path,sequence_path] if p.exists()},
          intervention_sources={str(p.relative_to(output)):sha(p) for p in (output/'state_interventions/physiology').glob('*.json')}))


def profile_checks(output, stage, smoke):
    from scripts import evaluate_landmark_handoffs as h
    import formulas as math
    from types import SimpleNamespace
    from rich.progress import track
    import numpy as np
    audit=read(ROOT/'data/mechanism/audit.json')
    paths=h.e.sources(SimpleNamespace(legacy_root=h.e.LEGACY))
    cache=h.e.load_episode(paths['reaches'])
    mean,std=h.e.stats(SimpleNamespace(legacy_root=h.e.LEGACY))
    source=audit['source_discovery_trials']
    task,_=h.e.make_task(physical_dt=.01);goals=h.e.array(task.targets)
    raw={}
    # Frozen original source cohort defines references; no new outcome determines cutoffs.
    for i in track(source,description=f'{stage}: frozen reference profiles'):
        phi=cache['phi'][i,:,:4] if stage=='human_controls' else cache['phi'][i]
        raw[i]=math.scores(phi,mean[:4] if stage=='human_controls' else mean,std[:4] if stage=='human_controls' else std)[0]
    refs=math.reference(np.concatenate([raw[i][:,:int(cache['hit'][i])] for i in source],axis=1))
    if stage=='human_controls':
        humans=read(ROOT/'data/mechanism/human/trial_signatures.json')
        profiles={i:math.percentiles(raw[i],refs) for i in source}
        results=[]
        for candidate in track(audit['candidates'],description='A–E vs phase/norm-matched non-landmarks'):
            profile=profiles[candidate['trial']][:,candidate['step']-1]
            controls=[];control_ids=[]
            for i in source:
                if int(cache['target'][i])!=int(candidate['target']):continue
                hit=int(cache['hit'][i])
                for step in range(1,hit):
                    xy=cache['phi'][i,step,:2]
                    if abs(step/hit-candidate['phase'])>.1 or np.linalg.norm(xy-goals,axis=1).min()<=.025:continue
                    if any(np.linalg.norm(xy-np.array(c['xy_m']))<=.02 for c in audit['candidates']):continue
                    control=profiles[i][:,step-1]
                    if not np.array_equal(np.isfinite(control),np.isfinite(profile)):continue
                    norm=np.linalg.norm(profile[np.isfinite(profile)])
                    if norm==0 or not .75<=np.linalg.norm(control[np.isfinite(control)])/norm<=1.25:continue
                    controls.append(control)
                    control_ids.append((i,step))
            if not controls:
                results.append(dict(candidate=candidate['id'],eligible_controls=0,status='unestimable'));continue
            controls=np.array(controls)
            h.e.save_arrays(output/'human_control_profiles'/f"{candidate['id']}.npz",candidate=profile,controls=controls,trial_step=np.array(control_ids))
            for passage in ['first','second']:
                people={}
                for trial in humans:
                    if trial.get('signature') is None:continue
                    human=np.array([np.nan if v is None else v for v in trial['signature'][passage]])
                    valid=np.isfinite(profile)&np.isfinite(human)
                    support=np.zeros(len(profile),dtype=bool);support[trial['signature']['valid_bands']]=True
                    valid &= support
                    if valid.sum()<3 or np.std(profile[valid])==0 or np.std(human[valid])==0:continue
                    eligible=np.std(controls[:,valid],axis=1)>0
                    if not eligible.any():continue
                    centered=controls[eligible][:,valid]-controls[eligible][:,valid].mean(axis=1,keepdims=True)
                    vector=human[valid]-human[valid].mean()
                    control_r=(centered@vector)/(np.linalg.norm(centered,axis=1)*np.linalg.norm(vector))
                    ae_r=np.corrcoef(profile[valid],human[valid])[0,1]
                    people.setdefault(str(trial['participant']),[]).append(float(ae_r-control_r.mean()))
                differences=np.array([np.mean(v) for v in people.values()])
                if not len(differences):
                    results.append(dict(candidate=candidate['id'],passage=passage,eligible_controls=len(controls),status='unestimable'));continue
                ci=np.quantile(np.random.default_rng(42).choice(differences,(10000,len(differences))).mean(1),[.025,.975])
                results.append(dict(candidate=candidate['id'],passage=passage,eligible_controls=len(controls),participants=len(people),
                                    participant_effects={p:float(np.mean(v)) for p,v in people.items()},eligible_trials={p:len(v) for p,v in people.items()},
                                    mean_correlation_advantage=float(differences.mean()),participant_bootstrap_95CI=ci.tolist(),status='complete'))
        write(output/'human_controls.json',dict(results=results,scope='Exploratory motor prediction-profile selectivity; all five candidates and both passages. Not human hierarchical planning.',
                                               matching='Same target, pre-acquisition phase ±0.1, identical finite-band support, profile norm ±25%, >25mm from targets, >20mm from A–E; every eligible control retained'))
        return
    alternate=ROOT/'checkpoints/alternate_policy/best_model.zip'
    policies=(h.SAC.load(str(alternate),device='cpu'),h.e.models()[1])
    policies[0].policy.set_training_mode(False)
    fresh=[]
    for target in track(range(1 if smoke else 8),description='Hash-linked policy-1001 isolated reaches'):
        rng=np.random.default_rng(20261007+target)
        postures=(np.array([45.,90.])+rng.uniform(-3,3,(1 if smoke else 3,2))).tolist()
        path=output/'alternate_sequence'/f'target{target}.npz'
        if not path.exists():
            task,obs=h.e.make_task((target,)*2000,.01,postures,1600)
            data=h.e.record(task,obs,'base',policies,(15.,1.5),1530)
            h.e.save_arrays(path,**data)
            write(path.with_suffix('.json'),dict(checkpoint_sha256=sha(alternate),target=target,postures_deg=postures,
                                               protocol='Fresh isolated-target assay with recorded post-acquisition tail; not a recovered historical discovery cache'))
        data=h.e.load_episode(path)
        for row,hit in enumerate(data['first_hit']):
            fresh.append(dict(target=target,hit=int(hit),phi=data['phi'][:,row],file=str(path),row=row))
    candidates=[]
    for trial,reach in enumerate(fresh):
        if reach['hit']<=0:continue
        score=math.scores(reach['phi'],mean,std)[0]
        for candidate in math.events(math.percentiles(score,refs),reach['phi'],reach['hit'],goals,reach['target'],trial):
            if candidate['nearest_target_m']>.025 and not any(math.same_event(candidate,old) for old in candidates):candidates.append(candidate)
    canonical=np.stack([(cache['phi'][c['trial'],c['step'],2:]-mean[2:])/std[2:] for c in audit['candidates']])
    comparisons=[]
    for candidate in candidates:
        reach=fresh[candidate['trial']]
        observed=(reach['phi'][candidate['step'],2:]-mean[2:])/std[2:]
        pool=[]
        for other in fresh:
            if other['target']!=candidate['target'] or other['hit']<=0:continue
            for step in range(1,other['hit']):
                if abs(step/other['hit']-candidate['phase'])<=.1 and np.linalg.norm(other['phi'][step,:2]-goals,axis=1).min()>.025:
                    pool.append((other['phi'][step,2:]-mean[2:])/std[2:])
        observed_distance=float(np.linalg.norm(canonical-observed,axis=1).min())
        null=np.linalg.norm(np.array(pool)[:,None]-canonical,axis=2).min(axis=1) if pool else np.array([])
        comparisons.append(dict(candidate=candidate,observed_motor_distance=observed_distance,eligible_phase_target_controls=len(pool),
                                null_motor_distances=null.tolist(),descriptive_lower_tail=float((1+sum(null<=observed_distance))/(1+len(null))) if len(null) else None))
    write(output/'alternate_sequence/RESULTS.json',dict(checkpoint_sha256=sha(alternate),reaches=len(fresh),failed_reaches=sum(r['hit']<=0 for r in fresh),
             comparisons=comparisons,scope='Fresh hash-linked alternate-trained-policy assay; standardized velocity and six muscle activations, excluding position. Descriptive phase/target-matched chance comparison; not proof of sequence-invariant identity.'))


def main(args):
    from rich.progress import Progress
    output=args.output
    sources = [Path(__file__), ROOT/'SAC/high_level.py', ROOT/'scripts/screen_landmark_evidence.py',
               ROOT/'scripts/run_five_claim_pilots.py', ROOT/'scripts/evaluate_landmark_handoffs.py',
               ROOT/'analysis/mechanism_experiment.py', ROOT/'data/mechanism/panel.json',
               ROOT/'checkpoints/base_policy/best_model.zip']
    # Resolve authoritative model/library paths rather than guessing checkpoint names.
    from scripts import evaluate_landmark_handoffs as h
    sources = [p for p in sources if p.exists()] + [h.e.BASE,h.e.REACHER,h.e.LIBRARY,*h.POLICIES.values()]
    paths = {'multiscale':h.e.LIBRARY,
             'spatial':ROOT/'data/discovery/controls/spatial_grid_options.json',
             'kinematic':ROOT/'data/discovery/controls/kinematic_extrema_options.json',
             'single':ROOT/'data/single_discount/single_discount.json'}
    sources += list(paths.values())
    sources += [ROOT/'checkpoints/alternate_policy/best_model.zip', ROOT/'checkpoints/alternate_policy/config.json',
                ROOT/'data/mechanism/human/trial_signatures.json',ROOT/'data/mechanism/audit.json',
                ROOT/'formulas.py']
    from types import SimpleNamespace
    legacy_paths=h.e.sources(SimpleNamespace(legacy_root=h.e.LEGACY))
    docs = list((ROOT/'docs').glob('*.md'))
    protocol = dict(version=1,smoke=args.smoke,training_steps=1200 if args.smoke else 60000,
                    selector_seeds=[7] if args.smoke else [7,101,202],libraries=list(paths),
                    warmup_decisions=1000,checkpoint='final.pt only',base_policy_seed=7,
                    evaluation_orders=1 if args.smoke else 24,contexts_per_order=1 if args.smoke else 4,
                    context_seed=20261007,dynamics='Euler 10ms; 800 commands, 8 seconds',
                    primary='full-sequence completion',secondary=['failure-capped completion time','targets acquired'],
                    inference='Exploratory order-bootstrap intervals; no hypothesis-test claims; report each fit separately',
                    replay_per_cell=1 if args.smoke else 6,
                    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
                    legacy_source_hashes={str(legacy_paths[key]):sha(legacy_paths[key]) for key in ['reaches','normalization']},
                    profile_control_matching='Target, phase ±0.1, finite support and norm ±25%; no relaxed criteria',
                    alternate_sequence_assay='24 fresh isolated reaches, recorded 15.3s tails, original frozen references; velocity and activation chance screen',
                    claim_document_hashes={str(p.relative_to(ROOT)):sha(p) for p in docs if p.exists()})
    protocol_path=output/'PROTOCOL.json'
    if protocol_path.exists():
        assert read(protocol_path)==protocol, 'Protocol or source changed; use a new output directory'
    else:
        write(protocol_path,protocol)
        for p in docs:
            if p.exists():
                dest=output/'claim_documents'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(p.read_bytes())
    for name,path in paths.items():
        library=read(path)
        if name=='single':
            library={'options':library['library']}
        assert len(library['options'])==5, f'{name}: unequal cardinality'
        library.update(timeout_steps=30,termination_radius_m=.02)
        for choice,option in enumerate(library['options'],1):
            option.update(choice=choice,timeout_steps=30,termination_radius_m=.02)
        write(output/'libraries'/f'{name}.json',library)
    common=[sys.executable,'-u',str(Path(__file__)), '--output',str(output)] + (['--smoke'] if args.smoke else [])
    screen=[sys.executable,'-u',str(ROOT/'scripts/screen_landmark_evidence.py'),'--replay','--output',str(output/'replay'),'--per-cell',str(protocol['replay_per_cell'])]
    if args.smoke:
        screen += ['--pilot']
    jobs=[('human_controls',common+['--worker','human_controls']),('sequence_discovery',common+['--worker','sequence_discovery']),
          ('replay',screen),('physiology',common+['--worker','physiology'])]
    jobs += [(f'{name}_{seed}',common+['--worker',f'{name}:{seed}']) for seed in protocol['selector_seeds'] for name in paths]
    assert len({name for name,_ in jobs})==len(jobs)
    status_path=output/'STATUS.json'; status=read(status_path) if status_path.exists() else {}
    for name,_ in jobs:status.setdefault(name,dict(state='pending'))
    write(status_path,status)
    deadline=time.monotonic()+args.hours*3600
    with Progress() as progress:
        task=progress.add_task('Overnight evidence jobs',total=len(jobs))
        for name,cmd in jobs:
            if status.get(name,{}).get('state')=='complete':
                progress.advance(task);continue
            remaining=deadline-time.monotonic()
            if remaining<=30:
                break
            print(f'\nStarting {name}; {remaining/3600:.2f} hours remain. Log: {output / "logs" / (name+".log")}',flush=True)
            log=output/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
            started=time.time();status[name]=dict(state='running',started=started,command=cmd);write(status_path,status)
            # Child Rich bars go to a durable log; parent refreshes elapsed progress each second.
            with log.open('a',encoding='utf-8') as stream:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
                while child.poll() is None and time.monotonic()<deadline:
                    progress.update(task,description=f'{name}: {(time.time()-started)/60:.1f} min elapsed')
                    time.sleep(1)
                if child.poll() is None:
                    child.terminate();child.wait();state='time_limit_partial'
                else:
                    state='complete' if child.returncode==0 else 'failed'
            status[name]=dict(state=state,seconds=time.time()-started,returncode=child.returncode,log=str(log));write(status_path,status)
            print(f'{name}: {state}',flush=True);progress.advance(task)
    summarize(output)
    print(f'Results retained: {output}\nRead STATUS.json for completed/failed/partial stages. No result was selected by sign.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=4.75)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.hours<=0:
        parser.error('--hours must be positive')
    worker(args) if args.worker else main(args)
