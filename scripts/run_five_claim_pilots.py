"""Small, fixed exploratory tests. Long stages are operator-launched; never select good outcomes."""

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("scripts/run_five_claim_pilots.py", {})
DEFAULTS = SETTINGS["defaults"].get("scripts/run_five_claim_pilots.py", {})
import argparse
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_landmark_handoffs as h
from rich.progress import track

DATA = ROOT/'data/discovery/controls'
PANEL = ROOT/'data/replay/selected_states'


def setup(output, smoke=DEFAULTS["setup"]["smoke"]):
    panel = h.e.read(ROOT/'data/mechanism/panel.json')
    sequences = [tuple(s) for group in panel['groups'].values() for s in group]
    orders = []
    offset = 0
    for group in panel['groups'].values():
        orders.append(next(offset+i for i, s in enumerate(group) if tuple(s) != h.e.SOURCE_SEQUENCE))
        offset += len(group)
    sources = [Path(__file__), ROOT/'scripts/evaluate_landmark_handoffs.py', ROOT/'analysis/mechanism_experiment.py',
               ROOT/'SAC/high_level.py', *h.POLICIES.values(), h.e.REACHER, h.e.SELECTOR, h.e.LIBRARY,
               DATA/'spatial_grid_options.json', DATA/'kinematic_extrema_options.json',
               ROOT/'data/human/published_models/utils.py', ROOT/'data/mechanism/human/experiment_sheet.svg']
    protocol = dict(version=1, smoke=smoke, scope='Exploratory pilots, not completed robustness/mediation/human-hierarchy validation',
                    orders=orders, contexts=2, policies=[7,101,202], dynamics=[.01,.001,.0005],
                    selector_training_steps=1200 if smoke else 6000, selector_seeds=[7], selector_choice='final.pt, no best-run selection',
                    selector_warmup_decisions=64, warmup_scope='Fixed short-pilot setting for all libraries; chosen after instrumentation showed no updates at original warmup',
                    physiology='10ms; fixed-position activation or velocity replacement using same-clock base donor; sham resynchronization in all arms',
                    human='Six SVG sequences; rigid 180deg rotation and translation only; geometric compatibility, separate final-hold/speed criteria',
                    hashes={str(p.relative_to(ROOT)):h.e.sha(p) for p in sources})
    if (output/'protocol.json').exists():
        assert h.e.read(output/'protocol.json') == protocol, 'Protocol changed: use a new output directory'
    h.e.write(output/'protocol.json', protocol)
    return panel, sequences, orders


def transfer(output, smoke=DEFAULTS["transfer"]["smoke"]):
    panel, sequences, orders = setup(output,smoke)
    policies = h.e.models()
    selector = h.e.verified_selector()
    library = h.e.read(h.e.LIBRARY)['options']
    jobs = [(seed, order, dt) for seed in h.POLICIES for order in orders for dt in (.01,.001,.0005)]
    for seed, order, dt in track(jobs[:1] if smoke else jobs, description='Full-sequence numerical pilot'):
        path = output/'transfer'/f'seed{seed}_order{order}_dt{dt}.json'
        if path.exists():
            continue
        base = policies[0] if seed == 7 else h.SAC.load(str(h.POLICIES[seed]), device='cpu')
        base.policy.set_training_mode(False)
        contexts = h.contexts(panel, order, 2)
        results = {}
        for method in ('base','learned'):
            task, obs = h.e.make_task(sequences[order], dt, [r['q'] for r in contexts], 800)
            results[method] = h.e.transfer_batch(task, obs, (base,policies[1]), (15.,1.5), method, library, selector, 800)
        h.e.write(path, dict(seed=seed, order=order, physical_dt=dt, contexts=contexts, results=results))


def physiology(output, smoke=DEFAULTS["physiology"]["smoke"]):
    setup(output,smoke)
    files = h.e.read(PANEL/'replay_protocol.json')['files']
    library = h.e.read(h.e.LIBRARY)['options']
    original, reacher = h.e.models()
    for name in track(files[:1] if smoke else files, description='Paired handoff-state interventions'):
        path = output/'physiology'/name
        if path.exists():
            continue
        source = ROOT/'data/handoffs/branches'/name
        row = h.e.read(source)
        base = original if row['metadata']['policy_seed'] == 7 else h.SAC.load(str(h.POLICIES[row['metadata']['policy_seed']]), device='cpu')
        base.policy.set_training_mode(False)
        state = h.e.load_episode(source.with_name(source.stem+'_start.npz'))
        task, obs = h.restore(state,.01)
        obs = h.repeat(task,obs,2)
        first = True
        saved = {}
        def selector(observation):
            nonlocal first
            q = h.torch.zeros((len(observation),6));q[:,0]=1
            if first:
                q[1,row['metadata']['choice']]=2;first=False
            return q
        def event(kind,step,index,choice,task,obs,reason):
            if kind == 'end' and index == 1 and not saved:
                saved.update(chosen=h.snapshot(task,obs,1), donor_activation=h.e.array(task.states['muscle'])[0,0],
                             donor_velocity=h.e.array(task.states['joint'])[0,2:], reason=reason)
        h.e.transfer_batch(task,obs,(base,reacher),(15.,1.5),'learned',library,selector,30,event_observer=event)
        if not saved:
            h.e.write(path, dict(metadata=row['metadata'], status='missing termination within 30 steps'))
            continue
        state = saved['chosen']
        if 800-int(state['elapsed_step']) < 50:
            h.e.write(path, dict(metadata=row['metadata'], status='less than 500ms original budget remaining'))
            continue
        task, obs = h.restore(state,.01)
        obs = h.repeat(task,obs,3)
        joint = task.states['joint'].clone()
        joint[2,2:] = h.torch.as_tensor(saved['donor_velocity'])
        activation = task.states['muscle'][:,:1].clone()
        activation[1,0] = h.torch.as_tensor(saved['donor_activation'])
        geometry = task.effector.get_geometry(joint)
        muscle = task.effector.muscle.integrate(0.,h.torch.zeros_like(activation),activation,geometry)
        task.effector._set_state(dict(joint=joint,muscle=muscle,geometry=geometry))
        np.testing.assert_allclose(h.e.array(task.states['cartesian'])[:,:2], np.repeat(state['state_cartesian'][:,:2],3,axis=0),atol=2e-6)
        np.testing.assert_allclose(h.e.array(task.states['joint'])[1],state['state_joint'][0],atol=0)
        assert np.isfinite(h.e.features(task)).all()
        # Instantaneous plant intervention; past delayed sensory samples are retained.
        # Current activation is explicitly supplied to the policy, so this is plant-plus-feedback sensitivity.
        obs[:,-6:] = h.e.array(activation[:,0]).astype(np.float32)
        goal = h.e.array(task.goal)[0]
        visit = int(task.sequence_index[0])
        distance = np.linalg.norm(h.e.features(task)[0,:2]-goal)
        acquired = np.zeros(3,bool)
        progress = None
        def observe(step,task,obs,executed):
            nonlocal progress
            acquired[:] |= h.e.array(task.finished).astype(bool)|(h.e.array(task.sequence_index)!=visit)
            if step == 50:
                progress = np.where(acquired,distance,distance-np.linalg.norm(h.e.features(task)[:,:2]-goal,axis=1))
        results = h.e.transfer_batch(task,obs,(base,reacher),(15.,1.5),'base',library,steps=50,observer=observe)
        if progress is None:
            assert all(r['success'] for r in results)
            progress = np.full(3,distance)
        h.e.write(path, dict(metadata=row['metadata'], status='complete', condition_names=['sham','base_activation','base_velocity'],
                             progress_500ms_m=progress.tolist(), outcomes=results, reason=saved['reason'],
                             donor_clock='Same elapsed time after identical invocation start',
                             interpretation='State-intervention sensitivity; not pure physiological mediation'))


def selectors(output, smoke=DEFAULTS["selectors"]["smoke"]):
    panel, sequences, orders = setup(output,smoke)
    from SAC import high_level
    high_level.WARMUP = 64
    train, load_checkpoint = high_level.train, high_level.load_checkpoint
    paths = dict(multiscale=h.e.LIBRARY, spatial=DATA/'spatial_grid_options.json', kinematic=DATA/'kinematic_extrema_options.json')
    for name, source in paths.items():
        library = h.e.read(source)
        assert len(library['options']) == 5
        library['timeout_steps']=30;library['termination_radius_m']=.02
        for option in library['options']:
            option.update(timeout_steps=30,termination_radius_m=.02)
        path = output/'libraries'/f'{name}.json'
        h.e.write(path,library)
        run = output/'selectors'/name
        if not (run/'final.pt').exists():
            if run.exists():
                raise RuntimeError(f'Incomplete selector training in {run}; retain it and use a new output directory')
            train(run,path,h.e.REACHER,base_checkpoint=h.e.BASE,timesteps=1200 if smoke else 6000,
                  seed=7,device='cpu',free_for_all=True,transfer_training=False,deliberation_cost=.1)
        assert h.e.read(run/'summary.json')['updates'] > 0, f'{name}: no learning updates; retain failed training'
        if smoke:
            break
    if smoke:
        return
    policies = h.e.models()
    for name in track(paths,description='Equal-budget final-selector evaluation'):
        selector = load_checkpoint(output/'selectors'/name/'final.pt','cpu')
        library = h.e.read(output/'libraries'/f'{name}.json')['options']
        for order in orders:
            path = output/'selector_evaluation'/f'{name}_order{order}.json'
            if path.exists():
                continue
            contexts = h.contexts(panel,order,2)
            task, obs = h.e.make_task(sequences[order],.01,[r['q'] for r in contexts],800)
            rows = h.e.transfer_batch(task,obs,policies,(15.,1.5),'learned',library,selector,800)
            h.e.write(path,dict(library=name,order=order,results=rows,contexts=contexts))


def sequence(output, smoke=DEFAULTS["sequence"]["smoke"]):
    setup(output,smoke)
    discovery = h.e.read(DATA/'seed_1001_discovery.json')
    cache = h.e.load_episode(DATA/'seed_1001_reaches.npz')
    canonical = np.array([o['xy_m'] for o in h.e.read(h.e.LIBRARY)['options']])
    candidates = discovery['candidates']
    task, _ = h.e.make_task()
    goals = h.e.array(task.targets)
    pools = []
    for candidate in candidates:
        pool = []
        for index,target in enumerate(cache['target']):
            if int(target) != int(candidate['target']):
                continue
            xy = cache['phi'][index,:int(cache['hit'][index])+1,:2]
            valid = np.linalg.norm(xy[:,None]-goals,axis=-1).min(axis=1)>.025
            pool.extend(xy[valid])
        assert pool, 'No target-matched eligible source states'
        pools.append(np.array(pool))
    rng = np.random.default_rng(62026)
    null = np.stack([pool[rng.integers(len(pool),size=9999)] for pool in pools],axis=1)
    distance = np.linalg.norm(null[:,:,None,:]-canonical,axis=-1).min(axis=2)*1000
    observed = np.linalg.norm(np.array([c['xy_m'] for c in candidates])[:,None]-canonical,axis=-1).min(axis=1)*1000
    h.e.write(output/'sequence.json',dict(observed_nearest_distances_mm=observed, null_mean_nearest_distance_mm=distance.mean(axis=1),
               observed_mean_mm=observed.mean(), randomization_lower_tail=(1+sum(distance.mean(axis=1)<=observed.mean()))/10000,
               observed_matches_20mm=int(sum(observed<=20)),null_matches_20mm=(distance<=20).sum(axis=1),
               provenance='Historical cache has no checkpoint hash. Cache-only diagnostic; cannot certify policy-1001 origin or sequence-invariant identity.',
               source_hashes={str(p.relative_to(ROOT)):h.e.sha(p) for p in [DATA/'seed_1001_reaches.npz',DATA/'seed_1001_discovery.json',h.e.LIBRARY]}))


def human(output, smoke=DEFAULTS["human"]["smoke"]):
    setup(output,smoke)
    spec = importlib.util.spec_from_file_location('published_utils',ROOT/'data/human/published_models/utils.py')
    utils = importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    basket = utils.coords_from_svg(filename=str(ROOT/'data/mechanism/human/experiment_sheet.svg'),reference='mk_tr')
    labels = ['pt_start','pt_left','pt_right','st_left','st_top','st_right']
    raw = np.array([basket[n].center for n in labels])/1000
    radii = np.array([basket[n].radius for n in labels])/1000
    assert np.allclose(radii[1:],[.055,.055,.035551197,.045,.035551197],atol=1e-8)
    reference,_ = h.e.make_task()
    xy = h.e.array(reference.states['cartesian'])[0,:2]-(raw-raw[0])
    ik = reference._reachable_joint_angles(h.torch.tensor(xy,dtype=h.torch.float32))
    # Current policies keep their target identities; this novel geometry is an OOD compatibility pilot.
    conditions = [(1,3),(1,5),(1,4),(2,3),(2,5),(2,4)]
    policies = h.e.models();selector=h.e.verified_selector();library=h.e.read(h.e.LIBRARY)['options']
    jobs = [(i,seed,method) for i in range(6) for seed in h.POLICIES for method in ('base','learned')]
    for condition, seed, method in track(jobs[:1] if smoke else jobs,description='Human-geometry compatibility pilot'):
        path = output/'human'/f'condition{condition}_seed{seed}_{method}.json'
        if path.exists():
            continue
        first,second = conditions[condition]
        task, obs = h.e.make_task((first,second,0),.01,max_steps=300)
        task.targets[:6] = h.torch.tensor(xy,dtype=h.torch.float32)
        task.target_joint_angles[:6] = ik
        task.goal = task.targets[first][None]
        obs[:,:2] = xy[first]
        # SequentialReachEnv permits a per-row hit radius; use the original SVG circle at each active goal.
        step = task.step
        def adapted_step(action, deterministic=True):
            task.hit_radius = float(radii[int(task._sequence[task.sequence_index[0]])])
            return step(action,deterministic=deterministic)
        task.step = adapted_step
        base = policies[0] if seed==7 else h.SAC.load(str(h.POLICIES[seed]),device='cpu')
        base.policy.set_training_mode(False)
        trace = [h.e.features(task)[0,:4]]
        def observe(k,task,obs,executed):
            trace.append(h.e.features(task)[0,:4])
        result = h.e.transfer_batch(task,obs,(base,policies[1]),(15.,1.5),method,library,selector,300,observer=observe)
        values = np.array(trace)
        h.e.save_arrays(path.with_suffix('.npz'),phi=values,targets=xy,radii=radii)
        h.e.write(path,dict(condition=condition,policy_seed=seed,method=method,results=result,
                             mean_speed_mps=float(np.linalg.norm(values[1:,2:],axis=1).mean()),
                             scope='Original physical SVG geometry and radii; 3s movement budget. One-second final hold and paper speed-success criterion not enforced; not human hierarchy.',
                             rigid_transform='180deg rotation; translate pt_start to original controller launch endpoint; no rescaling'))


def plots(output, smoke=DEFAULTS["plots"]["smoke"]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scripts.plot_landmark_evidence import OPTION_COLORS
    font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'))
    plt.rcParams.update({'font.family':'EB Garamond','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    dest=output/'plots';dest.mkdir(exist_ok=True)
    manifest=[]
    def save(fig,name,caption,sources):
        fig.subplots_adjust(bottom=.23,wspace=.4)
        fig.text(.02,.03,caption,fontsize=10)
        fig.patch.set_edgecolor('black');fig.patch.set_linewidth(72/600)
        fig.savefig(dest/f'{name}.png',dpi=600,facecolor='white');plt.close(fig)
        manifest.append(dict(figure=name,caption=caption,script_sha256=h.e.sha(Path(__file__)),
                             source_hashes={str(p.resolve().relative_to(ROOT)):h.e.sha(p) for p in sources}))
    files=list((output/'transfer').glob('*.json'))
    if files:
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,seed in zip(axes,h.POLICIES):
            for label,method in [('Base','base'),('DDQN','learned')]:
                rates=[]
                for dt in (.01,.001,.0005):
                    rows=[r for p in files for j in [h.e.read(p)] if j['seed']==seed and j['physical_dt']==dt for r in j['results'][method]]
                    rates.append(100*np.mean([r['success'] for r in rows]) if rows else np.nan)
                ax.plot(range(3),rates,'o-',label=label)
            ax.set(title=f'Policy {seed}',xticks=range(3),xticklabels=['Euler 10ms','RK4 1ms','RK4 0.5ms'],ylabel='Sequence completion (%)',ylim=(-5,105));ax.legend()
        save(fig,'full_sequence_dynamics',f'Exploratory fixed panel: {len(files)} policy/order/integrator jobs saved, two contexts per job; failures retained.\nNo population confidence intervals.',files)
    files=list((output/'physiology').glob('*.json'))
    rows=[h.e.read(p) for p in files if h.e.read(p)['status']=='complete']
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for index,(ax,condition) in enumerate(zip(axes,['Base activation at same clock','Base velocity at same clock']),1):
            for choice in range(1,6):
                values=[(r['progress_500ms_m'][0]-r['progress_500ms_m'][index])*1000 for r in rows if r['metadata']['choice']==choice]
                ax.scatter(np.linspace(choice-.09,choice+.09,len(values)),values,alpha=.7,color=OPTION_COLORS[choice-1])
            counts=[sum(r['metadata']['choice']==c for r in rows) for c in range(1,6)]
            ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(1,6),xticklabels=[f'{name}\nn={n}' for name,n in zip('ABCDE',counts)],ylabel='Intact minus perturbed progress (mm)',title=condition)
        save(fig,'handoff_state_interventions','Exploratory 10ms state interventions; fixed hand position, consistent force reconstruction, shared donor clock.\nPositive: intact state has greater 500ms progress. Feedback changes too; not pure mediation.',files)
    path=output/'sequence.json'
    if path.exists():
        d=h.e.read(path);fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].hist(d['null_mean_nearest_distance_mm'],bins=35,color='#90B4CC');axes[0].axvline(d['observed_mean_mm'],color='black',label='Cached candidates');axes[0].legend();axes[0].set(xlabel='Mean nearest canonical-landmark distance (mm)',ylabel='Random sets',title='Target-matched chance comparison')
        axes[1].bar(range(len(d['observed_nearest_distances_mm'])),d['observed_nearest_distances_mm'],color='#90B4CC');axes[1].axhline(20,color='black',ls='--');axes[1].set(xlabel='Cached alternate-sequence candidate',ylabel='Nearest canonical distance (mm)')
        save(fig,'sequence_chance_control','9,999 random sets from target-matched off-target pre-acquisition states.\nCache lacks checkpoint provenance; this cannot establish sequence-invariant identity.',[path])
    files=list((output/'selector_evaluation').glob('*.json'))
    if files:
        names=['multiscale','spatial','kinematic'];fig,axes=plt.subplots(1,2,figsize=(11,4))
        rates=[];counts=[];targets=[]
        for name in names:
            rows=[r for p in files for j in [h.e.read(p)] if j['library']==name for r in j['results']]
            assert rows, f'Missing selector evaluations: {name}'
            rates.append(100*np.mean([r['success'] for r in rows]))
            counts.append(len(rows));targets.append([r['targets_completed'] for r in rows])
        assert len(set(counts))==1, 'Incomplete paired selector panel'
        colors=['#28658B','#B56A38','#57774B']
        axes[0].bar(names,rates,color=colors);axes[0].set(ylabel='Sequence completion (%)',ylim=(0,105),title='Primary outcome: full completion')
        for index,(values,color) in enumerate(zip(targets,colors)):
            axes[1].scatter(np.linspace(index-.08,index+.08,len(values)),values,color=color,alpha=.7)
            axes[1].plot([index-.15,index+.15],[np.mean(values)]*2,color='black',lw=2)
        axes[1].set(xticks=range(3),xticklabels=names,ylabel='Targets acquired (of 8)',ylim=(-.1,2),title='Secondary outcome: partial progress')
        save(fig,'equal_budget_libraries',f'6,000 primitive steps per selector; 64-decision warmup, one initialization, five options; final checkpoints.\n{counts[0]} paired cases per library; dots: all cases, black bars: means. Short training does not establish library superiority.',files)
    files=list((output/'human').glob('*.json'))
    if files:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        rows=[h.e.read(p) for p in files]
        for offset,method,label in [(-.15,'base','Base'),(.15,'learned','DDQN')]:
            color='#28658B' if method=='base' else '#B56A38'
            for condition in range(6):
                selected=[r for r in rows if r['condition']==condition and r['method']==method]
                if selected:
                    axes[0].scatter(condition+offset,100*np.mean([r['results'][0]['success'] for r in selected]),color=color,label=label if condition==0 else None)
                    axes[1].scatter(condition+offset,np.mean([r['mean_speed_mps'] for r in selected]),color=color)
        axes[0].set(xlabel='Published sequence (0–5)',ylabel='Geometric completion (%)',ylim=(-5,105));axes[0].legend()
        axes[1].axhspan(.6,.9,color='gray',alpha=.2,label='Paper speed-success range');axes[1].set(xlabel='Published sequence (0–5)',ylabel='Mean movement speed (m/s)');axes[1].legend()
        save(fig,'human_geometry_compatibility',f'Exploratory original SVG scale/radii, rigid placement, 3s budget; {len(files)} condition/policy/controller jobs saved.\nFinal 1s hold and full joint hierarchy signatures not tested; geometric completion is not paper-defined success.',files)
    path=ROOT/'data/mechanism/human/ae_human_correlations.json'
    data=h.e.read(path)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,passage in zip(axes,['first','second']):
        for index,name in enumerate('ABCDE'):
            values=np.array([r[passage+'_mean_correlation'] for r in data if r['candidate']==name and r[passage+'_mean_correlation'] is not None])
            assert len(values)==20
            draws=np.random.default_rng(42).choice(values,size=(10000,len(values)))
            low,high=np.quantile(draws.mean(axis=1),[.025,.975])
            mean=values.mean()
            ax.errorbar(index,mean,yerr=[[mean-low],[high-mean]],fmt='o',capsize=4,color=OPTION_COLORS[index])
        ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(5),xticklabels=list('ABCDE'),ylim=(-1,1),ylabel='Mean participant profile correlation',title=f'{passage.capitalize()} target passage')
    save(fig,'human_prediction_profile_correspondence','20 participants; first passage: primary target, second: secondary target. All A–E/passage comparisons shown.\nExploratory participant-bootstrap 95% intervals; predictive-profile correspondence, not DDQN human hierarchy.',[path])
    h.e.write(dest/'FIGURE_MANIFEST.json',manifest)
    print(f'{len(manifest)} exploratory PNGs generated; every plotted source retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','transfer','physiology','selectors','sequence','human','plots'],default='all')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/pilots')
    args=parser.parse_args();args.output=args.output.resolve();h.torch.set_num_threads(1);started=time.perf_counter()
    for stage in (['sequence','transfer','physiology','selectors','human','plots'] if args.stage=='all' else [args.stage]):
        print(f'Stage: {stage}',flush=True)
        globals()[stage](args.output,args.smoke)
    h.e.write(args.output/f'runtime_{args.stage}.json',dict(seconds=time.perf_counter()-started,smoke=args.smoke))
