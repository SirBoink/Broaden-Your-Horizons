"""Learned SF replication and optional human kinematic compatibility analysis.

Neither stage changes the original A--E library or trains a motor controller.
"""
from __future__ import annotations

from config import PARAMETERS as SETTINGS
PARAMS = SETTINGS["modules"].get("analysis/mechanism_validation.py", {})
DEFAULTS = SETTINGS["defaults"].get("analysis/mechanism_validation.py", {})

import copy
import csv
import importlib.util
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rich.progress import track
from stable_baselines3 import SAC

NEURAL = SETTINGS["neural"]

import formulas as math
from formulas import ridge_fits, ridge_predict, principal_features
from . import mechanism_experiment as experiment


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        width, expanded = NEURAL["hidden_size"], NEURAL["expanded_size"]
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, expanded), nn.SiLU(), nn.Linear(expanded, width))

    def forward(self, x):
        return x + self.block(x)


class SuccessorNetwork(nn.Module):
    """Same architecture and state-dict names as the existing strong 23-head model."""
    def __init__(self, saved):
        super().__init__()
        for key in ('input_mean','input_std','target_mean','target_std','gammas'):
            self.register_buffer(key, torch.as_tensor(saved[key], dtype=torch.float32).clone())
        if self.gammas.shape != (NEURAL["head_count"],) or torch.any(self.target_std <= 0) or torch.any(self.input_std <= 0):
            raise ValueError('Invalid learned SF grid or normalization')
        self.input_layer = nn.Sequential(nn.Linear(NEURAL["input_size"], NEURAL["hidden_size"]), nn.SiLU())
        self.res_blocks = nn.Sequential(*(ResBlock() for _ in range(NEURAL["residual_blocks"])))
        self.head = nn.Linear(NEURAL["hidden_size"], NEURAL["head_count"] * NEURAL["feature_count"])

    def normalized(self, x):
        return self.head(self.res_blocks(self.input_layer((x-self.input_mean)/self.input_std))).reshape(-1, NEURAL["head_count"], NEURAL["feature_count"])

    def forward(self, x):
        return self.normalized(x)*self.target_std+self.target_mean


def load_network(path, device=DEFAULTS["load_network"]["device"]):
    saved = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    model = SuccessorNetwork(saved)
    model.load_state_dict(saved)
    return model.to(device).eval()


def save_torch(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary); temporary.replace(path)


def neural_sources(args):
    old = args.legacy_root
    return dict(model=old/'data/neural/initial_model/best.pt',
                config=old/'data/neural/initial_model/config.json',
                split=old/'config/neural_split.json',
                normalization=old/'data/neural/normalization.npz',
                data_manifest=old/'data/neural/manifest.json',
                policy=old/'checkpoints/neural_collection_policy.zip')


def neural_data(args, ids, gammas, normalization, description):
    """Read existing episodes; compute exact targets in memory, never write legacy caches."""
    fields = {k:[] for k in ('z','next_z','phi','done','returns','displacement')}
    source = args.legacy_root/'data/neural/episodes'
    index = {int(v['episode_id']):v for v in experiment.read(neural_sources(args)['data_manifest'])['files']}
    for identifier in track(ids, description=description):
        path = source/f'episode_{identifier:05d}.npz'
        if experiment.sha(path) != index[identifier]['sha256']:
            raise ValueError(f'Neural training episode hash mismatch: {identifier}')
        ep = experiment.load_episode(path)
        if not bool(ep['finished']):
            raise ValueError('Existing benchmark split must contain successful training episodes only')
        if np.any(ep['done'][:-1]) or not ep['done'][-1] or np.any(ep['truncated']):
            raise ValueError('Unexpected training episode terminal semantics')
        phi = (ep['feature'].astype(float)-normalization['phi_mean'])/normalization['phi_std']
        # The supervised return and TD target both use phi at the current step.
        fields['z'].append(np.column_stack((ep['state'],ep['action'])).astype(np.float32))
        fields['next_z'].append(np.column_stack((ep['next_state'],ep['next_action'])).astype(np.float32))
        fields['phi'].append(phi.astype(np.float32)); fields['done'].append(ep['done'].astype(np.float32))
        fields['returns'].append(math.returns(phi,gammas).transpose(1,0,2).astype(np.float32))
        displacement=np.full((len(phi),4),np.nan,dtype=np.float32)
        for j,lag in enumerate((15,30)):
            displacement[:-lag,2*j:2*j+2]=ep['feature'][lag:,:2]-ep['feature'][:-lag,:2]
        fields['displacement'].append(displacement)
    return {k:torch.from_numpy(np.concatenate(values)) for k,values in fields.items()}


@torch.no_grad()
def validation_loss(model, data, device):
    error, count = 0., 0
    for start in range(0,len(data['z']),NEURAL["prediction_batch"]):
        x=data['z'][start:start+NEURAL["prediction_batch"]].to(device); y=data['returns'][start:start+NEURAL["prediction_batch"]].to(device)
        error += float(((model(x)-y)/model.target_std).square().sum())
        count += y.numel()
    return error/count


def fit_network(args, directory, seed, saved, train, validation):
    """Epoch checkpoints include optimizer, EMA, scheduler and shuffle RNG for resumption."""
    if (directory/'complete.json').exists():
        return
    device=torch.device(args.device)
    experiment.write(directory/'training_config.json',dict(seed=seed,device=str(device),protocol=experiment.PROTOCOL,
        architecture='30 -> 256 -> 3 residual blocks (256 -> 512 -> 256) -> 23 x 10',
        initialization='Independent random parameters; reused original normalization buffers only'))
    torch.manual_seed(seed)
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
        torch.cuda.manual_seed_all(seed)
    model=SuccessorNetwork(saved).to(device)
    target=copy.deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=experiment.PROTOCOL["sf_lr"],weight_decay=NEURAL["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=NEURAL["scheduler_factor"],patience=NEURAL["scheduler_patience"],
              threshold=NEURAL["scheduler_threshold"],threshold_mode='abs',min_lr=NEURAL["minimum_learning_rate"])
    generator=torch.Generator().manual_seed(seed)
    start,best,stagnant,history=1,float('inf'),0,[]
    latest=directory/'latest.pt'
    if latest.exists():
        checkpoint=torch.load(latest,map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model']);target.load_state_dict(checkpoint['target'])
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        generator.set_state(checkpoint['rng'].cpu())
        start,best,stagnant,history=checkpoint['epoch']+1,checkpoint['best'],checkpoint['stagnant'],checkpoint['history']
    epochs=1 if args.limit is not None else experiment.PROTOCOL["sf_epochs"]
    for epoch in track(range(start,epochs+1),description=f'Fit SF seed {seed}'):
        if epoch>experiment.PROTOCOL["sf_min_epochs"] and stagnant>=experiment.PROTOCOL["sf_patience"]:
            break
        model.train()
        indices=torch.randperm(len(train['z']),generator=generator)
        for batch in track(indices.split(experiment.PROTOCOL["sf_batch"]),description=f'Seed {seed}, epoch {epoch}: MC + Bellman'):
            x=train['z'][batch].to(device); y=train['returns'][batch].to(device)
            prediction=model.normalized(x)
            with torch.no_grad():
                continuation=(1-train['done'][batch].to(device))[:,None,None]*model.gammas[None,:,None]*target(train['next_z'][batch].to(device))
                td=train['phi'][batch].to(device)[:,None,:]+continuation
            loss=((prediction-(y-model.target_mean)/model.target_std)**2).mean()+experiment.PROTOCOL["sf_bellman_weight"]*((prediction-(td-model.target_mean)/model.target_std)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite SF loss at seed {seed}, epoch {epoch}')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),NEURAL["gradient_clip"]);optimizer.step()
            with torch.no_grad():
                for a,b in zip(target.parameters(),model.parameters()):a.lerp_(b,NEURAL["target_ema"])
        value=validation_loss(model,validation,device);scheduler.step(value)
        if value<best-1e-4:
            best=value;stagnant=0
            save_torch(directory/'best.pt',dict(state_dict=model.state_dict(),epoch=epoch,val_loss=value))
        else:
            stagnant+=1
        history.append(dict(epoch=epoch,device=str(device),validation_nmse=value,lr=optimizer.param_groups[0]['lr']))
        save_torch(latest,dict(epoch=epoch,model=model.state_dict(),target=target.state_dict(),
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=generator.get_state(),
                   best=best,stagnant=stagnant,history=history))
    experiment.write(directory/'complete.json',dict(seed=seed,best_validation_nmse=best,history=history,
                       independently_initialized=True,smoke=args.limit is not None))


def fresh_neural_episodes(args, directory, policy_path):
    """Fresh matching-policy test, including failures; terminal futures are zero."""
    policy=SAC.load(str(policy_path),device='cpu');policy.policy.set_training_mode(False)
    plans=[(seed,trial) for seed in math.SEEDS for trial in range(16)]
    for seed,trial in track(experiment.limited(args,plans),description='Collect fresh learned-SF test episodes'):
        path=directory/f'fresh_{seed}_{trial}.npz'
        if path.exists():continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,trial,270021]))
        task,obs=experiment.make_task(physical_dt=.01,postures=[np.array([45.,90.])+rng.uniform(-5,5,2)],max_steps=800)
        rows=[]
        for step in range(800):
            action,_=policy.predict(obs,deterministic=True)
            excitation=np.clip((action+1)/2+rng.normal(0,.001,(1,6)),0,1).astype(np.float32)
            rows.append(dict(z=np.r_[experiment.physical_state(task)[0],2*excitation[0]-1],phi=experiment.features(task)[0]))
            obs,_,done,truncated,_=task.step(excitation,deterministic=True)
            obs=np.asarray(obs,dtype=np.float32)
            if done or truncated:break
        experiment.save_arrays(path,z=np.asarray([r['z'] for r in rows],dtype=np.float32),
                   phi=np.asarray([r['phi'] for r in rows]),seed=np.asarray(seed),trial=np.asarray(trial),
                   finished=np.asarray(bool(task.finished[0])),terminal=np.asarray('success' if done else 'timeout'))


@torch.no_grad()
def predictions(model, x):
    values=np.concatenate([model(batch).cpu().numpy() for batch in torch.as_tensor(x).float().split(NEURAL["prediction_batch"])])
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite learned SF predictions')
    return values











def stage_neural(args,root):
    directory=root/'neural';directory.mkdir(exist_ok=True)
    paths=neural_sources(args)
    provenance={k:experiment.sha(p) for k,p in paths.items()}
    config=experiment.read(paths['config']); source=experiment.read(paths['data_manifest'])
    if config['seed']!=42 or config['terminal_bootstrap'] or not config['use_bellman']:
        raise ValueError('Saved network training protocol differs from the replication')
    if experiment.sha(paths['policy'])!=source['checkpoint_sha256']:
        raise ValueError('Neural controller lineage mismatch')
    if (directory/'sources.json').exists() and experiment.read(directory/'sources.json')!=provenance:
        raise ValueError('Neural sources changed; use a new output directory')
    experiment.write(directory/'sources.json',provenance)
    saved=torch.load(paths['model'],map_location='cpu',weights_only=True)['state_dict']
    tau=np.geomspace(.01/-np.log(.5),.01/-np.log(.99),20)
    tau=np.r_[tau,tau[-1]*(tau[-1]/tau[-2])**np.arange(1,4)]
    gammas=np.exp(-.01/tau)
    np.testing.assert_allclose(saved['gammas'].numpy(),gammas,atol=3e-8,rtol=0)
    normalization=experiment.load_episode(paths['normalization'])
    split=experiment.read(paths['split'])
    if any(set(split[a])&set(split[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Training, validation and historical test episode IDs overlap')
    train_ids=split['train'] if args.limit is None else split['train'][:2]
    val_ids=split['validation'] if args.limit is None else split['validation'][:2]
    # Reuse the already completed seed-42 fit. Four new fits have independent initial weights.
    seeds=math.SEEDS if args.limit is None else (101,)
    needs_training=any(not (directory/f'seed_{s}/complete.json').exists() for s in seeds if s!=42)
    validation=neural_data(args,val_ids,gammas,normalization,'Load frozen SF calibration split')
    if needs_training:
        train=neural_data(args,train_ids,gammas,normalization,'Load frozen SF training split')
        for seed in seeds:
            if seed!=42:fit_network(args,directory/f'seed_{seed}',seed,saved,train,validation)
        del train
    fresh=directory/'fresh_physical_state';fresh.mkdir(exist_ok=True)
    fresh_neural_episodes(args,fresh,paths['policy'])
    valid=np.isfinite(validation['displacement'].numpy()).all(axis=1)
    calibration_y=validation['displacement'].numpy()[valid]
    calibration_x=validation['z'].numpy()[valid]
    # Episode split: entire first half of validation episodes fits readouts; second half selects.
    first_half=neural_data(args,val_ids[:max(1,len(val_ids)//2)],gammas,normalization,'Count readout training episode frames')
    boundary=int(np.isfinite(first_half['displacement'].numpy()).all(axis=1).sum());del first_half
    if not 0<boundary<len(calibration_y):raise ValueError('Need disjoint episodes for readout fitting and selection')
    results=[]
    fast=int(np.argmin(abs(-.01/np.log(gammas)-.05)));slow=int(np.argmin(abs(-.01/np.log(gammas)-.20)))
    for seed in track(seeds,description='Evaluate every SF fit on untouched fresh episodes'):
        model=load_network(paths['model'] if seed==42 else directory/f'seed_{seed}/best.pt')
        cal=predictions(model,calibration_x)
        multi,projection=principal_features((cal*(1-gammas[None,:,None])).reshape(len(cal),-1),boundary)
        features={'current':calibration_x,'multi':multi}
        features.update({f'single_{g}':cal[:,g] for g in range(23)})
        readouts={};scores={}
        for name,x in features.items():
            fits=ridge_fits(x[:boundary].astype(float),calibration_y[:boundary])
            losses=[float(np.mean((ridge_predict(f,x[boundary:])-calibration_y[boundary:])**2)) for f in fits]
            chosen=int(np.argmin(losses));readouts[name]=fits[chosen];scores[name]=losses[chosen]
        single=min(range(23),key=lambda g:(scores[f'single_{g}'],g))
        rows=[]
        for path in sorted(fresh.glob('*.npz')):
            ep=experiment.load_episode(path);n=len(ep['phi'])
            actual=math.returns((ep['phi']-normalization['phi_mean'])/normalization['phi_std'],gammas).transpose(1,0,2)
            pred=predictions(model,ep['z'])
            a=(1-gammas[slow])*actual[:,slow]-(1-gammas[fast])*actual[:,fast]
            p=(1-gammas[slow])*pred[:,slow]-(1-gammas[fast])*pred[:,fast]
            measured={}
            if n>30:
                y=np.column_stack([ep['phi'][lag:lag+n-30,:2]-ep['phi'][:n-30,:2] for lag in (15,30)])
                mean,scale,basis=projection
                multi=((pred*(1-gammas[None,:,None])).reshape(n,-1)-mean)/scale@basis
                for name,x in [('current',ep['z']),('multi',multi),('single',pred[:,single])]:
                    fitted=readouts[f'single_{single}' if name=='single' else name]
                    measured[name]=float(np.mean((ridge_predict(fitted,x[:n-30])-y)**2))
            rows.append(dict(evaluation_seed=int(ep['seed']),trial=int(ep['trial']),finished=bool(ep['finished']),
                 head_nmse=np.mean(((pred-actual)/model.target_std.numpy())**2,axis=(0,2)),
                 temporal_contrast_mse=float(np.mean((p-a)**2)),
                 zero_contrast_mse=float(np.mean(a**2)),
                 temporal_change_mse=float(np.mean((np.diff(p,axis=0)-np.diff(a,axis=0))**2)) if n>1 else None,
                 zero_change_mse=float(np.mean(np.diff(a,axis=0)**2)) if n>1 else None,
                 future_displacement_mse=measured))
        result=dict(training_seed=seed,reused_saved_fit=seed==42,model_hash=experiment.sha(paths['model'] if seed==42 else directory/f'seed_{seed}/best.pt'),
                    selected_single_head=single,selected_single_gamma=float(gammas[single]),readout_selection_scores=scores,
                    sf_readout_dimension=10,rows=rows)
        experiment.write(directory/f'evaluation_{seed}.json',result);results.append(result)
    contrasts={}
    for alternative in ('current','single'):
        per_fit=[]
        for result in results:
            values=[r['future_displacement_mse'][alternative]-r['future_displacement_mse']['multi'] for r in result['rows'] if r['future_displacement_mse']]
            if values:per_fit.append(float(np.mean(values)))
        contrasts[alternative]=math.interval(per_fit)
    experiment.write(directory/'summary.json',dict(complete=args.limit is None and len(results)==5,
       training_seeds=[r['training_seed'] for r in results],independent_new_fits=4 if args.limit is None else 1,
       fresh_episodes=len(list(fresh.glob('*.npz'))),future_displacement_advantage=contrasts,
       accuracy_by_fit=[dict(training_seed=r['training_seed'],
          success_episodes=sum(v['finished'] for v in r['rows']),timeout_episodes=sum(not v['finished'] for v in r['rows']),
          head_nmse=np.mean([v['head_nmse'] for v in r['rows']],axis=0),
          success_mean_nmse=float(np.mean([v['head_nmse'] for v in r['rows'] if v['finished']])) if any(v['finished'] for v in r['rows']) else None,
          timeout_mean_nmse=float(np.mean([v['head_nmse'] for v in r['rows'] if not v['finished']])) if any(not v['finished'] for v in r['rows']) else None) for r in results],
       gammas=gammas,fast_head=fast,slow_head=slow,
       semantics='SFs of normalized ten-dimensional phi, conditional on 24D physical state [q,dq,activation,goal,length,velocity] and applied action; episodic terminal zero',
       distinction='23-head learned predictors use the later final policy and original 10ms physics; A--E discovery uses exact 40-head Monte Carlo under the original best policy',
       readout_scope='All representations forecast identical 150/300ms hand displacements; multi-SF uses ten calibration-only principal components, single-SF uses ten features; ridge strength and strongest single head selected on old validation episodes, not fresh tests',
       replication_scope='Five training initializations share one frozen training dataset; five fresh evaluation panels share the same frozen motor policy',
       terminal_scope='Timeouts included as finite 800-command censored episodes; report success and timeout strata separately',smoke=args.limit is not None))


def author_models():
    """Load unmodified published model files without replacing installed packages permanently."""
    directory=experiment.ROOT/'data/human/published_models'
    names=('src','src.model','src.utils','src.model.single')
    previous={name:sys.modules.get(name) for name in names}
    modules={}
    try:
        package=types.ModuleType('src');package.__path__=[]
        model_package=types.ModuleType('src.model');model_package.__path__=[]
        sys.modules['src']=package;sys.modules['src.model']=model_package
        for name,file in [('src.utils','utils.py'),('src.model.single','single.py'),('mechanism_published_hierarchical','hierarchical.py')]:
            spec=importlib.util.spec_from_file_location(name,directory/file)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            if name in names:sys.modules[name]=module
            modules[file]=module
            if file=='utils.py':package.utils=module
            if file=='single.py':model_package.single=module
    finally:
        for name,value in previous.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return modules


def kinematics(xy):
    xy=np.asarray(xy,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():
        raise ValueError('Need at least three finite 100Hz hand samples')
    return np.column_stack((xy,np.gradient(xy,.01,axis=0)))


def passage_indices(xy,centers,radii,first,second):
    crossings=[];start=1
    for label in (first,second):
        hits=np.flatnonzero(np.linalg.norm(xy[start:]-centers[label],axis=1)<=radii[label])
        if not len(hits):return None
        step=int(start+hits[0]);crossings.append(step);start=step+1
    return crossings


def passage_signature(phi,passages,mean,std,refs,single_refs=None):
    multi,single=math.scores(phi,mean,std)
    percentile=math.percentiles(multi,refs)
    profiles=[]
    for step in passages:
        window=percentile[:,max(0,step-11):min(len(phi)-1,step+10)]
        finite=np.isfinite(window).all(axis=1)
        values=np.full(39,np.nan)
        values[finite]=window[finite].mean(axis=1)
        profiles.append(values)
    common=np.isfinite(profiles).all(axis=0)
    peaks=np.geomspace(.01,1.8,39)
    common &= peaks>=.05
    if common.sum()<3:return None
    delta=profiles[1]-profiles[0]
    single_effect=None
    if single_refs is not None:
        ranked=math.percentiles(single,single_refs)
        head_profiles=[]
        for step in passages:
            window=ranked[:,max(0,step-11):min(len(phi)-1,step+10)]
            finite=np.isfinite(window).all(axis=1);values=np.full(40,np.nan)
            values[finite]=window[finite].mean(axis=1);head_profiles.append(values)
        single_effect=head_profiles[1]-head_profiles[0]
    return dict(first=profiles[0],second=profiles[1],second_minus_first=delta,single_effect=single_effect,
                valid_bands=np.flatnonzero(common),mean_difference=float(delta[common].mean()))


def stage_human(args,root):
    directory=root/'human';directory.mkdir(exist_ok=True)
    archive=experiment.ROOT/'data/human/bids_data.zip'
    published=experiment.ROOT/'data/human/published_models'
    provenance=dict(dataset_sha256=experiment.sha(archive),code_doi='10.6084/m9.figshare.31169632',
                    model_hashes={n:experiment.sha(published/n) for n in ('single.py','hierarchical.py','utils.py')})
    if (directory/'sources.json').exists() and experiment.read(directory/'sources.json')!=provenance:
        raise ValueError('Human sources changed; use a new output directory')
    experiment.write(directory/'sources.json',provenance)
    models=author_models();utils=models['utils.py'];hierarchy=models['hierarchical.py']
    with zipfile.ZipFile(archive) as zipped:
        svg=directory/'experiment_sheet.svg'
        svg.write_bytes(zipped.read('bids/code/experiment_sheet.svg'))
    original_coords=utils.coords_from_svg
    utils.coords_from_svg=lambda **kwargs:original_coords(filename=str(svg),**kwargs)
    basket=utils.coords_from_svg(reference='mk_tr')
    labels=('pt_start','pt_left','pt_right','st_left','st_top','st_right')
    centers={n:np.asarray(basket[n].center)/1000 for n in labels}
    radii={n:float(basket[n].radius)/1000 for n in labels}
    conditions=[('pt_left','st_left'),('pt_left','st_right'),('pt_left','st_top'),
                ('pt_right','st_left'),('pt_right','st_right'),('pt_right','st_top')]
    simulations=[]
    # Published Figure-8 illustrative settings, fixed before analysing human SFs.
    # These are controller-family checks, not reproductions of the paper's full fitted parameter sweep.
    seeds=math.SEEDS if args.limit is None else math.SEEDS[:1]
    for seed in track(seeds,description='Published flat/HiSeq controller compatibility checks'):
        for family in ('flat','hierarchical'):
            pending=[i for i in range(6) if not (directory/f'{family}_{seed}_{i}.npz').exists()]
            if args.limit is not None:pending=pending[:1]
            if pending:
                if family=='flat':
                    agent=hierarchy.vpSOC(t_ends=1.6,h_scaling=[1,7,15],soc_pars={'add_noise':0,'r':2e-7})
                else:
                    # Published code updates its class-level defaults: give this instance a private copy.
                    defaults=copy.deepcopy(hierarchy.Simple.def_pars)
                    try:
                        hierarchy.Simple.def_pars=copy.deepcopy(defaults)
                        scales=np.array([[1,1]]+[[15,3,0]]*6+[[2,1]]*3,dtype=object)
                        agent=hierarchy.HiSeq(h_scaling=scales,tau23=3,monitor_sd=5,monitor_distance=30,
                             soc_pars={'add_noise':0,'r':5e-5},random_seed=seed)
                    finally:hierarchy.Simple.def_pars=defaults
                for condition in track(pending,description=f'{family}, seed {seed}: six conditions'):
                    agent.choose_seq(condition)
                    if family=='flat':
                        x0=agent.centers[0].copy();trace=agent.move(x_t=x0,ix_seq=condition,dyn_kwargs={'motor_noise':False,'obs_noise':False})[0]
                    else:
                        x0=agent.set_initial_conditions(agent.sequences[condition],background_noise=.2)
                        x0[:2]=agent.centers[0]
                        trace=agent.move(x_t=x0,t_ini=0,t_end=6.,id_stop='pt_start',dyn_kwargs={'motor_noise':False,'obs_noise':False})[0]
                    phi=kinematics(trace[:,:2]/1000)
                    # The HiSeq code appends c_t after a physical advance and duplicates its initial timestamp.
                    # Its advance is exactly dt; use the physical sample index rather than that label.
                    experiment.save_arrays(directory/f'{family}_{seed}_{condition}.npz',phi=phi,time_s=np.arange(len(phi))*.01)
            for condition in range(6):
                path=directory/f'{family}_{seed}_{condition}.npz'
                if path.exists():simulations.append(dict(family=family,seed=seed,condition=condition,phi=experiment.load_episode(path)['phi']))
    pooled=np.concatenate([s['phi'] for s in simulations])
    mean=pooled.mean(0);std=np.maximum(pooled.std(0),NEURAL["normalization_floor"])
    raw=[math.scores(s['phi'],mean,std) for s in simulations]
    refs=math.reference(np.concatenate([r[0] for r in raw],axis=1))
    single_refs=math.reference(np.concatenate([r[1] for r in raw],axis=1))
    simulated=[]
    for item in simulations:
        first,second=conditions[item['condition']]
        passages=passage_indices(item['phi'][:,:2],centers,radii,first,second)
        signature=passage_signature(item['phi'],passages,mean,std,refs,single_refs) if passages else None
        simulated.append(dict(family=item['family'],seed=item['seed'],condition=item['condition'],passages=passages,signature=signature))
    experiment.write(directory/'model_signatures.json',simulated)
    eligible=[]
    for head,gamma in enumerate(math.discount_grid()):
        if .01*gamma/(1-gamma)<.05:continue
        contrasts=[]
        for seed in seeds:
            for condition in range(6):
                values={s['family']:s['signature']['single_effect'][head] for s in simulated
                        if s['seed']==seed and s['condition']==condition and s['signature'] is not None}
                if len(values)==2 and np.isfinite(list(values.values())).all():contrasts.append(values['hierarchical']-values['flat'])
        if contrasts:eligible.append((float(np.mean(contrasts)),head))
    selected_single=max(eligible,key=lambda value:(value[0],-value[1]))[1] if eligible else None
    rows=[];skipped=0
    with zipfile.ZipFile(archive) as zipped:
        names=sorted(n for n in zipped.namelist() if n.endswith('_motion.tsv') and '_task-sequential' in n)
        for name in track(experiment.limited(args,names),description='All participants: frozen kinematic compatibility test'):
            frame=list(csv.DictReader(io.StringIO(zipped.read(name).decode('utf-8')),delimiter='\t'))
            if not all(r['success'].lower()=='true' for r in frame):
                raise ValueError('Unexpected failures in the published success-only archive')
            groups={}
            for row in frame:groups.setdefault(row['unique_trial_id'],[]).append(row)
            trials=sorted(groups.items())
            if args.limit is not None:trials=trials[:1]
            for identifier,trial in trials:
                trial=sorted(trial,key=lambda row:float(row['time']));time=np.array([float(row['time']) for row in trial])
                if not np.allclose(np.diff(time),.01,atol=1e-6):raise ValueError('Human time grid is not 100Hz')
                xy=np.array([[float(row['SensorID_0_x']),float(row['SensorID_0_y'])] for row in trial])/1000
                first=trial[0]['second'];second=trial[0]['third']
                if (first,second) not in conditions:raise ValueError('Unknown published sequence geometry')
                condition=conditions.index((first,second))
                passages=passage_indices(xy,centers,radii,first,second)
                signature=passage_signature(kinematics(xy),passages,mean,std,refs,single_refs) if passages else None
                if signature is None:skipped+=1
                rows.append(dict(participant=int(trial[0]['part']),trial=str(identifier),condition=condition,
                            passages=passages,signature=signature))
    experiment.write(directory/'trial_signatures.json',rows)
    participant_effects=[];similarities=[]
    for participant in sorted({r['participant'] for r in rows}):
        valid=[r for r in rows if r['participant']==participant and r['signature'] is not None]
        if valid:participant_effects.append(dict(participant=participant,value=float(np.mean([r['signature']['mean_difference'] for r in valid])),trials=len(valid)))
        differences=[]
        for r in valid:
            distances={}
            for family in ('flat','hierarchical'):
                comparisons=[]
                for s in simulated:
                    if s['condition']!=r['condition'] or s['family']!=family or s['signature'] is None:continue
                    common=sorted(set(r['signature']['valid_bands'])&set(s['signature']['valid_bands']))
                    if len(common)>=3:
                        a=np.asarray(r['signature']['second_minus_first'])[common]
                        b=np.asarray(s['signature']['second_minus_first'])[common]
                        comparisons.append(float(np.mean((a-b)**2)))
                if comparisons:distances[family]=float(np.mean(comparisons))
            if len(distances)==2:differences.append(distances['flat']-distances['hierarchical'])
        if differences:similarities.append(dict(participant=participant,value=float(np.mean(differences))))
    # Projection is a diagnostic: the ten-feature A--E labels remain frozen.
    old=experiment.load_episode(experiment.sources(args)['reaches']);audit=experiment.read(root/'audit.json')
    old_mean,old_std=experiment.stats(args);projection=[]
    for candidate in audit['candidates']:
        phi=old['phi'][candidate['trial'],:,:4]
        score,_=math.scores(phi,old_mean[:4],old_std[:4])
        k=candidate['step']-1
        projection.append(dict(id=candidate['id'],original_raw_change=candidate['raw_change'],
                           kinematic_band_change=score[:,k],feature_scope='four-dimensional projection; no rediscovery or muscle imputation'))
    experiment.write(directory/'ae_kinematic_projection.json',projection)
    # Dimensionless scale profiles connect fixed A--E to human passages without aligning spatial coordinates.
    source_scores=[math.scores(old['phi'][trial,:,:4],old_mean[:4],old_std[:4])[0][:,:int(old['hit'][trial])]
                   for trial in audit['source_discovery_trials']]
    source_refs=math.reference(np.concatenate(source_scores,axis=1))
    ae_profiles={c['id']:math.percentiles(np.asarray(p['kinematic_band_change'])[:,None],source_refs)[:,0]
                 for c,p in zip(audit['candidates'],projection)}
    correlations=[];single_comparisons=[]
    for participant in sorted({r['participant'] for r in rows}):
        valid=[r for r in rows if r['participant']==participant and r['signature'] is not None]
        for name,profile in ae_profiles.items():
            values={'first':[],'second':[]}
            for row in valid:
                common=np.asarray(row['signature']['valid_bands']);common=common[np.isfinite(profile[common])]
                for passage in values:
                    human=np.asarray(row['signature'][passage])[common]
                    if len(common)>=3 and np.std(human)>1e-12 and np.std(profile[common])>1e-12:
                        values[passage].append(float(np.corrcoef(profile[common],human)[0,1]))
            correlations.append(dict(participant=participant,candidate=name,
                    first_mean_correlation=float(np.mean(values['first'])) if values['first'] else None,
                    second_mean_correlation=float(np.mean(values['second'])) if values['second'] else None,
                    trials=len(values['first'])))
        if selected_single is not None:
            values=[r['signature']['mean_difference']-r['signature']['single_effect'][selected_single]
                    for r in valid if np.isfinite(r['signature']['single_effect'][selected_single])]
            if values:single_comparisons.append(dict(participant=participant,value=float(np.mean(values))))
    experiment.write(directory/'ae_human_correlations.json',correlations)
    experiment.write(directory/'summary.json',dict(complete=args.limit is None,participants=len({r['participant'] for r in rows}),
       trials=len(rows),not_estimable=skipped,participant_effects=participant_effects,
       second_minus_first_predictive_change=math.interval([r['value'] for r in participant_effects]),
       hierarchical_vs_flat_signature_similarity=math.interval([r['value'] for r in similarities]),participant_similarity=similarities,
       selected_single_head=selected_single,single_selection='Maximal hierarchy-minus-flat passage contrast in published-model traces only; human data never used for gamma selection',
       multiscale_minus_single_passage_contrast=math.interval([r['value'] for r in single_comparisons]),
       model_control='Fixed illustrative Figure-8 controller settings; no claim of exhaustive model discrimination or a fitted reproduction',
       observable='100Hz hand kinematics in physical seconds; geometric circle entry labels independent of SF score',
       support='At least three eligible horizons >=50ms; <=0.5% missing band-kernel mass for both passage windows',
       limitation='Successful human trials only; no muscle, EMG, force or neural recordings; compatibility is not proof of the MotorNet mechanism',
       relation_to_ae='Fixed A--E projected scale-profile correlations with human first/second geometric passages; source and model-only empirical ranks; >=50ms shared support; descriptive similarity, no spatial transport or neural identity claim',
       smoke=args.limit is not None))
