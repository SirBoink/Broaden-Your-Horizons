"""Actual DDQN invocation branches and shared-hierarchy transfer across base policies.

No training. --pilot is bounded and produces runtime/replay checks, not scientific evidence.
Full runs are operator-launched, resumable and show Rich progress.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from rich.progress import Progress
import torch
from stable_baselines3 import SAC
from analysis import mechanism_experiment as e

POLICIES={7:e.BASE,101:ROOT/'checkpoints/base_policy_101/best_model.zip',202:ROOT/'checkpoints/base_policy_202/best_model.zip'}

def snapshot(task, obs, row):
    state={k:np.asarray(v[row])[None].copy() for k,v in e.capture(task,obs).items()}
    state['sequence']=np.asarray(task._sequence).reshape(-1).copy()
    state['elapsed_step']=np.asarray(task.elapsed_steps)
    return state

def restore(state, dt):
    task,obs=e.restore(state,0,dt)
    task.elapsed_steps=int(state['elapsed_step']);task.elapsed=task.elapsed_steps*.01
    task.max_episode_steps=800
    return task,obs

def repeat(task,obs,n):
    task.effector.states={k:v.repeat((n,)+(1,)*(v.ndim-1)) for k,v in task.states.items()}
    for key in ['sequence_index','finished','hold_count']:setattr(task,key,getattr(task,key).repeat(n))
    task.goal=task.goal.repeat(n,1)
    task.obs_buffer={k:[v.repeat(n,1) for v in values] for k,values in task.obs_buffer.items()}
    return np.repeat(obs,n,axis=0)

def mirror_waypoint(start,goal,waypoint,task):
    axis=goal-start;length=np.linalg.norm(axis)
    if length<1e-9:return None,'zero goal distance'
    axis/=length;delta=waypoint-start
    mirror=start+2*axis*np.dot(delta,axis)-delta
    sk=task.effector.skeleton
    cosine=(mirror@mirror-sk.L1**2-sk.L2**2)/(2*sk.L1*sk.L2)
    if abs(cosine)>1:return None,'outside arm reach'
    elbow=np.arccos(cosine)
    shoulder=np.arctan2(mirror[1],mirror[0])-np.arctan2(sk.L2*np.sin(elbow),sk.L1+sk.L2*np.cos(elbow))
    q=np.array([shoulder,elbow])
    if np.any(q<e.array(task.effector.pos_lower_bound).reshape(-1)) or np.any(q>e.array(task.effector.pos_upper_bound).reshape(-1)):
        return None,'joint limits'
    if np.linalg.norm(mirror-waypoint)<.005:return None,'mirror coincides with chosen waypoint'
    if np.min(np.linalg.norm(e.array(task.targets)-mirror,axis=1))<=.025:return None,'near task target'
    assert np.isclose(np.linalg.norm(mirror-start),np.linalg.norm(waypoint-start),atol=1e-8)
    assert np.isclose(np.linalg.norm(mirror-goal),np.linalg.norm(waypoint-goal),atol=1e-8)
    return mirror,None

def endpoint(task,row):
    activation=e.array(task.states['muscle'])[row,0]
    force=e.array(e.muscle_forces(task.effector,e.array(task.states['muscle'])[:,0]))[row]
    moments=e.array(task.states['geometry'])[row,2:]
    return dict(joint=e.array(task.states['joint'])[row].tolist(),cartesian=e.array(task.states['cartesian'])[row].tolist(),
                activation=activation.tolist(),force_N=force.tolist(),muscle_joint_torque_Nm=(-moments*force).sum(axis=1).tolist())

def branches(state,choice,library,policies,dt,horizon):
    task,obs=restore(state,dt);start=e.features(task)[0,:2];goal=e.array(task.goal)[0];visit=int(task.sequence_index[0])
    distance=np.linalg.norm(goal-start)
    mirror,reason=mirror_waypoint(start,goal,np.array(library[choice-1]['xy_m']),task)
    branch_library=list(library)
    names=['base','chosen'];choices=[0,choice]
    for j,option in enumerate(library,1):
        if j!=choice and np.linalg.norm(np.array(option['xy_m'])-start)>.02:
            names.append('other_'+str(j));choices.append(j)
    if mirror is not None:
        branch_library.append(dict(xy_m=mirror.tolist(),timeout_steps=30));names.append('mirror');choices.append(len(branch_library))
    n=len(names);obs=repeat(task,obs,n)
    first=True
    def selector(observation):
        nonlocal first
        q=torch.zeros((len(observation),len(branch_library)+1));q[:,0]=1
        if first:
            assert len(observation)==n
            for row,c in enumerate(choices):q[row,c]=2
            first=False
        return q
    acquisition=np.full(n,-1,int);progress=None;ends={}
    traces={'phi':[e.features(task)],'joint':[e.array(task.states['joint'])]}
    def observe(step,task,obs,executed):
        nonlocal progress
        hit=e.array(task.finished).astype(bool)|(e.array(task.sequence_index)!=visit)
        acquisition[(acquisition<0)&hit]=step
        if step==50:progress=np.where(acquisition>=0,distance,distance-np.linalg.norm(e.features(task)[:,:2]-goal,axis=1))
        traces['phi'].append(e.features(task));traces['joint'].append(e.array(task.states['joint']))
    def event(kind,step,row,c,task,obs,why):
        if kind=='end':ends[row]=dict(step=step,reason=why,**endpoint(task,row))
    budget=min(horizon,800-int(state['elapsed_step']))
    result=e.transfer_batch(task,obs,policies,(15.,1.5),'learned',branch_library,selector,budget,observer=observe,event_observer=event)
    if progress is None and len(traces['phi'])<=50 and all(r['success'] for r in result):
        progress=np.full(n,distance)
    rows=[]
    for row,(name,outcome) in enumerate(zip(names,result)):
        rows.append(dict(branch=name,choice=choices[row],progress_500ms_m=None if progress is None else float(progress[row]),
                         target_acquired_500ms=bool(0<acquisition[row]<=50),first_target_acquisition_step=int(acquisition[row]),
                         endpoint=ends.get(row),**outcome))
    return dict(rows=rows,mirror_exclusion=reason,original_goal_m=goal.tolist(),initial_distance_m=float(distance),
                horizon_steps=budget,estimand='One naturally terminating option, then base continuation; no subsequent selector interventions'),traces

def contexts(panel,order,count):
    # Identical established context definition; this replication panel remains retrospective.
    from scripts.evaluate_hierarchy_geometry import contexts as existing
    return existing(panel,order,count)

def analyze(root):
    contrasts=[];endpoints=[]
    metrics={'progress_500ms_m':1,'success':1,'restricted_completion_s':-1,'targets_completed':1}
    for file in sorted((root/'branches').glob('*.json')):
        trial=e.read(file);meta=trial['metadata'];by={r['branch']:r for r in trial['rows']}
        endpoints.append(dict(**meta,actual_endpoint=trial.get('actual_endpoint'),mirror_exclusion=trial['mirror_exclusion']))
        for comparator in ['base','mirror','other_mean']:
            others=[v for k,v in by.items() if k.startswith('other_')] if comparator=='other_mean' else [by[comparator]] if comparator in by else []
            for metric,sign in metrics.items():
                values=[r[metric] for r in others if r[metric] is not None]
                chosen=by['chosen'][metric]
                if values and chosen is not None:
                    contrasts.append(dict(**meta,comparator=comparator,metric=metric,benefit=sign*(float(chosen)-float(np.mean(values)))))
    summary=[]
    for comparator in ['base','mirror','other_mean']:
        for metric in metrics:
            subset=[r for r in contrasts if r['comparator']==comparator and r['metric']==metric];groups={}
            for row in subset:groups.setdefault((row['policy_seed'],row['order'],row['context_seed'],row['posture']),[]).append(row['benefit'])
            policy_means={str(seed):float(np.mean([np.mean(v) for key,v in groups.items() if key[0]==seed])) for seed in sorted({key[0] for key in groups})}
            summary.append(dict(comparator=comparator,metric=metric,invocations=len(subset),episodes=len(groups),policy_mean_benefits=policy_means,
                                mean_episode_benefit=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None))
    transfer={}
    for file in sorted((root/'rollouts').glob('*.json')):
        job=e.read(file);seed=str(job['policy_seed']);dest=transfer.setdefault(seed,{'cases':0,'base_successes':0,'ddqn_successes':0,'paired_success_difference':[]})
        for base,ddqn in zip(job['base'],job['ddqn']):
            dest['cases']+=1;dest['base_successes']+=int(base['success']);dest['ddqn_successes']+=int(ddqn['success'])
            dest['paired_success_difference'].append(int(ddqn['success'])-int(base['success']))
    for value in transfer.values():value['paired_success_effect']=float(np.mean(value.pop('paired_success_difference')))
    e.write(root/'branch_contrasts.json',contrasts);e.write(root/'actual_endpoints.json',endpoints)
    e.write(root/'summary.json',dict(contrasts=summary,transfer=transfer,protocol=e.read(root/'protocol.json'),
                                    endpoint_invocations=len(endpoints),recorded_terminations=sum(r['actual_endpoint'] is not None for r in endpoints),
                                    scope='Exploratory; invocation effects averaged within episodes; three base policies share one selector/reacher. Positive benefit favors chosen.'))
    print(json.dumps({'contrasts':summary,'transfer':transfer},indent=2))

def run(args):
    torch.set_num_threads(1);started=time.perf_counter()
    panel_path=ROOT/'data/mechanism/panel.json';panel=e.read(panel_path)
    sequences=[seq for group in panel['groups'].values() for seq in group];library=e.read(e.LIBRARY)['options']
    seed_list=[7] if args.pilot else [7,101,202];orders=1 if args.pilot else args.orders;context_count=2 if args.pilot else args.contexts
    steps=200 if args.pilot else 800;max_events=2 if args.pilot else None
    protocol=dict(version=2,pilot=args.pilot,policy_seeds=seed_list,orders=orders,contexts=context_count,collection_steps=steps,
                  physical_dt=args.physical_dt,branch_horizon_steps=800,command_dt=.01,option_cap=30,termination_radius_m=.02,
                  shared_reacher_and_selector=True,retrospective_existing_order_panel=True,
                  primary_outcome='Remaining-sequence completion within original 8s episode budget; first average invocations within episodes',
                  secondary_outcomes=['restricted remaining completion time, failures capped at remaining budget','500ms original-target progress, acquisition absorbing','targets completed'],
                  inference='Exploratory summaries only; full clustered uncertainty required before slide claims',
                  comparisons=['chosen-minus-base','chosen-minus-geometric-mirror','chosen-minus-mean-other-eligible-landmarks'],
                  mirror_matching='Same initial distance to waypoint and waypoint distance to current goal; pre-intervention IK/target exclusions',
                  source_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [*POLICIES.values(),e.REACHER,e.SELECTOR,e.LIBRARY,panel_path,Path(__file__),ROOT/'analysis/mechanism_experiment.py']})
    root=args.output
    if (root/'protocol.json').exists() and e.read(root/'protocol.json')!=protocol:raise ValueError('Changed protocol; use a new output directory')
    e.write(root/'protocol.json',protocol)
    original,reacher=e.models();selector=e.verified_selector();checks=[];event_count=0
    with Progress() as progress:
        jobs=progress.add_task('Policy/order evaluations',total=len(seed_list)*orders)
        for seed in seed_list:
            base=original if seed==7 else SAC.load(str(POLICIES[seed]),device='cpu');base.policy.set_training_mode(False);policies=(base,reacher)
            for order in range(orders):
                batch=contexts(panel,order,context_count);job=root/'rollouts'/f'seed{seed}_order{order:02d}.json'
                if job.exists():progress.advance(jobs);continue
                task,obs=e.make_task(sequences[order],args.physical_dt,[r['q'] for r in batch],800)
                events=[];pending={}
                def event(kind,step,row,choice,task,obs,reason):
                    nonlocal event_count
                    if kind=='start':
                        if max_events is not None and event_count>=max_events:return
                        event_count+=1
                        item=dict(metadata=dict(policy_seed=seed,order=order,**batch[row],invocation=len(events),choice=choice,start_step=step),state=snapshot(task,obs,row))
                        events.append(item);pending[row]=item
                    elif row in pending:
                        item=pending.pop(row);item['actual_endpoint']=dict(step=step,reason=reason,**endpoint(task,row))
                ddqn=e.transfer_batch(task,obs,policies,(15.,1.5),'learned',library,selector,steps,event_observer=event)
                task,obs=e.make_task(sequences[order],args.physical_dt,[r['q'] for r in batch],800)
                baseline=e.transfer_batch(task,obs,policies,(15.,1.5),'base',library,steps=steps)
                branch_progress=progress.add_task(f'Seed {seed}, order {order}: actual invocation branches',total=len(events))
                for index,item in enumerate(events):
                    dest=root/'branches'/f'seed{seed}_order{order:02d}_event{index:04d}'
                    if dest.with_suffix('.json').exists():progress.advance(branch_progress);continue
                    state=item.pop('state');task,obs=restore(state,args.physical_dt)
                    # Replay sanity: captured observation, buffers and native state are restored exactly.
                    np.testing.assert_allclose(e.features(task)[0],state['phi'][0],atol=2e-6,rtol=0)
                    result,traces=branches(state,item['metadata']['choice'],library,policies,args.physical_dt,800)
                    selected=result['rows'][1]['endpoint']
                    if item.get('actual_endpoint') and selected:
                        assert selected['step']==item['actual_endpoint']['step']-item['metadata']['start_step']
                        np.testing.assert_allclose(selected['joint'],item['actual_endpoint']['joint'],atol=3e-5,rtol=0)
                    e.save_arrays(dest.with_suffix('.npz'),**{k:np.asarray(v) for k,v in traces.items()})
                    e.save_arrays(dest.with_name(dest.name+'_start').with_suffix('.npz'),**state)
                    e.write(dest.with_suffix('.json'),dict(**item,**result));progress.advance(branch_progress)
                e.write(job,dict(policy_seed=seed,order=order,contexts=batch,ddqn=ddqn,base=baseline,invocations=len(events)))
                progress.advance(jobs)
                if args.pilot and time.perf_counter()-started>240:raise TimeoutError('Pilot budget reached; launch full run manually')
    e.write(root/'runtime.json',dict(seconds=time.perf_counter()-started,pilot=args.pilot,events=event_count))
    analyze(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--physical-dt',type=float,default=.01)
    parser.add_argument('--orders',type=int,default=24);parser.add_argument('--contexts',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.orders<=24 or not 1<=args.contexts<=10:parser.error('orders 1–24; contexts 1–10')
    if args.analyze:analyze(args.output)
    else:run(args)
