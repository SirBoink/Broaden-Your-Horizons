"""Plot landmark discovery, numerical sensitivity and muscle mechanics."""
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import sys
import csv
import io
import zipfile
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/figures'; OUT.mkdir(parents=True, exist_ok=True)
BUILD=ROOT/'data/summary'
font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'))
plt.rcParams.update({'font.family':'EB Garamond','font.size':12,'mathtext.fontset':'cm',
                     'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
COLORS=['#D55E00','#0072B2','#CC79A7','#56B4E9','#E69F00']
sources={Path(__file__)};figures=[];audit={}


def read(path):
    path=ROOT/path;sources.add(path)
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    path=ROOT/path;sources.add(path)
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}


def save(fig,name):
    path=OUT/(name+'_v14.png');fig.savefig(path,dpi=600,facecolor='white',bbox_inches='tight')
    plt.close(fig);figures.append(path)
    return path


catalog=read('data/mechanism/audit.json')
trajectory=arrays('data/transfer/evaluation_seeds/rollouts/order_16_ddqn.npz')
targets=trajectory['targets'];center=targets.mean(axis=0)


def ring(ax):
    ax.add_patch(Circle(center,.1,fill=False,edgecolor='.65',ls=':',lw=1))
    for i,xy in enumerate(targets):
        ax.add_patch(Circle(xy,.02,fill=False,edgecolor='black',lw=.8))
        ax.text(*xy,f'T{i}',ha='center',va='center',fontsize=10)
    ax.set(xlabel='Fingertip x (m)',ylabel='Fingertip y (m)',aspect='equal',
           xlim=(center[0]-.14,center[0]+.14),ylim=(center[1]-.14,center[1]+.14))


fig,axes=plt.subplots(1,2,figsize=(9.3,3.7),layout='constrained');ring(axes[0])
for i,candidate in enumerate(catalog['candidates']):
    axes[0].scatter(*candidate['xy_m'],color=COLORS[i],s=44,edgecolor='black',lw=.5,zorder=4)
    axes[0].annotate(candidate['id'],candidate['xy_m'],xytext=(5,6),textcoords='offset points',color=COLORS[i],fontsize=12)
    option=catalog['original_library'][i]
    tau=-.01/np.log(option['gamma'])
    axes[1].scatter(candidate['step']*10,tau*1000,color=COLORS[i],s=50,edgecolor='black',lw=.5)
    axes[1].annotate(candidate['id'],(candidate['step']*10,tau*1000),xytext=(5,5),textcoords='offset points')
axes[0].set_title('Actual landmarks within the target-ring workspace')
axes[1].set(xlabel='Discovery time after target activation (ms)',ylabel='Option discount horizon (ms)',yscale='log',title='Elapsed time and prediction horizon differ')
save(fig,'landmarks_ring')

from scripts.plot_subgoal_selection_clear import load_selection
selected=load_selection()
sources.update([ROOT/'scripts/plot_subgoal_selection_clear.py',ROOT/'data/mechanism/references.npz',
                ROOT/'data/discovery/normalization/temporal_band_pilot.npz'])
fig=plt.figure(figsize=(10,5.5),layout='constrained');grid=fig.add_gridspec(2,6)
slots=[grid[0,:2],grid[0,2:4],grid[0,4:],grid[1,1:3],grid[1,3:5]]
for item,color,slot in zip(selected,COLORS,slots):
    candidate=item['candidate'];sources.add(ROOT/f"data/mechanism/integration/source_{candidate['trial']}.npz")
    values=item['raw'][candidate['peak_band']];column=candidate['step']-1
    assert np.isclose(values[column],candidate['raw_change'],rtol=0,atol=1e-12)
    ax=fig.add_subplot(slot);time=item['time_ms']/1000
    ax.plot(time,values,color=color,lw=1.7);ax.scatter(time[column],values[column],color=color,s=30)
    ax.annotate(f"{candidate['step']*10} ms",(time[column],values[column]),xytext=(3,7),textcoords='offset points',fontsize=11,color=color)
    ax.set(xlabel='Time after target activation (s)',ylabel='Cosine change per 10 ms',title=f"Option {candidate['id']}",ylim=(0,np.nanmax(values)*1.25))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
save(fig,'selection_peaks')

# Native recorded phase planes. One common-launch invocation per option, no artificial curves.
kinematics=arrays('data/handoffs/controlled_launch/option_termination_kinematics.npz')
meta=read('data/handoffs/controlled_launch/option_termination_kinematics.json')
fig,axes=plt.subplots(1,3,figsize=(10,3.7),layout='constrained')
for name,color in zip('ABCDE',COLORS):
    joint=kinematics[name+'_joint_window'];cart=kinematics[name+'_cartesian_window']
    for ax,values in zip(axes,[np.rad2deg(joint[:,:2]),joint[:,2:],cart[:,2:]]):
        assert values.shape==(11,2)
        ax.plot(*values[:6].T,color=color,lw=1.8,label=name)
        ax.plot(*values[5:].T,color=color,lw=1.3,ls='--',alpha=.6)
        ax.scatter(*values[0],s=22,facecolor='white',edgecolor=color)
        ax.scatter(*values[5],s=36,color=color,zorder=4)
for ax,title,x,y in zip(axes,['Joint configuration','Joint velocity','Fingertip velocity'],
                       ['Shoulder angle (deg)','Shoulder velocity (rad/s)','Horizontal velocity (m/s)'],
                       ['Elbow angle (deg)','Elbow velocity (rad/s)','Vertical velocity (m/s)']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.margins(.15)
axes[0].legend(ncol=5,loc='upper center',bbox_to_anchor=(1.8,1.2))
save(fig,'termination_phase_planes')

# Evaluation random seeds alter initial postures, not controller weights or action noise.
folder='data/transfer/evaluation_seeds/rollouts'
by_seed=defaultdict(lambda:defaultdict(list));by_order=defaultdict(lambda:defaultdict(list))
for order in range(24):
    base=read(f'{folder}/order_{order:02d}_base.json');learned=read(f'{folder}/order_{order:02d}_ddqn.json')
    assert base['contexts']==learned['contexts']
    for j,context in enumerate(base['contexts']):
        for method,data in [('base',base),('ddqn',learned)]:
            row=data['outcomes'][j];by_seed[context['context_seed']][method].append(row);by_order[order][method].append(row)
seeds=[42,101,202,303,404];fig,axes=plt.subplots(2,2,figsize=(10,6.4),layout='constrained')
for offset,method,label,color in [(-.18,'base','Base','#777777'),(.18,'ddqn','DDQN + A–E','#B52F62')]:
    rates=[100*np.mean([r['success'] for r in by_seed[seed][method]]) for seed in seeds]
    times=[np.mean([r['restricted_completion_s'] for r in by_seed[seed][method]]) for seed in seeds]
    axes[0,0].bar(np.arange(5)+offset,rates,.35,label=label,color=color)
    axes[0,1].bar(np.arange(5)+offset,times,.35,label=label,color=color)
    values=[np.mean([r['targets_completed'] for r in by_order[i][method]]) for i in range(16,24)]
    axes[1,0].plot(range(16,24),values,'o-',color=color,label=label)
    for i,rate in enumerate(rates):axes[0,0].text(i+offset,rate+1.5,f'{rate:.1f}',ha='center',fontsize=9)
for ax in axes[0]:ax.set(xticks=range(5),xticklabels=[str(s) for s in seeds],xlabel='Evaluation seed (two starting postures per order)')
axes[0,0].set(ylabel='Sequence completion (%)',ylim=(0,105),title='All 24 orders: 48 trials per seed');axes[0,0].legend()
axes[0,1].set(ylabel='Failure-capped completion time (s)',title='Every failure contributes the 8 s cap')
axes[1,0].set(xlabel='Target order index',ylabel='Targets acquired (of 8)',ylim=(0,8.5),title='All eight short-transition orders')
gains=[np.mean([int(d['success'])-int(b['success']) for b,d in zip(by_seed[seed]['base'],by_seed[seed]['ddqn'])])*100 for seed in seeds]
axes[1,1].bar(range(5),gains,color='#B52F62');axes[1,1].axhline(0,color='black',lw=.8)
axes[1,1].set(xticks=range(5),xticklabels=[str(s) for s in seeds],xlabel='Evaluation seed',ylabel='Paired completion gain (percentage points)',title='Positive gain under every tested start seed')
save(fig,'evaluation_seed_transfer')
audit['evaluation_seeds']={str(seed):{method:dict(successes=sum(r['success'] for r in by_seed[seed][method]),n=len(by_seed[seed][method])) for method in ['base','ddqn']} for seed in seeds}

# Actual predictive scale profiles from the original cache and all human participant means.
import formulas as math
legacy=ROOT
cache_path=legacy/'data/discovery/isolated_reaches/reaches.npz';sources.add(cache_path)
with np.load(cache_path) as archive:cache={k:archive[k].copy() for k in archive.files}
stats=arrays('data/discovery/normalization/temporal_band_pilot.npz')
raw={i:math.scores(cache['phi'][i,:,:4],stats['feature_mean_raw'][:4],stats['feature_std'][:4])[0] for i in catalog['source_discovery_trials']}
refs=math.reference(np.concatenate([raw[i][:,:int(cache['hit'][i])] for i in raw],axis=1))
profiles={c['id']:math.percentiles(raw[c['trial']],refs)[:,c['step']-1] for c in catalog['candidates']}
human=read('data/mechanism/human/trial_signatures.json')
people=defaultdict(lambda:defaultdict(list))
for row in human:
    if row['signature'] is None:continue
    for passage in ['first','second']:
        values=np.array([np.nan if x is None else x for x in row['signature'][passage]])
        mask=np.zeros(39,dtype=bool);mask[row['signature']['valid_bands']]=True;values[~mask]=np.nan
        people[row['participant']][passage].append(values)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
scale_time=np.geomspace(10,1800,39)
for passage,color,style in [('first','#0072B2','--'),('second','#0072B2','-')]:
    participant_means=np.stack([np.nanmean(v[passage],axis=0) for v in people.values()])
    axes[0].plot(scale_time,np.nanmean(participant_means,axis=0),color=color,ls=style,label='Human '+passage)
for name,color,style in [('A',COLORS[0],'-'),('E',COLORS[4],':')]:axes[0].plot(scale_time,profiles[name],color=color,ls=style,label='Landmark '+name)
axes[0].set(xlabel='Band kernel peak time (ms)',xscale='log',ylabel='Within-band percentile (0–1)',title='Kinematic prediction profiles on the same reference');axes[0].legend(fontsize=10)
corr=read('data/mechanism/human/ae_human_correlations.json')
for offset,key,label,color in [(-.17,'first_mean_correlation','First passage','#56B4E9'),(.17,'second_mean_correlation','Second passage','#0072B2')]:
    for index,name in enumerate('ABCDE'):
        values=np.array([r[key] for r in corr if r['candidate']==name]);assert len(values)==20
        lo,hi=np.quantile(np.random.default_rng(42).choice(values,(10000,20)).mean(1),[.025,.975]);mean=values.mean()
        axes[1].bar(index+offset,mean,.33,color=color,label=label if index==0 else None)
        axes[1].errorbar(index+offset,mean,yerr=[[mean-lo],[hi-mean]],color='black',fmt='none',capsize=2)
axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=range(5),xticklabels=list('ABCDE'),ylim=(-1,1),ylabel='Participant-mean profile correlation',title='Both intermediate target passages; n = 20');axes[1].legend(fontsize=10)
save(fig,'scale_profiles_selectivity')

# Restore the three-panel human movement layout with measured, unsynthesized curves.
aggregates=arrays('data/summary/human_curve_aggregates_v10.npz')
archive=ROOT/'data/human/bids_data.zip';sources.add(archive)
with zipfile.ZipFile(archive) as z:
    path=next(n for n in z.namelist() if 'sub-401_task-sequential0_tracksys-qualisys_motion.tsv' in n)
    paths=defaultdict(list)
    for row in csv.DictReader(io.StringIO(z.read(path).decode()),delimiter='\t'):paths[row['unique_trial_id']].append(row)
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for index,(identifier,rows) in enumerate(sorted(paths.items())):
    rows.sort(key=lambda r:float(r['time']))
    xy=np.array([[float(r['SensorID_0_x']),float(r['SensorID_0_y'])] for r in rows])/1000
    axes[0].plot(*xy.T,color='#0072B2',alpha=.2,lw=.8)
    if index==0:
        example=xy;example_id=identifier
axes[0].plot(*example.T,color='#0072B2',lw=2)
signature=next(r for r in human if str(r['trial'])==str(example_id))
via=example[signature['passages'][1]]
axes[0].scatter(*via,color='#0072B2',s=35,edgecolor='black',zorder=5)
axes[0].annotate('Second target entry',via,xytext=(8,5),textcoords='offset points',fontsize=10)
axes[0].set(xlabel='Fingertip x (m)',ylabel='Fingertip y (m)',aspect='equal',title=f'Participant 401, sequence 0: {len(paths)} trials')
for key,values in aggregates.items():
    label='Human' if key=='Human' else 'Base' if key=='Base' else 'DDQN + A–E'
    color={'Human':'#0072B2','Base':'#777777','DDQN + A–E':'#B52F62'}[label]
    for j,ax in enumerate(axes[1:]):
        lo,median,hi=np.nanpercentile(values[:,j],[25,50,75],axis=0)
        ax.fill_between(np.arange(-30,31)*10,lo,hi,color=color,alpha=.12)
        ax.plot(np.arange(-30,31)*10,median,color=color,label=label,lw=1.7)
for ax in axes[1:]:ax.axvline(0,color='black',ls=':',lw=.8);ax.set(xlabel='Time from second target entry (ms)',xlim=(-300,300))
axes[1].set(ylabel='Speed / trial maximum (0–1)',ylim=(0,1),title='Measured velocity profiles');axes[1].legend(fontsize=10)
axes[2].set(ylabel=r'Trajectory curvature $\kappa$ (m$^{-1}$)',yscale='log',title='Measured curvature profiles')
save(fig,'human_trajectory_velocity_curvature')
audit['human_trajectory']=dict(subject=401,condition=0,trials=len(paths),highlighted_trial=example_id,
                             selection='First sorted trial, no outcome/profile selection',source_member=path,
                             curve_scope='Human curves aggregate 20 participants; agent curves aggregate 24 orders. This spatial panel is one participant/condition.')

overnight='data/controls'
source_controls=ROOT/overnight/'plots/human_profile_controls.png';sources.add(source_controls)
sources.update([ROOT/overnight/'human_controls.json',ROOT/'scripts/run_overnight_claims.py'])
control_copy=OUT/'human_matched_controls_v14.png';shutil.copyfile(source_controls,control_copy);figures.append(control_copy)
contrasts=read(overnight+'/replay/replay_comparison.json')
fig,axes=plt.subplots(1,2,figsize=(9.8,3.6),layout='constrained')
for i,(dt,title) in enumerate([('0.01','Euler 10 ms'),('0.0005','RK4 0.5 ms')]):
    for choice,color in enumerate(COLORS,1):
        values=np.array([r['chosen_minus_base_progress_500ms_m'][dt]*1000 for r in contrasts if r['metadata']['choice']==choice])
        axes[i].scatter(choice+np.linspace(-.12,.12,len(values)),values,s=15,color=color,alpha=.7)
        axes[i].plot([choice-.15,choice+.15],[values.mean()]*2,color='black',lw=2)
    axes[i].axhline(0,color='black',lw=.8)
    axes[i].set(xticks=range(1,6),xticklabels=list('ABCDE'),ylabel='Chosen minus base progress at 500 ms (mm)',title=title)
save(fig,'expanded_local_continuation')
audit['local_replay']={dt:dict(n=len(contrasts),positive=sum(r['chosen_minus_base_progress_500ms_m'][dt]>1e-6 for r in contrasts),
                                 negative=sum(r['chosen_minus_base_progress_500ms_m'][dt]<-1e-6 for r in contrasts),
                                 mean_mm=float(np.mean([r['chosen_minus_base_progress_500ms_m'][dt] for r in contrasts])*1000)) for dt in ['0.01','0.001','0.0005']}

# Small new fit comparison belongs in backup and the final limitations, with every seed.
summary=read(overnight+'/SUMMARY.json');base_successes=64
fig,ax=plt.subplots(figsize=(9.3,3.4),layout='constrained')
names=['multiscale','single','spatial','kinematic']
for index,name in enumerate(names):
    values=[base_successes/96*100+100*r['success_effect'] for r in summary['effects'] if r['fit'].startswith(name+'_')]
    assert len(values)==3
    ax.scatter(index+np.array([-.08,0,.08]),values,color='#777777',s=35)
    ax.plot([index-.17,index+.17],[np.mean(values)]*2,color='black',lw=2)
ax.axhline(100*64/96,color='#B52F62',ls='--',label='Base: 64/96')
ax.set(xticks=range(4),xticklabels=['Multiscale A–E','Selected single','Spatial','Kinematic'],ylabel='Full-sequence completion (%)',ylim=(-2,105),title='Equal 60,000-step budgets; final checkpoints; all three DDQN fits');ax.legend()
save(fig,'overnight_libraries')

# Native muscle moment arms and saved forces, never illustrative torque numbers.
from scripts import evaluate_landmark_handoffs as h
h.torch.set_num_threads(1)
endpoints=read('data/handoffs/actual_endpoints.json')
valid=[r for r in endpoints if r['actual_endpoint'] is not None]
joint=h.torch.tensor([r['actual_endpoint']['joint'] for r in valid],dtype=h.torch.float32)
task,_=h.e.make_task(physical_dt=.01)
moments=h.e.array(task.effector.get_geometry(joint))[:,2:]
forces=np.array([r['actual_endpoint']['force_N'] for r in valid]);contributions=-moments*forces[:,None,:]
net=np.array([r['actual_endpoint']['muscle_joint_torque_Nm'] for r in valid])
np.testing.assert_allclose(contributions.sum(axis=2),net,atol=2e-4,rtol=2e-5)
groups=defaultdict(list)
for row,torques in zip(valid,contributions):
    key=(row['choice'],row['policy_seed'],row['order'],row['context_seed'],row['posture'])
    groups[key].append(torques)
fig,axes=plt.subplots(1,2,figsize=(9.8,3.7),layout='constrained');torque_audit={}
for option in range(1,6):
    # Average torque summaries within episode before reporting their across-episode means.
    selected=[np.array(value) for key,value in groups.items() if key[0]==option]
    positive=np.stack([np.maximum(v,0).sum(axis=2).mean(axis=0) for v in selected])
    negative=np.stack([np.minimum(v,0).sum(axis=2).mean(axis=0) for v in selected])
    for joint_index,ax in enumerate(axes):
        ax.bar(option-.15,positive[:,joint_index].mean(),.28,color='#0072B2',label='Positive muscle torque' if option==1 else None)
        ax.bar(option+.15,negative[:,joint_index].mean(),.28,color='#D55E00',label='Negative muscle torque' if option==1 else None)
        ax.scatter(option,(positive[:,joint_index]+negative[:,joint_index]).mean(),color='black',s=26,zorder=4,label='Net muscle torque' if option==1 else None)
    torque_audit[chr(64+option)]=dict(episodes=len(selected),invocations=sum(len(v) for v in selected),
                                    positive_mean_Nm=positive.mean(axis=0).tolist(),negative_mean_Nm=negative.mean(axis=0).tolist())
for ax,title in zip(axes,['Shoulder','Elbow']):
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(1,6),xticklabels=list('ABCDE'),xlabel='Option arrival',ylabel='Muscle torque (N m)',title=title)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.07),ncol=3,fontsize=10)
save(fig,'actual_torque_components');audit['torques']=torque_audit
sources.update([ROOT/'scripts/evaluate_landmark_handoffs.py',ROOT/'docs/references/motornet/effector.py'])

# Reward terms use the actual StraightReach environment used by the selector.
sources.update([ROOT/'SAC/changed/env.py',ROOT/'SAC/high_level.py'])
fig,ax=plt.subplots(figsize=(10,3.6));ax.axis('off')
ax.text(.03,.88,r'$r_t=\frac{d_t-d_{t+1}}{0.01}-0.1\frac{d_{t+1}}{0.10}-\lambda_{\rm path}\frac{w_t}{0.01}+20 I_t+200 F_t$',fontsize=22)
ax.text(.03,.64,r'$w_t=\max(\|x_{t+1}-x_t\|-(d_t-d_{t+1}),0)$',fontsize=21)
ax.text(.03,.40,r'$R_k=\sum_{j=0}^{D_k-1}0.99^j r_{t+j}-c_{\rm delib}$',fontsize=22)
ax.text(.03,.11,'Distance d is measured to the current task target. I marks target acquisition, F sequence completion.\nThe path term penalizes excess travel. R is the selector return over the executed option duration.',fontsize=13)
save(fig,'reward_equations')

(OUT/'v14_figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(OUT/'v14_figure_manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf-8')
print(f'Generated {len(figures)} source-backed 600-dpi panels; native torque identity verified.')
