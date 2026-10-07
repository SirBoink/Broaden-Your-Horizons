"""Paper Eq10/author curvature routine at A-E. No training, no synthetic data."""
from pathlib import Path
import json,csv,ast,importlib.util,hashlib,warnings
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from rich.progress import track
ROOT=Path(__file__).resolve().parents[1];paper=ROOT/'data/human/published_models';raw=ROOT/'data/human/base_trajectories';out=ROOT/'results/curvature';out.mkdir(parents=True,exist_ok=True)
import sys
sys.path.insert(0, str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_paths=[Path(__file__),paper/'analyses.py',paper/'import_data.py',paper/'utils.py',paper/'all_data.csv',paper/'data_chalf.csv',ROOT/'docs/references/sequential_movements.pdf',ROOT/'data/mechanism/human/experiment_sheet.svg',ROOT/'data/discovery/options/off_target_band_options.json']
node=next(n for n in ast.parse((paper/'analyses.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='curvature_one_traj')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'author_curvature_function','exec'),namespace);author=namespace['curvature_one_traj']
def curvature_mm(xy,t):
 with np.errstate(divide='ignore',invalid='ignore'):
  v=np.gradient(xy,t,axis=0);acc=np.gradient(v,t,axis=0);c=(v*acc[:,::-1]*[1,-1]).sum(axis=1)/np.linalg.norm(v,axis=1)**3
 kept=np.abs(c)<10
 np.testing.assert_allclose(c[kept],author(xy,t))
 return np.where(kept,np.abs(c)*1000,np.nan)
theta=np.linspace(0,1,201);xy=100*np.column_stack([np.cos(theta),np.sin(theta)]);assert np.allclose(np.nanmedian(curvature_mm(xy,theta)[20:-20]),10,rtol=.001)
def region(xy,t,center,radius):
 inside=np.linalg.norm(xy-center,axis=1)<=radius+1
 hits=np.flatnonzero(inside)
 if not len(hits):return None,'no_entry'
 first=int(hits[0]);exits=np.flatnonzero(~inside[first:])
 if first<5:return None,'entry_before_five_frame_padding'
 if not len(exits):return None,'no_exit'
 end=first+int(exits[0]);indices=np.arange(first-5,min(end+5,len(xy)))
 with np.errstate(divide='ignore',invalid='ignore'):values=author(xy[indices],t[indices])
 values=np.abs(values)*1000
 if not len(values):return None,'no_retained_curvature'
 return dict(entry=first,exit=end,maximum=float(values.max()),discarded_samples=int(len(indices)-len(values))),None
# Author supplied, preprocessed time stamps/positions; no additional smoothing.
spec=importlib.util.spec_from_file_location('paper_utils',paper/'utils.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
basket=u.coords_from_svg(filename=str(source_paths[7]),reference='mk_tr')
human_trials=defaultdict(list);human_meta={}
with (paper/'all_data.csv').open() as f:
 for row in csv.DictReader(f):
  if row['condition']!='sequential':continue
  uid=row['unique_trial_id'];human_trials[uid].append([float(row['pos_x'] or 'nan'),float(row['pos_y'] or 'nan'),float(row['tframe'] or 'nan')]);human_meta[uid]=(int(row['part']),row['third'])
published={}
with (paper/'data_chalf.csv').open() as f:
 for row in csv.DictReader(f):published[row['unique_trial_id']]=float(row['curvature_st'])*1000
human_curves=defaultdict(list);human_max=defaultdict(list);reproduction=[];human_exclusions=defaultdict(int);grid=np.arange(-30,31)*.01
for uid,rows in track(human_trials.items(),description='Verify author human curvature'):
 a=np.array(rows);xy=a[:,:2];t=a[:,2];part,label=human_meta[uid];circle=basket[label];result,reason=region(xy,t,circle.center,circle.radius)
 if reason:human_exclusions[reason]+=1;continue
 human_max[part].append(published[uid]);reproduction.append(dict(uid=uid,recalculated=result['maximum'],published=published[uid],abs_difference=abs(result['maximum']-published[uid])))
 k=result['entry'];aligned=t-t[k]
 if aligned[0]>-.3 or aligned[-1]<.3:human_exclusions['temporal_window']+=1;continue
 c=curvature_mm(xy,t);human_curves[part].append(np.interp(grid,aligned,c))
assert len(human_trials)==2400 and len(human_max)==20
# A-E: same millimetre units, 1mm tolerance, five-frame pad and first passage.
library=json.loads(source_paths[8].read_text())['options'];seeds=[7,101,202,1001]
curves={(seed,name):defaultdict(list) for seed in seeds for name in 'ABCDE'};maxima={(seed,name):defaultdict(list) for seed in seeds for name in 'ABCDE'};counts={(seed,name):defaultdict(int) for seed in seeds for name in 'ABCDE'};events=[]
for p in track(sorted(raw.glob('seed*_order*.npz')),description='Paper-method base curvature at A-E'):
 source_paths.extend([p,p.with_suffix('.json')]);a=np.load(p);meta=json.loads(p.with_suffix('.json').read_text());seed=meta['policy'];order=meta['order']
 for posture in range(2):
  episode=meta['result'][posture];end=round(episode['completion_s']/.01) if episode['success'] else len(a['xy'])-1
  xy=a['xy'][:end+1,posture].astype(float)*1000;t=np.arange(len(xy))*.01;c=curvature_mm(xy,t)
  for option in library:
   name=option['id'][-1];key=(seed,name);counts[key]['attempted']+=1;result,reason=region(xy,t,np.array(option['xy_m'])*1000,float(option['termination_radius_m'])*1000)
   if reason:counts[key][reason]+=1
   else:
    maxima[key][order].append(result['maximum']);counts[key]['regional_episodes']+=1;events.append(dict(policy=seed,order=order,posture=posture,subgoal=name,**result))
   inside=np.linalg.norm(xy-np.array(option['xy_m'])*1000,axis=1)<=float(option['termination_radius_m'])*1000+1
   hits=np.flatnonzero(inside & np.r_[False,~inside[:-1]]);windows=[];counts[key]['temporal_entries']+=len(hits)
   for k in hits:
    if k<30 or k+30>=len(xy):counts[key]['temporal_window_exclusion']+=1;continue
    windows.append(c[k-30:k+31]);counts[key]['eligible_temporal_entries']+=1
   if windows:
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);average=np.nanmean(windows,axis=0)
    curves[key][order].append(average);counts[key]['temporal_episodes']+=1
with warnings.catch_warnings():
 warnings.simplefilter('ignore',RuntimeWarning)
 human_array=np.array([np.nanmedian(v,axis=0) for v in human_curves.values()]);human_scalar=np.array([np.mean(v) for v in human_max.values()])
 arrays={'Human':human_array,'Human_region_max':human_scalar};metrics={}
 for key in curves:
  seed,name=key;v=np.array([np.nanmedian(rows,axis=0) for rows in curves[key].values()]);scalar=np.array([np.mean(rows) for rows in maxima[key].values()]);arrays[f'{seed}_{name}']=v;arrays[f'{seed}_{name}_region_max']=scalar
  metrics[f'{seed}_{name}']=dict(counts[key],temporal_orders=len(v),regional_orders=len(scalar),median_order_mean_region_max_m_inv=float(np.median(scalar)) if len(scalar) else None,median_curve_peak_m_inv=float(np.nanmax(np.nanmedian(v,axis=0))) if len(v) else None)
np.savez_compressed(out/'aggregates.npz',**arrays)
repro=dict(trials=len(reproduction),max_abs_error_m_inv=max(v['abs_difference'] for v in reproduction),matched_within_1e_minus_6=sum(v['abs_difference']<1e-6 for v in reproduction))
report=dict(method=dict(paper='Eq10, Methods Curvature p6; Figs6-7 use regional maxima',author_code='analyses.py curvature_one_traj / curvature_at_target',coordinate_units='mm; convert output mm^-1 to m^-1 by multiplying1000',derivatives='np.gradient positions and velocity using native timestamps; no added Gaussian filter or speed floor',filter='Author |signed curvature|<10mm^-1; nonfinite discarded',region='First entry and first exit; pad5 native frames; radius+1mm as author Circle.__call__',guards='Regional maxima exclude first entry<5frames, missing exits or no retained samples; counts explicit',time_profiles='Descriptive extension: whole-trial Eq10 derivatives, magnitudes, 100Hz display resampling for human. Base all outside-to-inside entries; +/-300ms window; mean repeated windows within episode then median within/across orders. Not paper classification assay.',scope='192 base episodes; A-E regions differ from human targets; no claim that peaks alone establish hierarchy'),human_reproduction=repro,human_exclusions=dict(human_exclusions),human_curve_participants=len(human_array),human_median_participant_mean_region_max_m_inv=float(np.median(human_scalar)),metrics=metrics,events=events,source_hashes={str(p):sha(p) for p in source_paths})
font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'));plt.rcParams.update({'font.family':'EB Garamond','font.size':11,'axes.spines.top':False,'axes.spines.right':False});colors={7:'#777777',101:'#D55E00',202:'#009E73',1001:'#CC79A7'}
def panel(ax,name):
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);lo,med,hi=np.nanpercentile(human_array,[25,50,75],axis=0)
 ax.fill_between(grid*1000,lo,hi,color='#0072B2',alpha=.15);ax.plot(grid*1000,med,color='#0072B2',lw=2,label='Human: author-preprocessed trajectories')
 for seed in seeds:
  v=arrays[f'{seed}_{name}']
  if not len(v):continue
  with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(v,axis=0)
  ax.plot(grid*1000,med,color=colors[seed],ls='--',lw=1.3,label=f'Policy {seed}')
 count='/'.join(str(metrics[f'{seed}_{name}'].get('temporal_episodes',0)) for seed in seeds)
 ax.axvline(0,color='black',ls=':',lw=.8);ax.set(title=f'Landmark {name}; episodes 7/101/202/1001: {count}',xlabel='Time from region entry (ms)',ylabel=r'Curvature (m$^{-1}$)',xlim=(-300,300),yscale='log',ylim=(.5,10000));ax.title.set_fontsize(10)
fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
for name,ax in zip('ABCDE',axes.flat):panel(ax,name)
axes.flat[5].axis('off');h=[Line2D([],[],color='#0072B2',lw=2)]+[Line2D([],[],color=colors[seed],ls='--') for seed in seeds];l=['Human: author-preprocessed trajectories']+[f'Policy {seed}' for seed in seeds];axes.flat[5].legend(h,l,loc='upper left',frameon=False)
axes.flat[5].text(0,.45,'Paper Eq10 and author derivatives.\nNo extra smoothing or speed floor.\nMagnitudes; log scale.\nHuman: second target region.\nBase: all eligible A-E entries, averaged\nwithin episode, then median by order.\nTemporal plots extend the paper method;\nits reported outcome is regional maximum.',transform=axes.flat[5].transAxes,va='top')
fig.savefig(out/'all_subgoal_curvature.png',dpi=600,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,axes=plt.subplots(1,5,figsize=(13,4.2),layout='constrained')
all_scalar=np.concatenate([human_scalar]+[arrays[f'{seed}_{name}_region_max'] for seed in seeds for name in 'ABCDE']);limits=(np.min(all_scalar[all_scalar>0])*.7,np.max(all_scalar)*1.4)
for name,ax in zip('ABCDE',axes):
 values=[human_scalar]+[arrays[f'{seed}_{name}_region_max'] for seed in seeds];cs=['#0072B2']+[colors[seed] for seed in seeds]
 for index,(v,color) in enumerate(zip(values,cs)):
  if not len(v):continue
  ax.scatter(np.full(len(v),index)+np.linspace(-.12,.12,len(v)),v,color=color,alpha=.4,s=12);lo,med,hi=np.percentile(v,[25,50,75]);ax.errorbar(index,med,yerr=[[med-lo],[hi-med]],fmt='o',color=color,capsize=3,markersize=5)
 labels=['Human\nn=20']+[f'{seed}\nn={len(arrays[f"{seed}_{name}_region_max"])}' for seed in seeds]
 ax.set(title=f'Landmark {name}',xticks=range(5),xticklabels=labels,yscale='log',ylim=limits,ylabel=r'Region maximum curvature (m$^{-1}$)' if name=='A' else '',xlabel='Policy');ax.tick_params(axis='x',labelrotation=45)
fig.text(.5,-.045,'Dots: human participant means or base order means. Large dots/IQR across groups. Same numerical rule; different region geometry.',ha='center',fontsize=10)
fig.savefig(out/'paper_region_maxima.png',dpi=600,bbox_inches='tight',facecolor='white');plt.close(fig)
for name in 'ABCDE':
 fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained');panel(ax,name);ax.legend(frameon=False,fontsize=9);fig.savefig(out/f'subgoal_{name}_curvature.png',dpi=600,bbox_inches='tight',facecolor='white');plt.close(fig)
report['figure_hashes']={p.name:sha(p) for p in out.glob('*.png')};(out/'results.json').write_text(json.dumps(report,indent=2));(out/'human_reproduction.json').write_text(json.dumps(reproduction,indent=2))
print(json.dumps(dict(human_reproduction=repro,human_median_participant_mean_region_max=report['human_median_participant_mean_region_max_m_inv'],base_medians={k:v['median_order_mean_region_max_m_inv'] for k,v in metrics.items()}),indent=2))
