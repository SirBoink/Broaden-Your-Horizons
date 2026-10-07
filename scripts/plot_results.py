"""Plot transfer, handoff and human-profile results from saved observations."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Circle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
BUILD=ROOT/'data/summary'
OUT=ROOT/'results/figures'; OUT.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'))
plt.rcParams.update({'font.family':'EB Garamond','font.size':13,'axes.spines.top':False,
                    'axes.spines.right':False,'mathtext.fontset':'cm','legend.frameon':False})
COLORS=['#D55E00','#0072B2','#CC79A7','#56B4E9','#E69F00']
METHODS=['base','single','learned']; MC=['#777777','#0072B2','#B52F62']
sources={Path(__file__)};figures=[];audit={}


def read(name):
    p=ROOT/name;sources.add(p)
    return json.loads(p.read_text(encoding='utf-8'))


def arrays(name):
    p=ROOT/name;sources.add(p)
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}


def save(fig,name):
    p=OUT/(name+'_v15.png');fig.savefig(p,dpi=600,facecolor='white',bbox_inches='tight')
    plt.close(fig);figures.append(p)


d=read('data/summary/evidence.json')
kin=arrays('data/handoffs/controlled_launch/option_termination_kinematics.npz')
act=arrays('data/handoffs/controlled_launch/option_termination_muscle_activations.npz')
tm=act['time_from_termination_ms'];zero=int(np.flatnonzero(tm==0)[0]);assert zero==5
fig,axes=plt.subplots(1,3,figsize=(11.4,3.8))
fig.subplots_adjust(left=.07,right=.985,bottom=.23,top=.78,wspace=.42)
for name,color in zip('ABCDE',COLORS):
    joint=kin[name+'_joint_window'];cart=kin[name+'_cartesian_window']
    for ax,values in zip(axes,[np.rad2deg(joint[:,:2]),joint[:,2:],cart[:,2:]]):
        ax.plot(*values[:zero+1].T,color=color,lw=2,label=name)
        ax.plot(*values[zero:].T,color=color,lw=1.7,ls='--')
        ax.scatter(*values[0],s=28,facecolors='white',edgecolors=color,zorder=4)
        ax.scatter(*values[zero],s=48,color=color,edgecolors='black',lw=.4,zorder=5)
        ax.set_box_aspect(1)
for ax,title,x,y in zip(axes,['Posture','Joint motion','Hand motion'],
        ['Shoulder angle (deg)','Shoulder velocity (rad/s)','Horizontal velocity (m/s)'],
        ['Elbow angle (deg)','Elbow velocity (rad/s)','Vertical velocity (m/s)']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.margins(.16);ax.tick_params(labelsize=11)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,bbox_to_anchor=(.5,1.02))
save(fig,'phase_planes')

# Distinguish single controlled launch from actual selected arrivals. Average within episode first.
endpoints=read('data/handoffs/actual_endpoints.json')
groups=defaultdict(list)
for r in endpoints:
    if r['actual_endpoint'] is not None:
        groups[(r['choice'],r['policy_seed'],r['order'],r['context_seed'],r['posture'])].append(r['actual_endpoint']['activation'])
example=np.stack([act[o+'_activation_window'][zero] for o in 'ABCDE'])
means=[];counts=[]
for option in range(1,6):
    rows=[np.mean(v,axis=0) for key,v in groups.items() if key[0]==option]
    means.append(np.mean(rows,axis=0));counts.append(len(rows))
means=np.stack(means);assert means.shape==example.shape==(5,6)
fig,axes=plt.subplots(1,2,figsize=(10.8,3.9),layout='constrained')
for ax,values,title,labels in zip(axes,[example,means],['Controlled launch: one arrival per option','DDQN-selected arrivals: episode means'],
        [list('ABCDE'),[f'{o}  (n={n})' for o,n in zip('ABCDE',counts)]]):
    im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set(yticks=range(5),yticklabels=labels,xticks=range(6),xticklabels=['Pect','Delt','Brach','TriLat','Bic','TriLong'],title=title)
    for i in range(5):
        for j in range(6):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>.55 else 'black',fontsize=12)
fig.colorbar(im,ax=axes,shrink=.86,label='Activation (0–1)')
save(fig,'muscle_assays');audit['muscle_assays']={'controlled_launch':example.tolist(),'actual_episode_means':means.tolist(),'option_episode_counts':counts,'actual_invocations':sum(len(v) for v in groups.values()),'averaging':'Repeated invocations averaged within option/episode; episode means pooled across three policies.'}

# Historical paired trial outcomes. Same outcomes, clearer rates, order variation and uncertainty.
rows=[]
for folder in ['data/mechanism/historical_transfer','data/single_discount/single_transfer']:
    for p in sorted((ROOT/folder).glob('*.json')):rows.extend(r for r in read(str(p.relative_to(ROOT)))['rows'] if r['controller']=='base')
assert len(rows)==720
rng=np.random.default_rng(42)
for group,orders in [('small',list(range(16,24))),('full',list(range(24)))]:
    fig,axes=plt.subplots(1,2,figsize=(10.7,3.7),layout='constrained',gridspec_kw={'width_ratios':[1,1.2]})
    sampled=rng.integers(len(orders),size=(10000,len(orders)))
    for j,(method,color,label) in enumerate(zip(METHODS,MC,['Base SAC','Single discount','DDQN + A–E'])):
        rates=np.array([np.mean([r['success'] for r in rows if r['method']==method and r['order']==o])*100 for o in orders])
        lo,hi=np.quantile(rates[sampled].mean(1),[.025,.975]);mean=rates.mean()
        axes[0].scatter(j+np.linspace(-.13,.13,len(orders)),rates,color=color,alpha=.30,s=22)
        axes[0].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,ms=9,lw=2,capsize=5)
        axes[0].text(j,1.04,f"{mean:.2f}%\n{d[group][method]['successes']}/{d[group][method]['n']}",transform=axes[0].get_xaxis_transform(),ha='center',color=color,fontsize=13)
    for j,(comparator,label) in enumerate([('base','DDQN − base'),('single','DDQN − single')]):
        effect=d[group]['vs_'+comparator];mean=effect['effect_pp'];lo,hi=effect['ci95_pp']
        axes[1].errorbar(mean,1-j,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#B52F62',ms=8,lw=2,capsize=4)
        axes[1].text(mean,1-j+.17,f'+{mean:.2f} pp  [{lo:.2f}, {hi:.2f}]',ha='center',fontsize=12)
    axes[0].set(xticks=range(3),xticklabels=['Base','Single','DDQN'],ylabel='Sequence completion (%)',ylim=(-5,105))
    axes[0].set_title('Order means and 95% order-bootstrap intervals',pad=46)
    axes[1].axvline(0,color='black',ls=':',lw=1)
    axes[1].set(yticks=[1,0],yticklabels=['Versus base','Versus single'],ylim=(-.45,1.6),xlabel='Paired completion gain (percentage points)',title='Gain using the same starting states',xlim=(-5,80 if group=='small' else 40))
    save(fig,'transfer_'+group)
    selected={m:{(r['order'],r['seed'],r['posture']):r for r in rows if r['method']==m and r['order'] in orders} for m in METHODS}
    assert selected['base'].keys()==selected['learned'].keys()
    audit[group+'_paired']={'converted_failures':sum(not b['success'] and selected['learned'][k]['success'] for k,b in selected['base'].items()),'lost_successes':sum(b['success'] and not selected['learned'][k]['success'] for k,b in selected['base'].items()),'n':len(selected['base'])}

fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),layout='constrained')
for m,c,label in zip(METHODS,MC,['Base','Single','DDQN']):
    axes[0].plot(range(16,24),d['small'][m]['by_order_pct'],'o-',color=c,label=label,lw=1.8)
for comp,color,label in [('base','#777777','Versus base'),('single','#0072B2','Versus single')]:
    axes[1].plot(range(16,24),np.array(d['small']['learned']['by_order_pct'])-d['small'][comp]['by_order_pct'],'o-',color=color,label=label)
axes[0].set(ylabel='Completion (%)',ylim=(-4,105),title='Every subgroup order retained');axes[1].set(ylabel='DDQN paired gain (pp)',title='Seven positive gains versus base; one ceiling tie');axes[1].axhline(0,color='black',ls=':',lw=.8)
for ax in axes:ax.set(xlabel='Target order ID',xticks=range(16,24));ax.legend(fontsize=11)
save(fig,'order_gains')

fig,ax=plt.subplots(figsize=(7.6,3.7),layout='constrained')
values=[d['small'][m]['restricted_s'] for m in METHODS];bars=ax.bar(['Base SAC','Single discount','DDQN + A–E'],values,color=MC,edgecolor='black',lw=.6)
for j,(bar,value) in enumerate(zip(bars,values)):
    reduction=100*(1-value/values[0]);label=f'{value:.3f} s\n'+('Reference' if j==0 else f'−{reduction:.1f}% vs base')
    ax.text(bar.get_x()+bar.get_width()/2,value+.16,label,ha='center',fontsize=14)
ax.set(ylabel='Failure-capped completion time (s)',ylim=(0,7.5));save(fig,'completion_time')

fig,ax=plt.subplots(figsize=(7.6,3.6),layout='constrained')
vals=[100*d['usage'][k]['option_bout_fraction'] for k in ['A','B']]
bars=ax.bar(['Sequence A','Sequence B'],vals,color='#B52F62',edgecolor='black',lw=.6)
ax.bar_label(bars,labels=[f'{v:.2f}%\n{n}' for v,n in zip(vals,['122/262 segments','127/270 segments'])],padding=6,fontsize=14)
ax.set(ylabel='Option invocations / all control segments (%)',ylim=(0,70));save(fig,'option_segments')

fig,ax=plt.subplots(figsize=(8.4,3.5),layout='constrained')
values=[d['full'][m]['completion_pct'] for m in ['single','learned']]
bars=ax.bar(['Selected single-discount pipeline','Multiscale DDQN pipeline'],values,color=MC[1:],edgecolor='black',lw=.6)
ax.bar_label(bars,labels=['70.42% · 169/240','85.00% · 204/240'],padding=6,fontsize=14)
ax.set(ylabel='Sequence completion (%)',ylim=(0,105),title='Same 240 evaluation starts; different discovery and scheduling')
save(fig,'pipeline_comparison')

fig,ax=plt.subplots(figsize=(8.1,3.8),layout='constrained')
keys=['base','multiscale','single','learned'];values=[100*d['fine'][k]['successes']/d['fine'][k]['n'] for k in keys]
bars=ax.bar(['Base','Fixed A–E','Single','DDQN'],values,color=['#777777','#99B8C9','#0072B2','#B52F62'],edgecolor='black',lw=.6)
for i,(bar,value) in enumerate(zip(bars,values)):
    ax.text(bar.get_x()+bar.get_width()/2,value+.8,f'{value:.2f}%\n'+('Reference' if i==0 else f'+{value-values[0]:.2f} pp vs base'),ha='center',fontsize=12)
ax.set(ylabel='Sequence completion (%)',ylim=(0,45));save(fig,'fine_transfer')

readouts=read('data/summary/readout_reaudit.json')['by_fit']
fig,ax=plt.subplots(figsize=(9.8,3.5),layout='constrained')
for r in readouts:
    vals=[r[k]*1e6 for k in ['current','multi','single']]
    ax.plot(range(3),vals,'o-',color='#777777',alpha=.5,lw=.9,ms=4)
means=np.array([[r[k]*1e6 for k in ['current','multi','single']] for r in readouts]).mean(0)
ax.scatter(range(3),means,color=['#777777','#B52F62','#0072B2'],s=60,zorder=5)
for i,v in enumerate(means):ax.text(i,v+55,f'{v:.1f} mm²',ha='center',fontsize=13)
ax.set(xticks=range(3),xticklabels=['Current-state readout','Multiscale readout','Selected single-discount readout'],ylabel='Displacement prediction MSE (mm²)',ylim=(0,means.max()*1.3),title='Five paired fits; all three feature sets retained')
save(fig,'readout_comparison')

# Policy heterogeneity stays visible. Remove spaghetti, add exact counts and effect labels.
policy_audit=read('data/summary/transfer_audit_v12.json')
fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),layout='constrained')
for j,r in enumerate(policy_audit):
    seed=r['policy'];rates=r['rates_pct'];c=['#0072B2','#D55E00','#777777'][j]
    axes[0].plot(rates,[j,j],'o-',color=c,lw=2,ms=8)
    for rate,label in zip(rates,['Base','DDQN']):axes[0].annotate(f'{label}: {rate:.2f}%',(rate,j),xytext=(0,10 if label=='DDQN' else -18),textcoords='offset points',ha='center',fontsize=11)
    mean=r['gain_pp'];lo,hi=r['order_bootstrap_95CI_pp']
    axes[1].errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=c,ms=8,capsize=4)
    axes[1].text(38,j,f'+{mean:.2f} pp',va='center',fontsize=12)
for ax in axes:ax.set(yticks=range(3),yticklabels=[f'Policy {r["policy"]}' for r in policy_audit],ylim=(2.7,-.65))
axes[0].set(xlim=(15,105),xlabel='Sequence completion (%)',title='Same shared selector and executor')
axes[1].axvline(0,color='black',ls=':',lw=.8);axes[1].set(xlim=(-10,51),xlabel='DDQN − base completion (pp)',title='95% whole-order bootstrap intervals')
save(fig,'policy_transfer')
contrasts=read('data/replay/selected_states/saved_data_audit.json')['contrasts']
fig,axes=plt.subplots(1,3,figsize=(11.3,3.6),layout='constrained')
for ax,comp,title in zip(axes,['base','mirror','other_mean'],['Continue base','Reflected waypoint','Other A–E options']):
    for j,seed in enumerate([7,101,202]):
        r=next(r for r in contrasts if r['policy_seed']==seed and r['comparator']==comp and r['metric']=='success')
        mean=r['mean']*100;lo,hi=np.array(r['exploratory_order_bootstrap_95CI'])*100
        c=['#0072B2','#D55E00','#777777'][j]
        ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=c,capsize=3)
        ax.text(22,j,f'{mean:+.2f}',va='center',fontsize=11)
    ax.axvline(0,color='black',ls=':',lw=.8);ax.set(yticks=range(3),yticklabels=['Policy 7','Policy 101','Policy 202'],ylim=(2.5,-.5),xlim=(-17,33),xlabel='Chosen completion gain (pp)',title=title)
save(fig,'local_completion')

replay=read('data/controls/replay/replay_comparison.json')
fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
classification={}
for ax,dt,title in zip(axes,['0.01','0.0005'],['Historical Euler: 10 ms','Fine RK4: 0.5 ms']):
    counts=[]
    for option in range(1,6):
        v=np.array([r['chosen_minus_base_progress_500ms_m'][dt] for r in replay if r['metadata']['choice']==option])
        row=[int(sum(v < -1e-6)),int(sum(abs(v)<=1e-6)),int(sum(v > 1e-6))];counts.append(row)
        left=0
        for n,c in zip(row,['#D55E00','#DDDDDD','#0072B2']):
            ax.barh(option,n,left=left,color=c,edgecolor='white')
            if n:ax.text(left+n/2,option,str(n),ha='center',va='center',color='white' if c!='#DDDDDD' else 'black',fontsize=12)
            left+=n
    ax.set(yticks=range(1,6),yticklabels=list('ABCDE'),ylim=(5.7,.3),xlabel='Number of paired handoff starts',title=title,xlim=(0,30))
    classification[dt]=counts
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(['#D55E00','#DDDDDD','#0072B2'],['Chosen worse','Tied','Chosen better'])],loc='upper center',ncol=3,bbox_to_anchor=(.5,1.09))
save(fig,'handoff_outcomes');audit['local_classification']=classification

# Human correspondence: show measured correlations rather than imply equality of raw amplitudes.
corr=read('data/mechanism/human/ae_human_correlations.json')
fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
for ax,key,title,color in zip(axes,['first_mean_correlation','second_mean_correlation'],['Entry into first intermediate target','Entry into second intermediate target'],['#56B4E9','#0072B2']):
    for i,name in enumerate('ABCDE'):
        v=np.array([r[key] for r in corr if r['candidate']==name]);assert len(v)==20
        lo,hi=np.quantile(np.random.default_rng(42).choice(v,(10000,20)).mean(1),[.025,.975]);mean=v.mean()
        ax.scatter(i+np.linspace(-.12,.12,20),v,color=color,alpha=.3,s=14)
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],color=color,fmt='o',capsize=4,ms=7)
        ax.text(i,1.07,f'{mean:+.2f}',ha='center',color=color,fontsize=13)
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(5),xticklabels=list('ABCDE'),ylim=(-1.06,1.2),ylabel='Profile correlation r',xlabel='Predictive landmark',title=title)
save(fig,'human_passages')
controls=read('data/controls/human_controls.json')['results']
fig,axes=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained',gridspec_kw={'width_ratios':[.85,1.25]})
for i,(name,passage) in enumerate([('A','second'),('E','first')]):
    r=next(r for r in controls if r['candidate']==name and r['passage']==passage)
    v=np.array(list(r['participant_effects'].values()));mean=r['mean_correlation_advantage'];lo,hi=r['participant_bootstrap_95CI']
    axes[0].scatter(i+np.linspace(-.13,.13,len(v)),v,color=COLORS[0 if name=='A' else 4],alpha=.45,s=25)
    axes[0].errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color='black',ms=8,capsize=4)
    axes[0].text(i,.79,f'+{mean:.3f}\n[{lo:.3f}, {hi:.3f}]',ha='center',fontsize=13)
axes[0].axhline(0,color='black',lw=.8);axes[0].set(xticks=[0,1],xticklabels=['A · second','E · first'],ylim=(-.05,1.0),ylabel='Correlation advantage over matched controls',title='Motivated contrasts: all 20 participants')
for j,r in enumerate(controls):
    mean=r['mean_correlation_advantage'];lo,hi=r['participant_bootstrap_95CI'];c='#B52F62' if (r['candidate'],r['passage']) in [('A','second'),('E','first')] else '#777777'
    axes[1].errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=c,capsize=2,ms=5)
axes[1].axvline(0,color='black',lw=.8);axes[1].set(yticks=range(10),yticklabels=[f"{r['candidate']} · {r['passage']}" for r in controls],ylim=(9.7,-.7),xlabel='Correlation advantage (95% bootstrap interval)',title='All ten contrasts retained',xlim=(-.85,.8))
save(fig,'human_controls')

neural=read('data/mechanism/neural/summary.json')
heads=np.stack([r['head_nmse'] for r in neural['accuracy_by_fit']]);gamma=np.array(neural['gammas']);tau=-10/np.log(gamma)
fig,axes=plt.subplots(1,2,figsize=(10.6,3.7),layout='constrained')
for r in neural['accuracy_by_fit']:
    axes[0].plot(tau,r['head_nmse'],color='#0072B2',alpha=.3,lw=1)
axes[0].plot(tau,heads.mean(0),color='#0072B2',lw=2,label='Mean of five fits');axes[0].legend()
axes[0].set(xscale='log',yscale='log',xlabel='Discount time constant (ms)',ylabel='Normalized prediction error',title='Error rises with prediction horizon')
success=[r['success_mean_nmse'] for r in neural['accuracy_by_fit']];timeout=[r['timeout_mean_nmse'] for r in neural['accuracy_by_fit']]
for j,v in enumerate([success,timeout]):axes[1].scatter(j+np.linspace(-.05,.05,5),v,color=['#0072B2','#777777'][j],s=35)
axes[1].set(xticks=[0,1],xticklabels=['79 completed episodes','1 timeout episode'],yscale='log',ylabel='Mean normalized prediction error',title='Separate completed and failed episodes',xlim=(-.5,1.5),ylim=(.04,20))
axes[1].text(0,.09,f'{min(success):.3f}–{max(success):.3f}',ha='center');axes[1].text(1,9,f'{np.mean(timeout):.2f}',ha='center')
save(fig,'neural_prediction')

fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),layout='constrained')
v1=np.array([r['chosen_minus_base_progress_500ms_m']['0.001'] for r in replay])*1000
v2=np.array([r['chosen_minus_base_progress_500ms_m']['0.0005'] for r in replay])*1000
axes[0].scatter(v1,v2,color='#0072B2',alpha=.7,s=24);bounds=[min(v1.min(),v2.min())-10,max(v1.max(),v2.max())+10];axes[0].plot(bounds,bounds,'--',color='black',lw=1)
axes[0].set(xlabel='Gain at 1 ms RK4 (mm)',ylabel='Gain at 0.5 ms RK4 (mm)',title='500 ms local contrast agrees across fine steps')
err=np.array([r['RK4_1ms_vs_half_ms_max_position_error_m'] for r in replay])*1000
axes[1].scatter(range(1,81),err,color='#777777',s=19);axes[1].set(xlabel='Saved handoff case (all 80)',ylabel='Maximum trajectory discrepancy (mm)',title='One-second paths can still diverge')
axes[1].text(.05,.88,f'Maximum: {err.max():.2f} mm',transform=axes[1].transAxes)
save(fig,'numerical_agreement');audit['fine_agreement']={'max_500ms_gain_disagreement_mm':float(np.max(abs(v1-v2))),'max_one_second_trajectory_disagreement_mm':float(err.max())}

(OUT/'v15_figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(OUT/'v15_figure_manifest.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2),encoding='utf-8')
print(f'Generated {len(figures)} source-backed v15 assets; all comparator and policy outcomes retained.')
