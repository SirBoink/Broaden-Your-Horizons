"""Plot controlled-launch kinematics and summarize paired transfer records."""
from pathlib import Path
import json
import hashlib
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/figures'
BUILD = ROOT / 'data/summary'
OUT.mkdir(parents=True, exist_ok=True)
FONT = ROOT / 'figures/style/fonts/EBGaramond-Regular.ttf'
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': 'EB Garamond', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 600, 'mathtext.fontset': 'cm'})
COLORS = ['#0072B2', '#D55E00', '#56B4E9', '#E69F00', '#009E73', '#CC79A7']
inputs = {Path(__file__).resolve()}

def read(path):
    p = ROOT / path
    inputs.add(p)
    return json.loads(p.read_text(encoding='utf-8'))

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def panel(directory):
    return [r for p in sorted((ROOT/directory).glob('*.json'))
            for r in read(str(p.relative_to(ROOT)))['rows'] if r['controller']=='base']

historical = panel('data/mechanism/historical_transfer')
historical += panel('data/single_discount/single_transfer')
assert len(historical)==720
small = list(range(16,24))  # panel generator appends 16 large-transition orders, then 8 small.
panel_config = read('data/mechanism/panel.json')
assert len(panel_config['groups']['large'])==16 and len(panel_config['groups']['small'])==8
methods = ['base','single','learned']
data = {}
for name, orders in [('full',list(range(24))),('small',small)]:
    selected = {m:[r for r in historical if r['method']==m and r['order'] in orders] for m in methods}
    assert all(len(v)==len(orders)*10 for v in selected.values())
    data[name] = {m:{'n':len(v),'successes':sum(r['success'] for r in v),
                     'completion_pct':100*np.mean([r['success'] for r in v]),
                     'restricted_s':np.mean([r['restricted_completion_s'] for r in v]),
                     'by_order_pct':[100*np.mean([r['success'] for r in v if r['order']==o]) for o in orders]}
                  for m,v in selected.items()}
    data[name]['orders']=orders
    for comparator in ['base','single']:
        effects=np.array(data[name]['learned']['by_order_pct'])-np.array(data[name][comparator]['by_order_pct'])
        rng=np.random.default_rng(42)
        ci=np.quantile(rng.choice(effects,(10000,len(orders))).mean(axis=1),[.025,.975])
        data[name]['vs_'+comparator]={'effect_pp':float(effects.mean()),'ci95_pp':ci.tolist()}
    times={m:np.array([np.mean([r['restricted_completion_s'] for r in selected[m] if r['order']==o]) for o in orders]) for m in methods}
    rng=np.random.default_rng(42)
    sampled=rng.integers(0,len(orders),size=(10000,len(orders)))
    reductions=100*(1-times['learned'][sampled].mean(axis=1)/times['base'][sampled].mean(axis=1))
    data[name]['time_reduction_vs_base']={'pct':float(100*(1-times['learned'].mean()/times['base'].mean())),
                                         'ci95_pct':np.quantile(reductions,[.025,.975]).tolist()}
assert data['small']['learned']['successes']==74 and data['full']['learned']['successes']==204

term = ROOT/'data/handoffs/controlled_launch'
inputs.update([term/'option_termination_kinematics.npz',term/'option_termination_muscle_activations.npz'])
z=np.load(term/'option_termination_kinematics.npz')
a=np.load(term/'option_termination_muscle_activations.npz')
meta=read('data/handoffs/controlled_launch/option_termination_kinematics.json')
names=['Pect','Delt','Brach','TriLat','Bic','TriLong']
states=[]
for o,trial in zip('ABCDE',meta['trials']):
    tm=z[o+'_time_from_termination_ms']; q=z[o+'_joint_window']; c=z[o+'_cartesian_window']; act=a[o+'_activation_window']; i=int(np.argmin(abs(tm)))
    assert tm[i]==0 and trial['reason']=='subgoal_reached'
    states.append(act[i])
    fig,axs=plt.subplots(2,2,figsize=(8.3,4.9),constrained_layout=True)
    for j in range(6): axs[0,0].plot(tm,act[:,j],color=COLORS[j],label=names[j],lw=1.7)
    axs[0,0].set(ylabel='Muscle activation (0–1)',ylim=(0,1.05)); axs[0,0].legend(ncol=3,fontsize=10,frameon=False)
    for j,label in enumerate(['Shoulder','Elbow']):
        axs[0,1].plot(tm,np.rad2deg(q[:,j]),label=label,color=COLORS[j],lw=1.7)
        axs[1,0].plot(tm,q[:,j+2],label=label,color=COLORS[j],lw=1.7)
    axs[0,1].set(ylabel='Joint angle (deg)',ylim=(30,115)); axs[0,1].legend(frameon=False,fontsize=10)
    axs[1,0].set(ylabel='Joint velocity (rad/s)',ylim=(-6,11))
    for j,label in enumerate([r'$v_x$',r'$v_y$']): axs[1,1].plot(tm,c[:,j+2],label=label,color=COLORS[j],lw=1.7)
    axs[1,1].plot(tm,np.linalg.norm(c[:,2:],axis=1),color='#222222',label='Speed',ls='--',lw=1.7)
    axs[1,1].set(ylabel='Fingertip velocity (m/s)',ylim=(-2.1,2.1)); axs[1,1].legend(frameon=False,ncol=3,fontsize=10)
    for ax in axs.flat:
        ax.axvline(0,color='#444444',ls=':',lw=1); ax.set(xlim=(-50,50),xlabel='Time from termination (ms)')
        if o=='E': ax.axvspan(-50,-40,color='#eeeeee')
    save(fig,'termination_'+o)

fig,ax=plt.subplots(figsize=(6.6,3.6),constrained_layout=True)
im=ax.imshow(states,vmin=0,vmax=1,cmap='Blues',aspect='auto'); ax.set_xticks(range(6),names); ax.set_yticks(range(5),list('ABCDE'))
for i in range(5):
    for j in range(6): ax.text(j,i,f'{states[i][j]:.2f}',ha='center',va='center',color='white' if states[i][j]>.65 else 'black')
fig.colorbar(im,ax=ax,label='Activation (0–1)'); save(fig,'termination_heatmap')

options=read('data/discovery/options/off_target_band_options.json')['options']
fig,axs=plt.subplots(1,2,figsize=(8,3.7),constrained_layout=True)
xy=np.array([o['xy_m'] for o in options]); axs[0].scatter(xy[:,0]*1000,xy[:,1]*1000,color='#FF0066',s=45)
for label,p in zip('ABCDE',xy): axs[0].annotate(label,p*1000,xytext=(6,5),textcoords='offset points')
axs[0].set(xlabel='Hand x (mm)',ylabel='Hand y (mm)'); axs[0].set_aspect('equal',adjustable='datalim')
horizons=[.01/-np.log(o['gamma'])*1000 for o in options]
axs[1].bar(list('ABCDE'),horizons,color='#0072B2'); axs[1].set(yscale='log',ylabel='Nominal discount horizon (ms)')
save(fig,'landmarks')
alternate=read('data/discovery/controls/seed_1001_discovery.json')
other=np.array([c['xy_m'] for c in alternate['candidates']])
dist=np.linalg.norm(other[:,None,:]-xy[None,:,:],axis=2).min(axis=1)*1000
data['alternate_distances_mm']=dist.tolist()
assert len(other)==4
fig,ax=plt.subplots(figsize=(6.4,4),constrained_layout=True)
ax.scatter(xy[:,0]*1000,xy[:,1]*1000,color='#FF0066',label='Canonical A–E',s=40)
ax.scatter(other[:,0]*1000,other[:,1]*1000,color='#0072B2',marker='D',label='Alternate-sequence cache',s=40)
for label,point in zip('ABCDE',xy): ax.annotate(label,point*1000,xytext=(5,4),textcoords='offset points')
for j,point in enumerate(other): ax.annotate('S'+str(j+1),point*1000,xytext=(5,5 if j==1 else -11),textcoords='offset points')
ax.set(xlabel='Hand x (mm)',ylabel='Hand y (mm)'); ax.set_aspect('equal',adjustable='datalim'); ax.legend(frameon=False,fontsize=11)
save(fig,'alternate_landmarks')

corr=read('data/mechanism/human/ae_human_correlations.json')
fig,ax=plt.subplots(figsize=(7.8,4.0),constrained_layout=True)
for pi,passage in enumerate(['first','second']):
    means=[]; err=[]
    for j,o in enumerate('ABCDE'):
        vals=np.array([r[passage+'_mean_correlation'] for r in corr if r['candidate']==o]); assert len(vals)==20
        x=j+(pi-.5)*.24; ax.scatter(np.full(20,x),vals,s=10,color=COLORS[pi],alpha=.35)
        means.append(vals.mean()); err.append(t.ppf(.975,19)*vals.std(ddof=1)/np.sqrt(20))
    ax.errorbar(np.arange(5)+(pi-.5)*.24,means,yerr=err,fmt='o',color=COLORS[pi],capsize=4,label=passage.title()+' passage')
ax.axhline(0,color='#aaaaaa',lw=.8); ax.set(xticks=range(5),xticklabels=list('ABCDE'),ylim=(-1,1),ylabel='Participant mean Pearson r'); ax.legend(frameon=False)
save(fig,'human_correlations')
data['human']=read('data/summary/recomputed_metrics.json')['human_descriptive']
data['usage']=read('data/summary/revision_evidence.json')['option_bout_accounting']
data['fine']=read('data/summary/recomputed_metrics.json')['fine']
data['geometry']=read('data/summary/recomputed_metrics.json')['geometry_completion']
data['policy_metadata']=read('data/summary/revision_evidence.json')['policy_metadata']
data['neural']=read('data/mechanism/neural/summary.json')
data['endpoints']=read('data/summary/termination_evidence.json')['termination']

equations={
 'features':[r'$\phi_t=[x_t,y_t,v_{x,t},v_{y,t},a_{1,t},\ldots,a_{6,t}]^{\mathsf{T}}$',r'$\widetilde{\phi}_{t,d}=(\phi_{t,d}-\mu_d)/\sigma_d$',r'$\dot{a}_m=(u_m-a_m)/\tau_m(u_m,a_m)$'],
 'laplace':[r'$\widehat{\psi}_{\gamma}(t)=\sum_{k=0}^{T-1-t}\gamma^k\widetilde{\phi}_{t+k},\quad s=-\log\gamma,\quad\lambda=s/\Delta t$',r'$B_i(t)=\frac{\widehat{\psi}_{\gamma_{i+1}}(t)-\widehat{\psi}_{\gamma_i}(t)}{|s_{i+1}-s_i|}\approx-\partial_s\psi_s(t)$',r'$-\partial_s\psi_s(t)=\sum_k k e^{-sk}\widetilde{\phi}_{t+k}$',r'$C_i(t)=1-\frac{B_i(t-1)^{\mathsf{T}}B_i(t)}{\|B_i(t-1)\|\,\|B_i(t)\|}$'],
 'ddqn':[r'$Q_\theta:\mathbb{R}^{30}\longrightarrow\mathbb{R}^{6}$',r'$o^*=\arg\max_{o\in\mathcal{A}(s^\prime)}Q_\theta(s^\prime,o)$',r'$y=\sum_{j=0}^{d-1}\gamma^j r_{t+j}-c_{delib}+1_{\neg terminal}\gamma^d Q_{\bar\theta}(s^\prime,o^*)$'],
 'mechanics':[r'$v=J(q)\dot q$',r'$M(q)\ddot q+c_{skel}(q,\dot q)=-R_{geom}(q)F_m-D_{eff}\dot q+\tau_{joint}+J(q)^{\mathsf{T}}f_{endpoint}$',r'$R_{jm}=\partial\ell_m/\partial q_j,\qquad F_m=F_m(a,\ell,\dot\ell)$']}
for name,lines in equations.items():
    fig=plt.figure(figsize=(9,2.3 if name!='laplace' else 3));
    for i,line in enumerate(lines): fig.text(.015,.86-i*(.72/max(1,len(lines)-1)),line,fontsize=17 if name!='mechanics' else 14)
    save(fig,'equation_'+name)
data['manifest']={'script':'scripts/plot_handoff_traces.py','sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}}
data['manifest']['figures']={}
for figure in OUT.glob('*.png'):
    if figure.stem.startswith('termination'):
        sources=['data/handoffs/controlled_launch/option_termination_kinematics.npz','data/handoffs/controlled_launch/option_termination_muscle_activations.npz']
    elif figure.stem=='human_correlations':
        sources=['data/mechanism/human/ae_human_correlations.json']
    elif figure.stem=='alternate_landmarks':
        sources=['data/discovery/controls/seed_1001_discovery.json','data/discovery/options/off_target_band_options.json']
    elif figure.stem.startswith('equation'):
        sources=['formulas.py','SAC/high_level.py','docs/references/motornet/muscle.py','docs/references/motornet/effector.py','docs/references/motornet/skeleton.py']
    else:
        sources=['data/discovery/options/off_target_band_options.json']
    data['manifest']['figures'][figure.name]={'script':data['manifest']['script'],'sources':sources,'sha256':hashlib.sha256(figure.read_bytes()).hexdigest(),'dpi':600,'commit':None}
(BUILD/'evidence.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(OUT/'figure_manifest.json').write_text(json.dumps(data['manifest'],indent=2),encoding='utf-8')
print('Verified 720 historical rows; regenerated A–E physiology, landmarks, human correlations and equation PNGs at 600 dpi.')

