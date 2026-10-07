"""Source-backed Matplotlib diagrams and plots for the annotated deck revision."""
from pathlib import Path
import json, hashlib, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import FancyBboxPatch, Circle

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
OUT=ROOT/'results/figures'; OUT.mkdir(parents=True, exist_ok=True)
BUILD=ROOT/'data/summary'
font_manager.fontManager.addfont(str(ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'))
plt.rcParams.update({'font.family':'EB Garamond','font.size':12,'mathtext.fontset':'cm','savefig.dpi':600,
                     'axes.spines.top':False,'axes.spines.right':False})
sources={Path(__file__)};outputs=[]
def read(name):
    p=ROOT/name;sources.add(p);return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    p=OUT/(name+'.png');fig.savefig(p,dpi=600,facecolor='white',bbox_inches='tight');plt.close(fig);outputs.append(p)
def canvas(w=9,h=3.6):
    fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,label,size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.01',facecolor='white',edgecolor='black',lw=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1,'color':'black'})

fig,ax=canvas()
steps=[('Recorded movements','Hand position, velocity\nand six activations'),('Future predictions','40 discounted\nfeature returns'),('Laplace derivatives','39 adjacent\nprediction bands'),('Candidate events','Temporal cosine changes\nacross connected bands'),('Landmark library','Filter targets and merge\nlocations into A–E'),('Hierarchical control','DDQN selects an option\nSAC executes the reach')]
positions=[(.015,.59),(.355,.59),(.695,.59),(.695,.10),(.355,.10),(.015,.10)]
for i,((title,detail),(x,y)) in enumerate(zip(steps,positions)):
    box(ax,x,y,.29,.26,f'{i+1}. {title}\n\n{detail}',12)
for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
    x,y=positions[a];u,v=positions[b]
    if y==v:arrow(ax,(x+(.29 if u>x else 0),y+.13),(u+(0 if u>x else .29),v+.13))
    else:arrow(ax,(x+.145,y),(u+.145,v+.26))
save(fig,'pipeline_annotated')

fig,ax=canvas(9,2.8)
xs=[.09,.35,.61,.88];counts=[5,6,6,6]
nodes=[]
for x,n in zip(xs,counts):
    ys=np.linspace(.2,.75,n);nodes.append(ys)
for j in range(3):
    for y in nodes[j]:
        for v in nodes[j+1]:ax.plot([xs[j]+.015,xs[j+1]-.015],[y,v],color='#b7b7b7',lw=.35,zorder=1)
for j,x in enumerate(xs):
    for y in nodes[j]:ax.add_patch(Circle((x,y),.017,facecolor='white',edgecolor='black',lw=.8,zorder=2))
    ax.text(x,.94,['30 observations','128 units','128 units','6 Q-values'][j],ha='center',fontsize=13)
    ax.text(x,.04,['State input','ReLU','ReLU','Base or A–E'][j],ha='center',fontsize=12)
for y,label in zip(nodes[3],['E','D','C','B','A','Base']):ax.text(.925,y,label,va='center',fontsize=11)
ax.text(.5,-.06,'Hidden layers show representative neurons. Dimensions are exact.',ha='center',fontsize=10)
save(fig,'ddqn_architecture_annotated')

fig,ax=canvas()
box(ax,.02,.40,.22,.22,'Current motor state\n30 observations')
box(ax,.34,.40,.23,.22,'Double DQN\nChoose base or A–E')
box(ax,.71,.69,.26,.19,'Base SAC\nContinue task sequence')
box(ax,.71,.13,.26,.25,'Universal SAC reacher\nReach chosen landmark\nthen return to base',12)
arrow(ax,(.24,.51),(.34,.51));arrow(ax,(.57,.56),(.71,.77));arrow(ax,(.57,.45),(.71,.25))
ax.text(.625,.72,'Base',ha='center',fontsize=11);ax.text(.625,.30,'A–E',ha='center',fontsize=11)
ax.plot([.84,.84,.12],[.13,.08,.08],color='black',lw=1);arrow(ax,(.12,.08),(.12,.40))
ax.text(.50,-.035,'New posture, velocity and activation feed the next decision',ha='center',fontsize=11)
save(fig,'execution_annotated')

equations={
'features_explained':[
('Motor features','Hand position and velocity, followed by six muscle activations.',r'$\phi_t=[x_t,y_t,v_{x,t},v_{y,t},a_{1,t},\ldots,a_{6,t}]^{\mathsf{T}}$'),
('Feature normalization','Each feature is centered and scaled using the source normalization.',r'$\widetilde\phi_{t,d}=(\phi_{t,d}-\mu_d)/\sigma_d$'),
('Activation dynamics','Excitation u drives activation a with a state-dependent time constant.',r'$\dot a_m=(u_m-a_m)/\tau_m(u_m,a_m)$')],
'laplace_explained':[
('Discounted future','Sum future motor features. A larger discount weights more distant events.',r'$\widehat\psi_\gamma(t)=\sum_{k=0}^{T-1-t}\gamma^k\widetilde\phi_{t+k},\quad s=-\log\gamma,\quad\lambda=s/\Delta t$'),
('Scale derivative','Adjacent discounts approximate how the predicted future changes with scale.',r'$B_i(t)=\frac{\widehat\psi_{\gamma_{i+1}}(t)-\widehat\psi_{\gamma_i}(t)}{|s_{i+1}-s_i|}\approx-\partial_s\psi_s(t)$'),
('Temporal weighting','Differentiation weights future samples by their lag k.',r'$-\partial_s\psi_s(t)=\sum_k k e^{-sk}\widetilde\phi_{t+k}$'),
('Event score','Cosine change measures directional change between consecutive band vectors.',r'$C_i(t)=1-\frac{B_i(t-1)^{\mathsf{T}}B_i(t)}{\|B_i(t-1)\|\,\|B_i(t)\|}$')],
'ddqn_explained':[
('Action values','The network estimates six values from the 30-dimensional motor state.',r'$Q_\theta:\mathbb{R}^{30}\longrightarrow\mathbb{R}^6$'),
('Double DQN selection','The online network selects the best eligible next action.',r'$o^*=\arg\max_{o\in\mathcal{A}(s^\prime)}Q_\theta(s^\prime,o)$'),
('Duration-aware target','Accumulate rewards over d commands, then bootstrap with the target network.',r'$y=\sum_{j=0}^{d-1}\gamma^j r_{t+j}-c_{delib}+1_{\neg terminal}\gamma^d Q_{\bar\theta}(s^\prime,o^*)$')],
'mechanics_explained':[
('Hand velocity','The Jacobian maps joint velocities to fingertip velocity.',r'$v=J(q)\dot q$'),
('Joint dynamics','Muscle torque, damping and external forces determine joint acceleration.',r'$M(q)\ddot q+c_{skel}(q,\dot q)=-R_{geom}(q)F_m-D_{eff}\dot q+\tau_{joint}+J(q)^{\mathsf{T}}f_{endpoint}$'),
('Muscle-to-joint coupling','Moment arms depend on posture. Muscle force depends on activation, length and velocity.',r'$R_{jm}=\partial\ell_m/\partial q_j,\qquad F_m=F_m(a,\ell,\dot\ell)$')]}
for name,rows in equations.items():
    fig=plt.figure(figsize=(12,5.2) if name=='laplace_explained' else (9,3.6))
    dy=.92/len(rows)
    for i,(label,meaning,eq) in enumerate(rows):
        y=.95-i*dy
        fig.text(.025,y,f'{i+1}. {label}',fontsize=13,weight='bold',va='top')
        if name=='laplace_explained':
            fig.text(.025,y-.065,textwrap.fill(meaning,42),fontsize=11,va='top')
            fig.text(.43,y-.10,eq,fontsize=13,va='center')
        else:
            fig.text(.025,y-.065,meaning,fontsize=11,va='top')
            fig.text(.05,y-.14,eq,fontsize=14 if name!='mechanics_explained' else 12,va='top')
    save(fig,name)

options=read('data/discovery/options/off_target_band_options.json')['options']
xy=np.array([r['xy_m'] for r in options])*1000
# Exact plant geometry: Arm26 uses links .309/.333 m and reference q=(45,90) deg.
sources.update([ROOT/'SAC/core/env.py',ROOT/'docs/references/motornet/effector.py'])
q=np.deg2rad([45,90]);center=1000*np.array([.309*np.cos(q[0])+.333*np.cos(q.sum()),.309*np.sin(q[0])+.333*np.sin(q.sum())])
angles=np.arange(8)*np.pi/4;targets=center+100*np.column_stack((np.cos(angles),np.sin(angles)))
def ring(ax):
    ax.add_patch(Circle(center,100,facecolor='none',edgecolor='#777777',linestyle=':',lw=1))
    for j,p in enumerate(targets):
        ax.add_patch(Circle(p,20,facecolor='white',edgecolor='#777777',lw=.7));ax.text(*p,'T'+str(j),ha='center',va='center',fontsize=10)
    ax.set_aspect('equal');ax.set(xlim=(center[0]-135,center[0]+135),ylim=(center[1]-135,center[1]+135),xlabel='Hand x (mm)',ylabel='Hand y (mm)')
fig,axs=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained');ring(axs[0])
axs[0].scatter(*xy.T,color='#b52f62',s=30,zorder=4)
for name,p in zip('ABCDE',xy):axs[0].annotate(name,p,xytext=(5,4),textcoords='offset points',fontsize=11)
axs[0].set_title('Landmarks within the eight-target ring',fontsize=12)
h=[.01/-np.log(r['gamma'])*1000 for r in options];axs[1].bar(list('ABCDE'),h,color='#769cb0',edgecolor='black',lw=.5)
axs[1].set(yscale='log',ylabel='Nominal discount horizon (ms)');save(fig,'landmarks_ring_annotated')

alt=read('data/discovery/controls/seed_1001_discovery.json');other=np.array([r['xy_m'] for r in alt['candidates']])*1000
distance=np.linalg.norm(other[:,None]-xy[None],axis=2);nearest=distance.argmin(axis=1)
fig,axs=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained');ring(axs[0])
axs[0].scatter(*xy.T,label='Original A–E',color='#b52f62',s=24,zorder=5);axs[0].scatter(*other.T,label='Alternate sequence',marker='D',color='#46748b',s=24,zorder=5)
for name,p in zip('ABCDE',xy):axs[0].annotate(name,p,xytext=(4,3),textcoords='offset points',fontsize=9)
for i,p in enumerate(other):
    axs[0].plot([p[0],xy[nearest[i],0]],[p[1],xy[nearest[i],1]],color='#777777',ls='--',lw=.8)
    axs[0].annotate('S'+str(i+1),p,xytext=(4,-10),textcoords='offset points',fontsize=9)
axs[0].legend(loc='lower left',fontsize=9,frameon=False)
axs[1].barh(['S'+str(i+1)+' to '+list('ABCDE')[j] for i,j in enumerate(nearest)],distance.min(axis=1),color='#769cb0',edgecolor='black',lw=.5)
axs[1].set_xlabel('Nearest original landmark (mm)');axs[1].invert_yaxis();save(fig,'alternate_ring_annotated')

sources.add(ROOT/'data/handoffs/controlled_launch/option_termination_kinematics.npz')
z=np.load(ROOT/'data/handoffs/controlled_launch/option_termination_kinematics.npz')
fig,ax=plt.subplots(figsize=(6.7,3.7),layout='constrained')
for name,color in zip('ABCDE',['#46748b','#b95d41','#7e9c73','#8e78a7','#ba9441']):
    tm=z[name+'_time_from_termination_ms'];c=z[name+'_cartesian_window'];speed=np.linalg.norm(c[:,2:4],axis=1)
    ax.plot(tm,speed,label=name,color=color,lw=1.6);i=np.argmin(abs(tm));ax.scatter(tm[i],speed[i],color=color,s=25)
ax.axvline(0,color='black',lw=.8,ls=':');ax.set(xlabel='Time from landmark arrival (ms)',ylabel='Fingertip speed (m/s)',xlim=(-50,50));ax.legend(title='Subgoal',ncol=5,frameon=False,fontsize=10)
save(fig,'termination_speed_lines')

d=read('data/summary/evidence.json')
rows=[]
for directory in ['data/mechanism/historical_transfer','data/single_discount/single_transfer']:
    for p in sorted((ROOT/directory).glob('*.json')):
        sources.add(p);rows.extend(r for r in json.loads(p.read_text())['rows'] if r['controller']=='base')
assert len(rows)==720
methods=['base','single','learned'];colors=['#777777','#46748b','#b52f62'];labels=['Base','Single discount','DDQN + A–E']
for group,orders in [('small',list(range(16,24))),('full',list(range(24)))]:
    fig,axs=plt.subplots(1,2,figsize=(8.7,2.8),layout='constrained')
    for j,m in enumerate(methods):
        vals=[r for r in rows if r['order'] in orders and r['method']==m];pct=100*np.mean([r['success'] for r in vals])
        assert np.isclose(pct,d[group][m]['completion_pct'])
        axs[0].bar(j,pct,color=colors[j],edgecolor='black',lw=.5);axs[0].text(j,pct+2,f'{pct:.2f}%',ha='center',fontsize=10)
    axs[0].set(xticks=range(3),xticklabels=labels,ylim=(0,105),ylabel='Sequence completion (%)')
    for j,c in enumerate(['base','single']):
        v=d[group]['vs_'+c];lo,hi=v['ci95_pp'];axs[1].errorbar(v['effect_pp'],j,xerr=[[v['effect_pp']-lo],[hi-v['effect_pp']]],fmt='o',color='#b52f62',capsize=4)
    axs[1].axvline(0,color='black',ls=':',lw=.8);axs[1].set(yticks=[0,1],yticklabels=['Versus base','Versus single'],xlabel='DDQN completion gain (percentage points)',ylim=(-.6,1.6));axs[1].set_title('95% order-bootstrap intervals',fontsize=11)
    save(fig,group+'_completion_annotated')
fig,axs=plt.subplots(1,2,figsize=(8.7,2.8),layout='constrained')
for m,color,label in zip(methods,colors,labels):axs[0].plot(range(16,24),d['small'][m]['by_order_pct'],marker='o',lw=1.1,color=color,label=label)
axs[0].set(xlabel='Movement order ID',ylabel='Completion (%)',ylim=(-5,105));axs[0].legend(frameon=False,fontsize=9)
for c,color,label in zip(['base','single'],colors[:2],labels[:2]):
    effects=np.array(d['small']['learned']['by_order_pct'])-d['small'][c]['by_order_pct'];axs[1].plot(range(16,24),effects,marker='o',lw=1,color=color,label='DDQN minus '+label)
axs[1].axhline(0,color='black',ls=':',lw=.8);axs[1].set(xlabel='Movement order ID',ylabel='Paired completion gain (pp)');axs[1].legend(frameon=False,fontsize=9)
save(fig,'order_effects_annotated')

fig,ax=plt.subplots(figsize=(6.7,3.4),layout='constrained');v=d['full']['vs_single'];effect=v['effect_pp'];lo,hi=v['ci95_pp']
ax.errorbar(effect,0,xerr=[[effect-lo],[hi-effect]],fmt='o',color='#b52f62',capsize=6,markersize=8);ax.axvline(0,color='black',ls=':',lw=.8)
ax.set(yticks=[0],yticklabels=['DDQN minus single discount'],xlabel='Completion gain (percentage points)',ylim=(-.7,.7),xlim=(-5,35))
save(fig,'single_transfer_effect')

summary=read('data/mechanism/neural/summary.json')
fig,axs=plt.subplots(1,2,figsize=(8.7,2.8),layout='constrained')
for r in summary['accuracy_by_fit']:
    axs[0].plot(range(1,24),r['head_nmse'],lw=1,alpha=.8,label=str(r['training_seed']))
axs[0].set(xlabel='Prediction head (short to long horizon)',ylabel='Mean NMSE, all evaluation episodes');axs[0].legend(title='Network fit seed',ncol=2,fontsize=8,frameon=False)
for r in summary['accuracy_by_fit']:axs[1].plot([0,1],[r['success_mean_nmse'],r['timeout_mean_nmse']],marker='o',lw=.8,alpha=.8)
axs[1].set(xticks=[0,1],xticklabels=['79 successes','1 timeout'],ylabel='Mean prediction NMSE',yscale='log')
save(fig,'neural_accuracy_annotated')
readout=[]
for seed in summary['training_seeds']:
    evaluation=read(f'data/mechanism/neural/evaluation_{seed}.json')
    readout.append({m:float(np.mean([r['future_displacement_mse'][m] for r in evaluation['rows'] if r['future_displacement_mse']])) for m in ['current','multi','single']})
for comp in ['current','single']:
    contrast=np.mean([r[comp]-r['multi'] for r in readout]);assert np.isclose(contrast,summary['future_displacement_advantage'][comp]['mean'])
fig,ax=plt.subplots(figsize=(6.7,3.4),layout='constrained')
for r in readout:ax.plot(range(3),[r[m] for m in ['current','multi','single']],marker='o',lw=1,alpha=.8)
ax.set(xticks=range(3),xticklabels=['Current state','Multiscale projection','Selected single SF'],ylabel='Future displacement MSE (m²)')
save(fig,'readout_actual_mse')
(BUILD/'readout_reaudit.json').write_text(json.dumps({'by_fit':readout,'definition':'comparator MSE minus multiscale MSE','contrast':summary['future_displacement_advantage']},indent=2),encoding='utf-8')

fig,ax=canvas(9,2.7)
for i,(title,detail) in enumerate([('Predictive discovery','Five candidate\nlandmark options'),('Embodied execution','Measured posture, velocity\nand activation at arrival'),('Transfer evaluation','204/240 completions\nversus 153/240 base')]):
    box(ax,.02+i*.335,.34,.285,.38,title+'\n\n'+detail,12)
    if i<2:arrow(ax,(.305+i*.335,.53),(.355+i*.335,.53))
ax.text(.5,.10,'Human passage-profile correlations provide a separate descriptive comparison.',ha='center',fontsize=12)
save(fig,'evidence_link_annotated')

fig,ax=canvas(9,2.8)
box(ax,.015,.37,.20,.28,'Same saved\nDDQN invocation\nstate',12)
for y,label in [(.74,'Continue base'),(.41,'Chosen A–E option'),(.08,'Other / mirrored waypoint')]:
    box(ax,.35,y,.27,.20,label,12);arrow(ax,(.215,.51),(.35,y+.10));arrow(ax,(.62,y+.10),(.76,.51))
box(ax,.76,.32,.225,.37,'Then continue base\n\nCompare remaining\nsequence completion',12)
save(fig,'matched_branches_annotated')

fig,ax=canvas(9,1.1)
for i,label in enumerate(['Start','First intermediate\ntarget','Second intermediate\ntarget','Final target']):
    x=.04+i*.245;box(ax,x,.20,.19,.60,label,11)
    if i<3:arrow(ax,(x+.19,.50),(x+.245,.50))

save(fig,'human_passages_explained')

manifest={'script':str(Path(__file__).relative_to(ROOT)),'commit':None,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
          'figures':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dpi':600} for p in outputs}}
(OUT/'annotated_figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Generated {len(outputs)} diagrams and plots. Re-audited all five readout fits and 720 transfer rows.')
