"""Direct 600 dpi Matplotlib PNGs from saved evidence; no simulated numbers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_landmark_evidence import clustered_effect, episode_key, read, write

SEEDS = (7, 101, 202)
COLORS = ('#28658B', '#B56A38', '#57774B')
OPTION_COLORS = ('#28658B', '#B56A38', '#57774B', '#865F8A', '#65717A')


def generate(source, screen):
    font = ROOT/'figures/style/fonts/EBGaramond-Regular.ttf'
    assert font.exists(), 'EB Garamond required'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'EB Garamond', 'font.size': 12, 'axes.titlesize': 15,
                         'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'legend.frameon': False, 'figure.facecolor': 'white'})
    dest = screen/'plots'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(fig, name, caption, inputs):
        fig.text(.02, .02, caption, fontsize=10, va='bottom')
        fig.subplots_adjust(bottom=.30 if name == 'numerical_replay_all_cases' else .18, top=.88, wspace=.35, hspace=.6)
        # Exactly one export pixel; PowerPoint insertion should also use a 1 px outline.
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(72/600)
        path = dest/(name+'.png')
        fig.savefig(path, dpi=600, facecolor='white')
        plt.close(fig)
        manifest.append(dict(figure=path.name, caption=caption,
                             sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             renderer='Matplotlib', dpi=600, font='EB Garamond', commit='Unavailable; source hashes recorded'))

    jobs = [read(p) for p in sorted((source/'rollouts').glob('*.json'))]
    assert len(jobs) == 72, 'Expected all three policies and all 24 orders'
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    transfer = []
    for i, seed in enumerate(SEEDS):
        subset = [j for j in jobs if j['policy_seed'] == seed]
        groups = {}
        rates = []
        for job in subset:
            assert len(job['base']) == len(job['ddqn']) == 10
            base = np.array([r['success'] for r in job['base']], float)
            ddqn = np.array([r['success'] for r in job['ddqn']], float)
            rates.append([base.mean(), ddqn.mean()])
            for context, value in enumerate(ddqn-base):
                groups[(seed, len(rates)-1, context, 0)] = [value]
        rates = np.array(rates)*100
        for row in rates:
            axes[0].plot(np.array([0, 1])+.11*(i-1), row, color=COLORS[i], alpha=.18, lw=.8)
        axes[0].plot(np.array([0, 1])+.11*(i-1), rates.mean(axis=0), 'o-', color=COLORS[i], lw=2.5, label=f'Base policy {seed}')
        effect = clustered_effect(groups)
        mean = effect['mean']*100
        low, high = np.array(effect['exploratory_order_bootstrap_95CI'])*100
        axes[1].errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color=COLORS[i], capsize=4)
        transfer.append(dict(seed=seed, **effect))
    axes[0].set(xticks=[0, 1], xticklabels=['Base', 'DDQN + base'], ylabel='Sequence completion (%)', ylim=(-3, 103), title='Completion across 24 target orders')
    axes[0].legend(fontsize=10, loc='lower left')
    axes[1].axvline(0, color='black', lw=.8)
    axes[1].set(yticks=range(3), yticklabels=[f'Policy {s}' for s in SEEDS], xlabel='Paired completion gain (percentage points)', title='Gains differ across trained policies')
    save(fig, 'transfer_all_policies', '240 paired evaluations per policy; 24 orders × 10 contexts. Thin lines: order means.\nError bars: exploratory 95% whole-order bootstrap intervals within each policy; all failures retained.', list((source/'rollouts').glob('*.json')))

    audit_path = screen/'saved_data_audit.json'
    if not audit_path.exists():
        from scripts.screen_landmark_evidence import audit
        audit(source, screen)
    audit_data = read(audit_path)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for col, comparator in enumerate(('base', 'mirror', 'other_mean')):
        for row, metric in enumerate(('success', 'progress_500ms_m')):
            ax = axes[row, col]
            scale = 100 if row == 0 else 1000
            for i, seed in enumerate(SEEDS):
                effect = next(r for r in audit_data['contrasts'] if (r['policy_seed'], r['comparator'], r['metric']) == (seed, comparator, metric))
                mean = effect['mean']*scale
                low, high = np.array(effect['exploratory_order_bootstrap_95CI'])*scale
                ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color=COLORS[i], capsize=3)
            ax.axvline(0, color='black', lw=.8)
            ax.set(yticks=range(3), yticklabels=[str(s) for s in SEEDS], ylabel='Base policy seed', xlabel='Completion gain (pp)' if row == 0 else '500 ms target-progress gain (mm)')
            if row == 0:
                ax.set_title({'base': 'Chosen option versus base', 'mirror': 'Versus reflected waypoint', 'other_mean': 'Versus other eligible options'}[comparator])
    save(fig, 'local_effects_all_controls', 'One option, then base continuation. Repeated invocations averaged within episodes.\nExploratory 95% whole-order bootstrap intervals; comparator-specific counts in saved_data_audit.json.', [audit_path, source/'branch_contrasts.json'])

    endpoints_path = source/'actual_endpoints.json'
    endpoints = read(endpoints_path)
    for seed in SEEDS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        counts = []
        means = []
        speeds = []
        for choice in range(1, 6):
            rows = [r for r in endpoints if r['policy_seed'] == seed and r['choice'] == choice and r['actual_endpoint']]
            groups = defaultdict(list)
            for r in rows:
                ep = r['actual_endpoint']
                groups[episode_key(r)].append(np.r_[ep['joint'], np.linalg.norm(ep['cartesian'][2:]), ep['activation']])
            values = np.array([np.mean(v, axis=0) for v in groups.values()])
            assert len(values)
            counts.append(f'{chr(64+choice)}: {len(rows)} invocations/{len(values)} episodes')
            means.append(values.mean(axis=0))
            speeds.append(values[:, 4])
            for panel, start, scale in ((axes[0, 0], 0, 180/np.pi), (axes[0, 1], 2, 1)):
                for joint in range(2):
                    data = values[:, start+joint]*scale
                    low, median, high = np.quantile(data, [.05, .5, .95])
                    x = choice-.1+.2*joint
                    panel.plot([x, x], [low, high], color=OPTION_COLORS[choice-1], lw=1.8, alpha=.75)
                    panel.scatter(x, median, color=OPTION_COLORS[choice-1], marker='o' if joint == 0 else 's', s=30)
        means = np.array(means)
        for ax in axes[0]:
            ax.set(xticks=range(1, 6), xticklabels=list('ABCDE'))
        axes[0, 0].set(ylabel='Joint angle (deg)', title='Joint configuration at arrival')
        axes[0, 1].set(ylabel='Joint velocity (rad/s)', title='Joint motion at arrival')
        for ax in axes[0]:
            ax.plot([], [], 'ko', label='Shoulder')
            ax.plot([], [], 'ks', label='Elbow')
            ax.legend(fontsize=9)
        parts = axes[1, 0].violinplot(speeds, showmedians=True, showextrema=False)
        for body, color in zip(parts['bodies'], OPTION_COLORS):
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(.55)
        axes[1, 0].set(xticks=range(1, 6), xticklabels=list('ABCDE'), ylabel='Hand speed (m/s)', title='Arrival need not mean stopping')
        image = axes[1, 1].imshow(means[:, 5:], vmin=0, vmax=1, cmap='Blues', aspect='auto')
        axes[1, 1].set(yticks=range(5), yticklabels=list('ABCDE'), xticks=range(6), xticklabels=['Pect', 'Delt', 'Brach', 'TriLat', 'Bic', 'TriLong'], title='Muscle activation at arrival')
        for row in range(5):
            for col in range(6):
                axes[1, 1].text(col, row, f'{means[row, col+5]:.2f}', ha='center', va='center', fontsize=10, color='white' if means[row, col+5] > .65 else 'black')
        fig.colorbar(image, ax=axes[1, 1], label='Activation (0–1)', shrink=.85)
        fig.suptitle(f'Actual DDQN landmark arrivals · base policy {seed}', fontsize=17)
        save(fig, f'actual_arrivals_policy{seed}', 'Episode means after averaging repeated invocations. Joint markers: median; bars: 5th–95th percentiles, not CIs.\nActivation: episode-weighted means. '+ '; '.join(counts), [endpoints_path])

    comparison_path = screen/'replay_comparison.json'
    if comparison_path.exists():
        rows = read(comparison_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        ticks = []
        for i, r in enumerate(rows):
            ticks.append(f"{chr(64+r['metadata']['choice'])} / {r['metadata']['policy_seed']}")
            for offset, dt, color in zip((-.16, 0, .16), ('0.01', '0.001', '0.0005'), ('#999999', '#28658B', '#B56A38')):
                value = r['chosen_minus_base_progress_500ms_m'][dt]
                if value is not None:
                    axes[0].scatter(i+offset, value*1000, color=color, s=25, label={'0.01': 'Euler 10 ms', '0.001': 'RK4 1 ms', '0.0005': 'RK4 0.5 ms'}[dt] if i == 0 else None)
            axes[1].scatter(i, r['RK4_1ms_vs_half_ms_max_position_error_m']*1000, color='#28658B', s=25)
        for ax in axes:
            if len(rows) <= 10:
                ax.set(xticks=range(len(rows)), xticklabels=ticks)
                ax.tick_params(axis='x', rotation=40)
            else:
                ax.set_xlabel('Locked replay case (all cases shown)')
        axes[0].axhline(0, color='black', lw=.8)
        axes[0].set(ylabel='Chosen minus base target progress (mm)', title='Local effects across numerical settings')
        axes[0].legend(fontsize=10)
        axes[1].set(ylabel='Maximum position discrepancy (mm)', title='RK4 1 ms versus 0.5 ms')
        save(fig, 'numerical_replay_all_cases', f'{len(rows)} outcome-blind saved starts; all eligible branches retained. Left: original-target progress at 500 ms.\nRight: maximum discrepancy across all branches/common timestamps. Diagnostic panel; no population CI.', [comparison_path, screen/'replay_protocol.json'])
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        curves = defaultdict(list)
        curve_inputs = [comparison_path, screen/'replay_protocol.json']
        for row in rows:
            path = screen/'replay'/(Path(row['file']).stem+'_dt0.001.json')
            result = read(path)
            curve_inputs.extend([path, path.with_suffix('.npz')])
            with np.load(path.with_suffix('.npz')) as arrays:
                phi = arrays['phi'][:51, :, :2]
            distance = np.linalg.norm(phi-np.array(result['original_goal_m']), axis=-1)
            progress = result['initial_distance_m']-distance
            for col, branch in enumerate(result['rows']):
                hit = branch['first_target_acquisition_step']
                if 0 < hit <= 50:
                    progress[hit:, col] = result['initial_distance_m']
            if len(progress) < 51:
                assert all(r['success'] for r in result['rows'])
                progress = np.concatenate([progress, np.repeat(progress[-1:], 51-len(progress), axis=0)])
            benefit = (progress[:, 1]-progress[:, 0])*1000
            stored = row['chosen_minus_base_progress_500ms_m']['0.001']
            if stored is not None:
                np.testing.assert_allclose(benefit[-1], stored*1000, atol=.001, rtol=0)
            curves[(row['metadata']['choice'], row['metadata']['policy_seed'])].append(benefit)
        for choice in range(1, 6):
            ax = axes.flat[choice-1]
            for seed, color in zip(SEEDS, COLORS):
                values = curves.get((choice, seed), [])
                for curve in values:
                    ax.plot(np.arange(51)*10, curve, color=color, alpha=.25, lw=.8)
                if values:
                    ax.plot(np.arange(51)*10, np.mean(values, axis=0), color=color, lw=2, label=f'Policy {seed}, n={len(values)}')
            ax.axhline(0, color='black', lw=.7)
            ax.set(title=f'Option {chr(64+choice)}', xlabel='Time since option invocation (ms)', ylabel='Chosen minus base progress (mm)')
            ax.legend(fontsize=8)
        axes.flat[5].axis('off')
        save(fig, 'fine_dynamics_continuation_curves', 'RK4 1 ms; original-target acquisition treated as absorbing. Thin lines: every selected state; thick lines: cell means.\nOutcome-blind panel from historical states; n denotes starts, not training seeds. No uncertainty band or causal-mediation claim.', curve_inputs)
    write(dest/'FIGURE_MANIFEST.json', manifest)
    write(dest/'transfer_effects.json', transfer)
    print(f'{len(manifest)} direct Matplotlib PNGs saved to {dest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'data/handoffs')
    parser.add_argument('--screen', type=Path, default=ROOT/'figures/output/correct_model/landmark_screen_v2_20261006')
    args = parser.parse_args()
    generate(args.source, args.screen)
