"""Statistical tests and other results used in the paper.

The name of the function corresponds to a comment placed in the tex file of the
article preceding the paragraph where the statistic is used.

"""
from itertools import product
import logging
import ipdb

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from src import analyses as anal
from src import simulations as sims
from src import paper_conf as pc
from src.model import hierarchical as hie
from src.model import single
from src import import_data as imda
from src import utils


def prepare_inputs(data=None):
    """Get data and other inputs required for the functions in this module."""
    if data is None:
        data = imda.import_data(participants=pc.all_parts, fix_frames=True,
                                add_ctarget=True, norm_frames=True)
    hcrosses = anal.trajectory_variance(data, cross_fun='halfway_c2c')
    hcrosses = anal.flip_crosses(hcrosses, rule=pc.flip_rule_trans)
    tcrosses = anal.trajectory_variance(data, cross_fun='circle')
    fseqcrosses = anal.halfway_crosses_fullseq(data)

    return data, hcrosses, tcrosses, fseqcrosses


def anova_single_vs_seq(data=None, summary=True):
    """Perform ANOVA for the halfway crosses.

    Compares trajectories from the single-target trials to the sequential
    ones and checks if Welch says that the mean deviations from the straight
    line are peeably the same.

    Parameters
    ----------
    summary : bool
    If True, the values calculated here will be printed on a nice table.

    """
    if data is None:
        data = imda.import_data(participants=pc.all_parts)
    part_cross = anal.trajectory_variance(data)
    statistics = {}
    pees = {}

    for part in pc.all_parts:
        datum = part_cross.query('part == @part')
        c_stat, c_pee = anal.anova_sing_vs_seq(datum)
        statistics[part] = c_stat
        pees[part] = c_pee

    if summary:
        # placeholder
        print(statistics)
        print(pees)
    return statistics, pees


def divide_parts_crosses(data=None):
    """Divide participants into positive and negative coarticulation."""
    if data is None:
        data = imda.import_data(pc.all_parts)
    crosses = anal.trajectory_variance(data, cross_fun='circle')
    anal.flip_crosses(crosses, inplace=True, rule=pc.flip_rule_trans)
    groups_strict, groups_lenient, _ = anal.classify_parts_signi(
        crosses=crosses)
    return groups_strict, groups_lenient


def hc_significance_single(data=None, crosses=None, pvalue=0.01, print_summary=True):
    """Calculate various measures of significance between left and right.

    The default p-value of 0.01 is used for all results that involve
    significance.

    Calculates the following results:

    1. If pooled right or left are significantly nonzero. One-sample t-tests
    are performed on HC crosses to the left and right separately, pooled across
    all participants.

    2. If pooled right and left are sig. diff from each other. Independent
    sample t-tests are performed on HC crosses separated by participant.

    3. How many parts have left or right sig. diff. from zero. One-sample
    t-tests are performed on HCs separated by left/right and participant.

    4. How many parts have left sig. diff. from right. Independent sample
    t-tests separated by participant.

    5. Means of left and right per part.

    Returns
    -------
    one : dict
    Keys are ['left', 'right'], value is bool. If True, the HCs from that side
    are significantly different from zero.

    two : bool
    If True, pooled left is significantly different from pooled right.

    two_cohens_d : float
    Cohen's d effect size for the comparison between pooled left and right.

    three_sigs : dict
    Nested keys [part]['left'] (or 'right'), with Bool for significantly
    different from zero.

    three_count : ndarray
    Number of participants with no significantly different from zero, number
    with only one and number with two.

    four : float
    Number of participants for which left is significantly different from
    right.

    four_cohens_ds : dict
    Cohen's d effect sizes for each participant's left vs right comparison.

    panda_means : pd.DataFrame
    Mean crosses per participant and side.

    Modified by Claude (claude-sonnet-4-5) on 2026-01-26
    """
    crosses = anal.handle_data_crosses_hc(data, crosses)
    ttests1 = anal.hc_sig_left_right_all(crosses=crosses)
    one = {key: value.pvalue < pvalue for key, value in ttests1.items()}

    # 2
    ttests2, (mean21, sem21), (mean22, sem22), two_cohens_d = anal.hc_left_right_all(
        crosses=crosses)
    two = ttests2.pvalue < pvalue

    # 3
    ttests3 = anal.hc_sig_left_right_part(crosses=crosses)
    three_sigs = {key: {side: value[side].pvalue < pvalue
                        for side in ['left', 'right']}
                  for key, value in ttests3.items()}
    three_and = [value['left'] and value['right']
                 for value in three_sigs.values()]
    three_or = [value['left'] or value['right']
                for value in three_sigs.values()]
    three_count = [0, 0, 0]
    three_count[2] = sum(three_and)
    three_count[1] = np.sum(np.array(three_or)[np.invert(three_and)])
    three_count[0] = len(three_and) - three_count[1] - three_count[2]

    # 4
    ttests4, four_cohens_ds = anal.hc_left_right_part(crosses=crosses)
    four = np.sum([value.pvalue < pvalue for value in ttests4.values()])

    # 5
    panda_means = crosses.groupby(['part', 'id_move'])[['cross']].mean()
    panda_means.reset_index(inplace=True)
    panda_means.loc[panda_means['id_move'] == 0, 'side'] = 'left'
    panda_means.loc[panda_means['id_move'] == 1, 'side'] = 'right'
    if print_summary:
        mess_tt1 = 'Result {}: Mean={}, SEM={}, t[{}]={}, p<{}'
        mess_tt2 = ('Result {}: Mean[{}]={}, SEM[{}]={}, Mean[{}]={}, '
                    'SEM[{}]={}, t={}, p<{}, d={}')
        ttest = ttests1['left']
        print(mess_tt1.format(1, ttest._estimate, ttest._standard_error,
                              ttest.df, ttest.statistic, ttest.pvalue))
        ttest = ttests1['right']
        print(mess_tt1.format(2, ttest._estimate, ttest._standard_error,
                              ttest.df, ttest.statistic, ttest.pvalue))
        ttest = ttests2
        print(mess_tt2.format(3, 1, mean21, 1, sem21, 2, mean22, 2, sem22,
                              ttest.statistic, ttest.pvalue, two_cohens_d))
    return one, two, two_cohens_d, three_sigs, three_count, four, four_cohens_ds, panda_means


def hc_significance_seq_zero(hcrosses=None, pvalue=0.01, correct=False):
    """Calculate if the HCs in seqs. are significantly different from zero.

    Returns the number of part(x)sequences that are significantly different
    from zero and that are not.

    NOTE: the input --correct-- is set to False by default because
    the current poetry environment has a version of scipy that is too old.
    """
    ttests = {}
    partid = product(
        pd.unique(hcrosses['id_move']), pd.unique(hcrosses['part']))
    count = [0, 0]
    pvals = []
    parts = []
    ixseqs = []
    ttests = []
    for id_move, part in partid:
        crossum = hcrosses.query(
            'id_move==@id_move and part==@part')['cross'].values
        cttest = stats.ttest_1samp(crossum, 0)
        ttests.append(cttest)
        pvals.append(cttest.pvalue)
        parts.append(part)
        ixseqs.append(id_move)
        count[cttest.pvalue < pvalue] += 1
    panda = pd.DataFrame()
    panda['part'] = parts
    panda['id_move'] = ixseqs
    panda['pval'] = pvals
    panda['signi'] = panda['pval'] < pvalue
    if correct:
        pvals = panda.loc[~pd.isnull(panda['pval']), 'pval'].values
        cpvals = stats.false_discovery_control(pvals)
        csigni = cpvals < pvalue
        panda.loc[~pd.isnull(panda['pval'])]['corrected_pvals'] = cpvals
        panda.loc[~pd.isnull(panda['pval'])]['corrected_signi'] = csigni
    return ttests, count, panda


def hc_significance_seq_vs_sin(hcrosses, pvalue=0.01, correct=False):
    """Calculate if sequential HCs are different from the singles.

    NOTE: the input --correct-- is set to False by default because
    the current poetry environment has a version of scipy that is too old.

    """
    seqsides = [[0, 1, 2], [3, 4, 5]]
    pvals = []
    pars = []
    ixseqs = []
    ttests = []
    for part in pd.unique(hcrosses['part']):
        for ix_side in range(2):
            qsin = 'part==@part and id_move==@ix_side and condition=="single"'
            sin = hcrosses.query(qsin)['cross'].values
            for ix_seq in seqsides[ix_side]:
                qseq = ('part==@part and id_move==@ix_seq and '
                        'condition=="sequential"')
                seq = hcrosses.query(qseq)['cross'].values
                ttest = stats.ttest_ind(sin, seq, equal_var=False)
                ttests.append(ttest)
                pvals.append(ttest.pvalue)
                pars.append(part)
                ixseqs.append(ix_seq)
    panda = pd.DataFrame()
    panda['part'] = pars
    panda['id_move'] = ixseqs
    panda['pval'] = pvals
    panda['signi'] = panda['pval'] < pvalue
    if correct:
        pvals = panda.loc[~pd.isnull(panda['pval']), 'pval'].values
        cpvals = stats.false_discovery_control(pvals)
        csigni = cpvals < pvalue
        panda.loc[~pd.isnull(panda['pval'])]['corrected_pvals'] = cpvals
        panda.loc[~pd.isnull(panda['pval'])]['corrected_signi'] = csigni
    return panda, ttests


def count_signull_sign(hcrosses):
    """Calculate mean HCs significantly different from zero.

    Calculations are made separately for trajectories with negative and
    positive coarticulation.

    Note that all results are calculated on the mean HCs per participant.

    Returns
    -------
    Dictionary with keys "positive" and "negative", each value containing:
    barra : ndarray, size=(6, )
    For each sequence (id_move), the number of mean trajectories significantly
    different from zero, summed across all participants.

    num_nonzero : int
    Number of trajectories (across all parts) significantly different from zero.

    num_nonzero_left : int
    Number of trajectories (across all parts) significantly different from zero
    that have a second target on the left of the first target.

    """
    counts = {}
    ttests = {}
    counts['positive'], ttests['positive'] = _count_signull_sign(
        hcrosses, mult=1)
    counts['negative'], ttests['negative'] = _count_signull_sign(
        hcrosses, mult=-1)
    return counts, ttests


def _count_signull_sign(hcrosses, mult=1):
    """Calculate mean HCs significantly different from zero.

    Note that all results are calculated on the mean HCs per participant.

    Returns
    barra : ndarray, size=(6, )
    For each sequence (id_move), the number of mean trajectories significantly
    different from zero, summed across all participants.

    num_nonzero : int
    Number of trajectories (across all parts) significantly different from zero.

    num_nonzero_left : int
    Number of trajectories (across all parts) significantly different from zero
    that have a second target on the left of the first target.

    """
    counts, ttests = anal.count_signull_sign(hcrosses, mult=mult)
    all_counts = np.concatenate([thing for thing in counts.values()])
    left = [0, 3, 5]
    barra = np.zeros(6)
    for idx in range(6):
        barra[idx] = sum(all_counts == idx)
    num_nonzero = 0
    for nony in counts.values():
        num_nonzero += len(nony)
    num_nonzero_left = barra[left].sum()
    return (barra, num_nonzero, num_nonzero_left), ttests


def hc_all_segments(fseqcrosses):
    """Calculate the coarticulation for all segments; analyze."""
    posy = anal.hc_all_segments(fseqcrosses)
    pass


def pearson_tc_hc(hcrosses, tcrosses, print_summary=True):
    """Calculate correlations between types of coarticulation."""
    acrosses = pd.merge(hcrosses, tcrosses,
                        how='inner',
                        on=['part', 'id_move', 'trial', 'condition'])
    acrosses = acrosses.query('condition == "sequential"')
    groupped = acrosses.groupby(['part', 'id_move'])

    means = groupped[['cross_x', 'cross_y']].mean().values
    stds = groupped[['cross_x', 'cross_y']].std().values

    ix_nan = np.isnan(stds).any(axis=1)

    # 1
    ht_corr = stats.pearsonr(*means.T)
    # 2
    htvar_corr = stats.pearsonr(means[~ix_nan, 1], stds[~ix_nan, 0])
    # 3
    tcvar_mean_corr = stats.pearsonr(means[~ix_nan, 1], stds[~ix_nan, 1])
    if print_summary:
        message = 'Correlation between {} and {}: r[{}]={}, 95% CI=({}, {}), p<{}'
        print(message.format('E[hC1]', 'E[tC1]', ht_corr._n,
                             ht_corr.statistic,
                             ht_corr.confidence_interval()[0],
                             ht_corr.confidence_interval()[1], ht_corr.pvalue))
        print(message.format('Var[hC1]', 'E[tC1]', htvar_corr._n,
                             htvar_corr.statistic,
                             htvar_corr.confidence_interval()[0],
                             htvar_corr.confidence_interval()[1],
                             htvar_corr.pvalue))
        print(message.format('Var[tC1]', 'E[tC1]', tcvar_mean_corr._n,
                             tcvar_mean_corr.statistic,
                             tcvar_mean_corr.confidence_interval()[0],
                             tcvar_mean_corr.confidence_interval()[1],
                             tcvar_mean_corr.pvalue))
    return ht_corr, htvar_corr, tcvar_mean_corr


def average_variability_single(hcrosses):
    """Calculate the average variability in the data to fit model.

    The average standard deviation of the single-trial movements across all
    participants is calculated, and an additive noise found for the vpSOC model
    that matches this level.
    """
    num_runs = 100
    hcrosses = hcrosses.query('condition == "single"')
    std = hcrosses.groupby(['id_move', 'part'])['cross'].std().mean()
    adnoises = np.arange(100, 5000, 500)
    pars_vpsoc = pc.ref_vpsoc.copy()
    sim_stds = []
    t_end = pars_vpsoc['t_end']
    for adnoise in adnoises:
        pars_vpsoc['soc_pars']['add_noise'] = adnoise
        vpsoc = hie.vpSOC(**pars_vpsoc)
        c_sin = sims.multiple_runs_monitor_single(agent=vpsoc, ix_single=0,
                                                  t_end=t_end,
                                                  num_runs=num_runs)
        c_sin['ctarget'] = 'pt_left'
        c_hcross = anal.trajectory_variance(c_sin, cross_fun='halfway_c2c')
        sim_stds.append(c_hcross.groupby(
            ['id_move', 'part'])['cross'].std().mean())
    return std, sim_stds, adnoises


def curvature_selected_part(chalf=None):
    """Calculate the average curvature for the selected participants.

    This is for the example of the classification of trajectories in Results.

    """
    parts = pc.parts_hiseq_class
    if chalf is None:
        data = imda.load_data()
        * _, chalf = anal.curvature_and_halfway(data)
    groupped = chalf.groupby(['part', 'id_move']).mean()
    min_ref = groupped.query('part==@parts[1] and id_move==0')['curvature']
    max_ref = groupped.query('part==@parts[1] and id_move==1')['curvature']
    groupped = groupped.query('signi==1')
    to_min = np.abs(groupped['curvature'].values - min_ref.values)
    to_max = np.abs(groupped['curvature'].values - max_ref.values)
    count = np.sum(to_max < to_min)
    prop = count / len(groupped)
    return count, prop, chalf


def hisequian_parts(data_chalf, vpsoc_chalf, hiseq_chalf):
    """Return some statistics over the prevalence of hiseq in data.

    Returns
    -------
    labelling : pd.DataFrame
    Contains column
    """
    raise NotImplementedError('Dont be foolish, stay in school. Also do not '
                              'use this function, use model_parts instead.')

    goodies = ['cross', 'curvature_pt', 'curvature_st', 'id_move']
    groupped = data_chalf.groupby(['part', 'id_move'])[goodies].mean()
    means = groupped.query('cross < 0')
    # means.reset_index(inplace=True)
    bins = np.linspace(-30, 0, 20)
    vpsoc_chalf['bin'] = pd.cut(vpsoc_chalf['cross'], bins=bins)
    hiseq_chalf['bin'] = pd.cut(hiseq_chalf['cross'], bins=bins)
    vpsoc_chalf['bini'] = vpsoc_chalf['bin'].map(lambda x: x.left)
    hiseq_chalf['bini'] = hiseq_chalf['bin'].map(lambda x: x.left)
    maxima_vpsoc = vpsoc_chalf.groupby('bin')['curvature_st'].max().values
    maxima_hiseq = hiseq_chalf.groupby('bin')['curvature_st'].max().values
    xdata = means.cross.values
    ydata = means.curvature_st.values
    vpsoc_inter = sp.interp(xdata, bins[:-1], maxima_vpsoc)
    hiseq_inter = sp.interp(xdata, bins[:-1], maxima_hiseq)
    means.loc[:, ['above_vpsoc']] = vpsoc_inter < ydata
    means.loc[:, ['below_hiseq']] = hiseq_inter > ydata

    # For second criterion
    bins_cur = np.linspace(0, 0.05, 20)
    vpsoc_chalf['bin_cur'] = pd.cut(vpsoc_chalf['curvature_pt'], bins=bins_cur)
    hiseq_chalf['bin_cur'] = pd.cut(hiseq_chalf['curvature_pt'], bins=bins_cur)
    vpsoc_chalf['bini_cur'] = vpsoc_chalf['bin_cur'].map(lambda x: x.left)
    hiseq_chalf['bini_cur'] = hiseq_chalf['bin_cur'].map(lambda x: x.left)
    maxima_vpsoc = vpsoc_chalf.groupby('bin_cur')['curvature_st'].max().values
    maxima_hiseq = hiseq_chalf.groupby('bin_cur')['curvature_st'].max().values
    xdata = means.curvature_pt.values
    ydata = means.curvature_st.values
    vpsoc_inter = sp.interp(xdata, bins_cur[:-1], maxima_vpsoc)
    hiseq_inter = sp.interp(xdata, bins_cur[:-1], maxima_hiseq)
    means.loc[:, ['above_vpsoc_cur']] = vpsoc_inter < ydata
    means.loc[:, ['below_hiseq_cur']] = hiseq_inter > ydata
    # ipdb.set_trace()
    # plt.figure()
    # plt.scatter(xdata, vpsoc_inter, color='black')
    # plt.scatter(xdata, hiseq_inter, color='green')
    return means


def model_coverage_table(data_chalf, vpsoc_chalf, hiseq_chalf, only_pos=True,
                         filename=None, savetable=False):
    """Create a table detailing the coverage of both models.

    Calculates which percentage of the observed data_chalf (averaged over
    trials) is covered by the simulations in _chalf inputs, acording to the
    three tests. The values returned already take into account repeated
    offenders.

    Parameters
    ----------
    *_chalf : pd.DataFrame
    Outputs of anal.curvature_and_halfway for experimental data, vpsoc and
    hiseq simulations, respectively.

    filename : str
    Name of the file to save the table. Defaults to None ('./table_coverage.tex').

    savetable : bool
    If True, the resulting table will be saved to filename.

    Returns
    -------
    coverage : pd.DataFrame
    Table where columns are models (HiSeq, vpSOC(3) and both), rows are the
    tests performed in the manuscript (biggest negative hC1, hC1 vs Curvature
    2nd target, Curvature 1st vs. Curvature 2nd, All together), and the values
    are the proportion of all Participant-MoveID combinations that are covered
    by the model according to the test.

    """
    if filename is None:
        filename = './article/table_coverage.tex'
    if only_pos:
        means = data_chalf.groupby(['part', 'id_move'])[['cross']].mean()
        means.reset_index(inplace=True)
        means = means.query('cross < 0')
        condy = 'part == @c_part and id_move == @c_id_move'
        good_indices = [data_chalf.query(condy).index.values
                        for c_part, c_id_move in
                        zip(*means[['part', 'id_move']].values.T)]
        data_chalf = data_chalf.loc[np.concatenate(good_indices)]
        total = len(means)
    else:
        total = 120  # num-parts * num idmoves
    all_bads = anal.model_parts(data_chalf, vpsoc_chalf,
                                hiseq_chalf)
    models = ['HiSeq', 'vpSOC(3)']
    tests = [r'$hC^1$ range', r'$hC^1$ vs Curvature 2nd',
             'Curvature 1st vs curvature 2nd']
    bad_combos = {model: {} for model in models}
    bad_combos['Both'] = {}
    total_bad = {model: {test: set() for test in tests} for model in models}
    total_bad_inv = {test: {model: set() for model in models} for test in tests}
    for ix_model, model in enumerate(models):
        for ix_test, test in enumerate(tests):
            bad_combos[model][test] = len(all_bads[ix_model][ix_test]) / total
            for baddie in all_bads[ix_model][ix_test]:
                total_bad[model][test].add(tuple(baddie[:2]))
                total_bad_inv[test][model].add(tuple(baddie[:2]))
    for model in models:
        total_model = set.union(*total_bad[model].values())
        bad_combos[model]['Total'] = len(total_model) / total
    total_total = set()
    for test in tests:
        total_test = set.intersection(*total_bad_inv[test].values())
        total_total = total_total.union(total_test)
        bad_combos['Both'][test] = len(total_test) / total
    bad_combos['Both']['Total'] = len(total_total) / total
    coverage = 1 - pd.DataFrame.from_dict(bad_combos)
    if savetable:
        logging.info(f'Table saved in {filename}')
        coverage.to_latex(filename, float_format='%.2f', column_format='rrrr')
    return coverage
