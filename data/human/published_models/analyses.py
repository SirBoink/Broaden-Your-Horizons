"""A variety of tools for analyses."""
import logging

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats, signal

from src import utils
from src import simulations as sims
from src import import_data as imda
from src import paper_conf as pc
from src.model import hierarchical as hie

logging.getLogger().setLevel(logging.INFO)


class NoCrossError(Exception):
    """Error class to specify in some fringe cases."""

    pass


def trajectory_variance(pandata, cross_fun=None, prop=0.5,
                        cross_fun_kwargs=None):
    """Calculate variance of trajectories in the data for one participant.

    Calculates the mean and standard deviation of the trajectories of each
    movement as they cross the half-way point on their way to the target.

    This is just a for loop with a call to trajectory_variance_one_part.

    Parameters
    ----------
    pandata : pd.DataFrame
    Data from the experiment or simulations. See the data folder's readme
    for more information on the elements.

    cross_fun : function, str
    Function that calculates the crosses. Defaults to _trajvar_c2chalf. If a
    string, can be in {'halfway_c2c', 'halfway_sc2', 'circle'}.

    prop : float in (0, 1]
    To be used with halfway crosses. The crosses are calculated at a --prop--
    distance between starting position and target. E.g. if prop=0.5, the
    crosses are calculated halfway through.

    Returns
    -------
    pandout : DataFrame
    Panda with a row for each movement, with the following columns:
      'cross' : float
      Point at which the trajectory crossed the halfway point, in the
      local coordinates of the straight line between start and end.

      'id_move' : int
      Yes

      'condition' : str
      Yes

    cross_fun_kwargs : dict
    Keyword arguments for cross_fun.

    """
    all_crosses = []
    if not ('part' in pandata.columns):
        pandata['part'] = 0
    for part in np.unique(pandata['part']):
        pandatum = pandata.query('part == @part and success')
        if len(pandatum) == 0:
            continue
        c_crosses = trajectory_variance_one_part(pandatum, cross_fun=cross_fun,
                                                 prop=prop,
                                                 cross_fun_kwargs=cross_fun_kwargs)
        c_crosses['part'] = part
        all_crosses.append(c_crosses)
    return pd.concat(all_crosses)


def trajectory_variance_one_part(pandata, cross_fun=None, prop=0.5,
                                 cross_fun_kwargs=None):
    """Calculate variance of trajectories in the data for one participant.

    Calculates the mean and standard deviation of the trajectories of each
    movement as they cross the half-way point on their way to the target.

    Note that this assumes that only one participant is in the data. Buyer
    beware.

    Parameters
    ----------
    cross_fun : function, str
    Function that calculates the crosses. Defaults to _trajvar_c2chalf. If a
    string, can be in {'halfway_c2c', 'halfway_s2c', 'circle'}.

    prop : float in (0, 1]
    To be used with halfway crosses. The crosses are calculated at a --prop--
    distance between starting position and target. E.g. if prop=0.5, the
    crosses are calculated halfway through.

    Returns
    -------
    pandout : DataFrame
    Panda with a row for each movement, with the following columns:
      'cross' : float
      Point at which the trajectory crossed the halfway point, in the
      local coordinates of the straight line between start and end.

      'id_move' : int
      Yes

      'condition' : str
      Yes

    cross_fun : {"halfway_c2c", "halfway_s2c", "circle"}  or function handle
    Function to use to calculate the crosses. For strings:
      halfway_c2c : The crossings are taken halfway between the
      center-to-center trajectory from starting position to first target. The
      crosses are calculated as the deviation from the center-to-center line.

      halfway_s2c : Like before, but the center-to-center line is from the
      starting position of the trajectory to the center of the first target.

      halfway_s2s : Like before, but the CtC line is between the starting point
      of the trajectory and its own ending point, ignoring circles. BEWARE,
      this makes no sense in sequential trials.

      circle : For sequential movements only. The crossing is calculated as the
      trajectory goes through the circle fo the first target. TODO: finish
      this.

    cross_fun_kwargs : dict
    Keyword arguments for cross_fun.

    """
    if cross_fun_kwargs is None:
        cross_fun_kwargs = {}
    if cross_fun is None or cross_fun == "halfway_c2c":
        cross_fun = _trajvar_c2chalf
    elif cross_fun == "halfway_s2c":
        cross_fun = _trajvar_s2chalf
    elif cross_fun == "halfway_s2s":
        cross_fun = _trajvar_s2shalf
    elif cross_fun == "circle":
        cross_fun = _trajvar_circle_c2chalf
    pandata = pandata.loc[pandata['success']]
    basket = utils.coords_from_svg(reference='mk_tr')
    all_crosses = []
    all_idmove = []
    all_cond = []
    all_trials = []
    for cond in np.unique(pandata['condition']):
        if cond == "single" and cross_fun is _trajvar_circle_c2chalf:
            logging.warning('Requested circle crossings with single condition'
                            '. Skipping.')
            continue
        datum_cond = pandata.loc[pandata['condition'] == cond]
        for id_move in np.unique(datum_cond['id_move']):
            datum = datum_cond.query('id_move == @id_move')
            crosses = cross_fun(datum, basket, all_trials, prop=prop,
                                **cross_fun_kwargs)
            if crosses is None:
                continue
            all_crosses.append(crosses)
            all_idmove.append(np.tile(id_move, len(crosses)))
            all_cond.append(np.tile(cond, len(crosses)))
    num_cross = np.concatenate(all_crosses).astype(float)
    num_id = np.concatenate(all_idmove)
    num_cond = np.concatenate(all_cond)
    # data = np.vstack([num_cross[None, :], num_id[None, :],
    #                   num_cond[None, :]])
    pandout = pd.DataFrame(num_cross, columns=['cross'])
    pandout['id_move'] = num_id
    pandout['condition'] = num_cond
    pandout['trial'] = all_trials
    return pandout


def flip_crosses(crosses, rule=None, inplace=False):
    """Flip the sign of the crosses given the rule.

    The idea is to make all "good" crosses (e.g. with the desired
    coarticulation) have the same sign so that they are easier to interpret in
    plots.

    Parameters
    ----------
    crosses : pd.DataFrame
    Output of trajectory_variance

    rule : dict
    Its elements should be (id_move, condition): multiplier, where all the
    --crosses-- with -id_move- and the --condition-- will be multiplied by the
    --multiplier--.

    inplace : bool
    Whether the work is done in place or a new DataFrame is returned.

    """
    if not inplace:
        crosses = crosses.copy()
    for (id_move, cond), multiplier in rule.items():
        iffy = np.logical_and(crosses['id_move'] == id_move,
                              crosses['condition'] == cond)
        crosses.loc[iffy, ['cross']] *= multiplier
    if not inplace:
        return crosses


def flip_crosses_all(crosses, rule=None, inplace=False):
    """Flip the sign of the crosses given the rule, for all segments.

    Like flip_crosses, the idea is to make all "good" crosses (e.g. with the
    desired coarticulation) have the same sign so that they are easier to
    interpret in plots.

    This is meant to work with the output of trajvar_all_segments.

    Parameters
    ----------
    crosses : pd.DataFrame
    Output of trajectory_variance

    rule : dict
    Its elements should be (id_move, condition): multiplier, where all the
    --crosses-- with -id_move- and the --condition-- will be multiplied by the
    --multiplier--.

    inplace : bool
    Whether the work is done in place or a new DataFrame is returned.

    """
    if rule is None:
        rule = pc.flip_rule_all
    if not inplace:
        crosses = crosses.copy()
    for (id_move, cond, seg), multiplier in rule.items():
        iffy = np.logical_and(crosses['id_move'] == id_move,
                              crosses['condition'] == cond)
        iffy = np.logical_and(iffy, crosses['segment'] == seg)
        if sum(iffy) == 0:
            print('yo')
        crosses.loc[iffy, ['cross']] *= multiplier
    if not inplace:
        return crosses


def _trajvar_s2chalf(datum, basket, all_trials, prop=0.5):
    """Calculate crossings halfway between start-to-center line.

    Parameters
    ----------
    prop : float in (0, 1]
    The crosses are calculated at a --prop-- distance between starting position
    and target. E.g. if prop=0.5, the crosses are calculated halfway through.

    """
    crosses = []
    for trial in np.unique(datum['trial']):
        label_end = datum.iloc[0]['second']
        traj = datum.loc[datum['trial'] == trial,
                         ['pos_x', 'pos_y']].to_numpy()
        center_ini = traj[0, :]
        center_end = basket[label_end].center
        center_diff = center_end - center_ini
        angle = np.arctan2(center_diff[1], center_diff[0])
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        rot_mat = np.array([[cosangle, sinangle], [-sinangle, cosangle]])
        x_mid = np.linalg.norm(center_diff) * prop
        traj_ori = traj - center_ini[None, :]
        traj_ori_rot = rot_mat.dot(traj_ori.T).T
        idx_cross = np.nonzero(traj_ori_rot[:, 0] > x_mid)[0]
        if len(idx_cross) == 0:
            crosses.append(np.nan)
        else:
            crosses.append(traj_ori_rot[idx_cross[0], 1])
        all_trials.append(trial)
    return crosses


def _trajvar_s2shalf(datum, basket, all_trials, prop=0.5,
                     ix_cutoffs=(0, -1)):
    """Calculate crossings from actual beginning to actual end.

    Parameters
    ----------
    prop : float in (0, 1]
    The crosses are calculated at a --prop-- distance between starting position
    and target. E.g. if prop=0.5, the crosses are calculated halfway through.

    ix_cutoffs : iterable, size=(2,)
    Indices for the starting and ending position in the trjectory. If (idx,
    idy), the idx-th row is used as the first point and the idy-th row as the
    last point; the crosses are then calculated as the deviation of the
    straight line between these two points.

    """
    crosses = []
    for trial in np.unique(datum['trial']):
        traj = datum.loc[datum['trial'] == trial,
                         ['pos_x', 'pos_y']].to_numpy()
        center_ini = traj[ix_cutoffs[0], :]
        center_end = traj[ix_cutoffs[1], :]
        center_diff = center_end - center_ini
        angle = np.arctan2(center_diff[1], center_diff[0])
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        rot_mat = np.array([[cosangle, sinangle], [-sinangle, cosangle]])
        x_mid = np.linalg.norm(center_diff) * prop
        traj_ori = traj - center_ini[None, :]
        traj_ori_rot = rot_mat.dot(traj_ori.T).T
        idx_cross = np.nonzero(traj_ori_rot[:, 0] > x_mid)[0]
        if len(idx_cross) == 0:
            crosses.append(np.nan)
        else:
            crosses.append(traj_ori_rot[idx_cross[0], 1])
        all_trials.append(trial)
    return crosses


def _trajvar_c2chalf(datum, basket, all_trials, prop=0.5, target_cols=None):
    """Calculate crossings halfway between center-to-center line.

    Parameters
    ----------
    prop : float in (0, 1]
    The crosses are calculated at a --prop-- distance between starting position
    and target. E.g. if prop=0.5, the crosses are calculated halfway through.

    """
    if target_cols is None:
        target_cols = ['first', 'second']
    label_ini = datum.iloc[0][target_cols[0]]
    label_end = datum.iloc[0][target_cols[1]]
    if not (isinstance(label_ini, str) and isinstance(label_end, str)):
        return None
    center_ini = basket[label_ini].center
    center_end = basket[label_end].center
    center_diff = center_end - center_ini
    angle = np.arctan2(center_diff[1], center_diff[0])
    sinangle = np.sin(angle)
    cosangle = np.cos(angle)
    rot_mat = np.array([[cosangle, sinangle], [-sinangle, cosangle]])
    crosses = []
    # ipdb.set_trace(cond=label_ini == "st_top")
    for trial in np.unique(datum['trial']):
        x_mid = np.linalg.norm(center_diff) * prop
        condi = np.logical_and(datum['trial'] == trial,
                               datum['ctarget'] == label_end)
        traj = datum.loc[condi,
                         ['pos_x', 'pos_y']].to_numpy()
        traj_ori = traj - center_ini[None, :]
        traj_ori_rot = rot_mat.dot(traj_ori.T).T
        idx_cross = np.nonzero(traj_ori_rot[:, 0] > x_mid)[0]
        if len(idx_cross) == 0:
            crosses.append(np.nan)
        else:
            crosses.append(traj_ori_rot[idx_cross[0], 1])
        all_trials.append(trial)
    return crosses


def trajvar_all_segments(data, signify=True, sweet=False):
    """Calculate the trajectory variance (halfway) for all segments.

    Parameters
    ----------
    sweet : Bool
    If False (default), the function throws an error if a trajectory did not
    reach all targets. If False, the function continues like nothing happened,
    leaving NaNs behind.

    """
    ordinals = [ordi for ordi in ['first', 'second', 'third', 'fourth']
                if ordi in data.columns]
    basket = utils.coords_from_svg('mk_tr')
    uids = []
    crosses = []
    segments = []
    for uid, datum in data.groupby('unique_trial_id'):
        targets = datum[ordinals].values[0]
        for segment, (start, finish) in enumerate(zip(targets, targets[1:])):
            datumtum = datum.query('ctarget==@finish')
            traj = datumtum[['pos_x', 'pos_y']].values
            if pd.isnull(finish):
                break
            x_ini = basket[start].center
            x_end = basket[finish].center
            cross = calculate_intersection(x_ini, x_end, halfway=0.5,
                                           trajectories=[traj])
            point = (x_ini + x_end) / 2
            distances = np.linalg.norm(point - cross[0], axis=1)
            try:
                ix_min = distances.argmin()
            except ValueError as err:
                if sweet:
                    continue
                raise err
            if signify:
                sign = -1 + 2 * (cross[0][ix_min, 1] > point[1])
            else:
                sign = 1
            distance = sign * distances.min()
            segments.append(segment)
            crosses.append(distance)
            uids.append(uid)
    trajvars = pd.DataFrame(data=np.array(crosses).T,
                            columns=['cross'])
    trajvars.loc[:, ['segment']] = segments
    trajvars.loc[:, ['unique_trial_id']] = uids
    return trajvars


def calculate_intersection(x_ini, x_end, halfway, trajectories,
                           only_nonnone=True):
    """Calculate the intersection of trajectories with hyperplane.

    Given a straight line AB between --x_ini-- and --x_end--, and some
    trajectories in the same space, calculate the intersection between each of
    these trajectories with a hyperplane perpendicular to AB, that intersects
    AB at a point --halfway-- between --x_ini-- and --x_end--.

    Adapted from the algorithm suggested by ChatGPT.

    Note: This is vectorized at the level of each trajectory. Because each
    trajectory can have a different number of frames, it is not possible to
    vectorize across trajectories.

    Parameters
    ----------
    trajectories : list[ndarray]
    Each element of the list is one TxN trajectory, where T is the number of
    frames in the trajectory (which depends on the trajectory) and N is the
    size of the space.

    only_nonnone : bool
    If True, only the sensor data at the points of intersection are
    returned. Otherwise, if a segment of a trajectory did not cross the
    hyperplane, a row of Nones is added to the output; this preserves the
    indices, if that's what you're into.

    Returns
    -------
    crosses : list
    Each element corresponds to a trajectory, and for each trajectory returns a
    list of masked ndarrays, each one being a point at which the trajectory
    crossed the hyperplane. The mask is 0 where the trajectory crossed the
    hyperplane.

    """
    # Calculate the direction vector AB and the midpoint M
    x_ini = np.array(x_ini)
    x_end = np.array(x_end)
    plane_normal = (x_end - x_ini)[None, :]
    plane_point = (x_ini * (1 - halfway) + x_end * halfway)[None, :]

    # Function to calculate the intersection of a line segment with the plane
    def line_plane_intersection(raw_traj):
        nonnan = np.logical_not(np.isnan(raw_traj).any(axis=1))
        traj = np.array(raw_traj[nonnan, :])
        line_direction = traj[1:, :] - traj[:-1, :]
        numerator = np.dot(plane_point - traj[:-1], plane_normal.T)[:, 0]
        denominator = np.dot(line_direction, plane_normal.T)[:, 0]
        denominator = np.ma.array(denominator, mask=denominator == 0,
                                  fill_value=np.inf)
        t = numerator / denominator
        inty = traj[:-1] + t[:, None] * line_direction
        inty[t < 0] = np.ma.masked
        inty[t > 1] = np.ma.masked
        return np.array(inty)[np.where(inty.mask == 0)[0], :]
    outie = list(map(line_plane_intersection, trajectories))
    return outie


def _calculate_only_intersects(intersects):
    """Return only those points of intersection.

    Works on the output of calculate_intersection, getting rid of all the
    instances where no intersection was found.
    """
    nonnone = []
    for intersect in intersects:
        idx_int = np.where((~np.isnan(intersect.filled())).all(axis=1))
        nonnone.append(np.array([intersect[idx] for idx in idx_int]))
    return nonnone


def _trajvar_circle_c2chalf(datum, basket, all_trials, prop=None):
    """Calculate transition intersections.

    The parameter --prop-- is a dummy for compatibility.

    """
    label_ini = datum.iloc[0]['first']
    label_mid = datum.iloc[0]['second']
    label_end = datum.iloc[0]['third']
    labels = [label_ini, label_mid, label_end]
    centers = np.array([basket[label].center
                        for label in labels])
    _, rot_mat = find_transition_angle_circle(centers)
    radius = basket[label_mid].radius
    crosses = []
    for trial in np.unique(datum['trial']):
        traj = datum.loc[datum['trial'] == trial,
                         ['pos_x', 'pos_y']].to_numpy()
        traj_ori = traj - centers[1]
        idx_in_circ = np.linalg.norm(traj_ori, axis=1) < radius
        if np.any(idx_in_circ):
            traj_in = traj_ori[idx_in_circ, :]
        else:  # Hack for simulations with failed triales
            logging.warning('Trajectory not entering first target!')
            traj_in = traj_ori
        traj_ori_rot = rot_mat.dot(traj_in.T).T
        idx_cross = np.argmin(np.abs(traj_ori_rot[:, 1]))
        # ipdb.set_trace(cond=datum.iloc[0]['id_move'] == 2)
        crosses.append(traj_ori_rot[idx_cross, 0])
        all_trials.append(trial)
    return crosses


def find_transition_angle_circle(centers):
    """Find the angle between the legs of sequential moves.

    In sequential movements defined by --centers--, find the counterclockwise
    angle that splits in half the counterclockwise angle between the first leg
    of the movement (centers[0] to centers[1]) and the second leg (centers[1]
    to centers[2]). The function returns this angle, making sure it lies
    between zero and 2pi.

    Parameters
    ----------
    centers : ndarray, size=(3, 2)
    Centers of the starting point, middle point and final point of the
    sequential trajectory.

    Returns
    -------
    angle : float
    Angle between 0 and 2pi.

    rot_mat : ndarray, size (2, 2)
    Rotation matrix given --angle--.

    """
    first_leg = centers[1, :] - centers[0, :]
    second_leg = centers[1, :] - centers[2, :]
    uni_1 = first_leg / np.linalg.norm(first_leg)
    uni_2 = second_leg / np.linalg.norm(second_leg)
    # alpha = np.arccos(uni_1.dot(uni_2)) / 2
    det = uni_1[0] * uni_2[1] - uni_1[1] * uni_2[0]
    alpha = np.arctan2(det, uni_1.dot(uni_2)) / 2
    if alpha < 0:
        alpha += np.pi
    gamma = np.arctan2(first_leg[1], first_leg[0])
    angle = np.pi + alpha + gamma
    sinangle = np.sin(angle)
    cosangle = np.cos(angle)
    rot_mat = np.array([[cosangle, sinangle], [-sinangle, cosangle]])
    return angle, rot_mat


def anova_sing_vs_seq(crosses):
    """One-way ANOVA to compare the first segments of all trajectories."""
    ix_sings = [0, 1]
    ix_seqs = [[0, 1, 2], [3, 4, 5]]

    statistics = np.zeros(len(ix_sings))
    pees = np.zeros(len(ix_sings))
    for ix_sing in ix_sings:
        iffy = 'condition == "single" and id_move == @ix_sing'
        crossum_sing = crosses.query(iffy)
        singles = crossum_sing['cross'].to_numpy()
        vectors = [singles]
        for idx, ix_seq in enumerate(ix_seqs[ix_sing]):
            iffy = 'condition == "sequential" and id_move == @ix_seq'
            vectors.append(crosses.query(iffy)['cross'].to_numpy())
        anova_out = stats.f_oneway(*vectors)
        statistics[ix_sing] = anova_out[0]
        pees[ix_sing] = anova_out[1]
    return statistics, pees


def classify_parts_signi(crosses):
    """Classify participants into positive and negative using stats.

    For each participant, each id_move, the repetitions are tested to see if
    they're significantly different from zero, using t-test (or Wilcoxon,
    depending on a Shapiro test for normality). A participant is labeled as
    strictly positive (negative) if they have at least one significantly
    positive (negative) id_move, and no significantly negative (positive). A
    more lenient definition (group_most) is when most of the sequences are
    positive.

    Returns
    -------
    groups_strict : list[int]
    List with participants in the [positive, negative, none] groups.

    groups_most : list[int]
    List with participants in the [positive, negative, none] groups.

    significance : dict[(str, int)]: float
    Significance for each (participant, ix_seq) combination.
    """
    significance = {}
    groups_strict = [[], []]
    groups_most = [[], []]
    for part in pd.unique(crosses['part']):
        has_negpos = [False, False]
        count = 0
        for idmove in pd.unique(crosses['id_move']):
            crossum = crosses.query('part == @part and id_move == @idmove')
            if len(crossum) == 0:
                mess = f'Part {part} and idmove {idmove} do not exist together'
                logging.warning(mess)
                continue
            crossy = crossum['cross'].values
            try:
                normy = stats.shapiro(crossy)
            except ValueError:
                normy = [0, 1]  # Dummy value to signify insignificance
            if normy[1] > 0.05:
                test = stats.ttest_1samp(crossy, 0)
            else:
                test = stats.wilcoxon(crossy)
            significance[(part, idmove)] = test
            if test[1] < 0.01:
                median = np.median(crossy)
                has_negpos[median > 0] = True
                count += 1 - 2 * (median < 0)
        if has_negpos[0] and not has_negpos[1]:
            groups_strict[0].append(part)
        elif has_negpos[1] and not has_negpos[0]:
            groups_strict[1].append(part)
        groups_most[count > 0].append(part)
    return groups_strict, groups_most, significance


def add_groups_to_panda(panda, groups_s, groups_l, inplace=False):
    """Add columns to --panda-- with the groups.

    For each groupping scheme (_s, _l), a column is added to the panda which
    says which group the participant belongs to. This assumes that the panda
    has a 'part' columns.

    Returns a copy of the panda unless inplace is True.

    """
    if not inplace:
        panda = panda.copy()
    labels = ['group_strict', 'group_lenient']
    for groups, label in zip([groups_s, groups_l], labels):
        for ix_group, group in enumerate(groups):
            for part in group:
                panda.loc[panda['part'] == part, label] = ix_group
    if not inplace:
        return panda


def count_signull_sign(crosses, mult=1):
    """Count the number of signif-diff-from-zero trajectories per part.

    Parameters
    mult : {-1, 1}
    If -1, returns counts negatively-coarticulated trajectories. If 1,
    the positively ones.
    """
    p_value = 0.01
    # rule = {('sequential', 0): -1,
    #         ('sequential', 4): -1}
    # crosses = flip_crosses(crosses, rule=rule)
    counts = {}
    parts = pd.unique(crosses['part'])
    id_moves = pd.unique(crosses['id_move'])
    for part in parts:
        counts[part] = []
        for id_move in id_moves:
            crossum = crosses.query('part == @part and id_move == @id_move')
            crossy = crossum['cross'].values
            normy = stats.shapiro(crossy)
            if normy[1] > p_value:
                test = stats.ttest_1samp(crossy, 0)
            else:
                test = stats.wilcoxon(crossy)
            median = np.median(crossy)
            if test[1] < p_value and mult * median > 0:
                counts[part].append(id_move)
    return counts


def halfway_crosses_fullseq(data):
    """Calculate the halfway crosses at different intervals.

    For each trial, the halfway crosses are calculated at multiple points of
    each segment.
    """
    targets = ['first', 'second', 'third', 'fourth']
    props = [0.25, 0.5, 0.75]
    all_crosses = []
    for idx, (taini, taend) in enumerate(zip(targets, targets[1:])):
        for prop in props:
            cf_kwargs = {'target_cols': [taini, taend]}
            crosses = trajectory_variance(data, cross_fun='halfway_c2c',
                                          prop=prop,
                                          cross_fun_kwargs=cf_kwargs)
            crosses['segment'] = idx + 1
            crosses['prop'] = prop
            all_crosses.append(crosses)
    return pd.concat(all_crosses, ignore_index=True)


def hc_left_right_all(data=None, crosses=None):
    """Do t-test on left vs right hc crosses across all parts in the data."""
    crosses = handle_data_crosses_hc(data, crosses)
    left = crosses.query('id_move == 0')['cross'].values
    right = crosses.query('id_move == 1')['cross'].values
    resy = stats.ttest_ind(-left, right, equal_var=False)
    return (resy,
            (np.mean(-left), stats.sem(-left)),
            (np.mean(right), stats.sem(right)))


def hc_left_right_part(data=None, crosses=None):
    """Do t-test on left vs right hc crosses for each part separately."""
    crosses = handle_data_crosses_hc(data, crosses)
    ttests = {}
    for part in pd.unique(crosses['part']):
        crossum = crosses.query('part == @part')
        ttests[part], *_ = hc_left_right_all(crosses=crossum)
    return ttests


def hc_sig_left_right_all(data=None, crosses=None):
    """Test significant different from zero, divided by left and right."""
    crosses = handle_data_crosses_hc(data, crosses)
    left = crosses.query('id_move == 0')['cross'].values
    right = crosses.query('id_move == 1')['cross'].values
    resy = {}
    resy['right'] = stats.ttest_1samp(right, 0)
    resy['left'] = stats.ttest_1samp(left, 0)
    return resy


def hc_sig_left_right_part(data=None, crosses=None):
    """Test significantly different from zero, divided by part, leftright."""
    crosses = handle_data_crosses_hc(data, crosses)
    ttests = {}
    for part in pd.unique(crosses['part']):
        crossum = crosses.query('part == @part')
        ttests[part] = hc_sig_left_right_all(crosses=crossum)
    return ttests


def handle_data_crosses_hc(data, crosses):
    """Auxiliary function to handle inputs.

    If --data-- or --crosses-- is not given, they are loaded and calculated. If
    given, it is ensured that only the 'single' condition survives.

    """
    if crosses is None:
        if data is None:
            data = imda.import_data(pc.all_parts, conditions=['single'])
        cdata = data.query('condition == "single"')
        crosses = trajectory_variance(cdata, cross_fun='halfway_s2s')
    return crosses.query('condition=="single"')


def hc_all_segments(fseqcrosses):
    """Calculate the coarticulation for all segments.

    This is done on a per-trial basis, calculating the crosses at the 0.25, 0.5
    and 0.75 part of the CtC. The coarticulation is measured as the difference
    between the point at 0.5 and the average between 0.25 and 0.75.

    Note that the input should be the output of halfway_crosses_fullseq.

    """
    posy = []
    indy = []
    rcross = []
    cols = ['part', 'condition', 'id_move', 'segment', 'trial']
    for index, rows in fseqcrosses.groupby(cols):
        hcs = rows['cross'].values
        relcross = (hcs[1] - ((hcs[0] + hcs[2]) / 2))
        rcross.append(relcross)
        posy.append(relcross < 0)
        indy.append(index)
    panda = pd.DataFrame()
    panda['posy'] = posy
    panda['relcross'] = rcross
    panda.loc[:, cols] = indy
    return panda


def add_speed_to_data(data, lowpass=False, **kwargs):
    """Add a speed column to the data."""
    butter = define_filter(**kwargs)
    for uid, datum in data.groupby('unique_trial_id'):
        dx = np.gradient(datum[['pos_x', 'pos_y']].values, axis=0)
        dt = np.gradient(datum['tframe'].values)
        speed = np.linalg.norm(dx / dt[:, None], axis=1)
        if lowpass:
            speed = signal.lfilter(*butter, speed)
        data.loc[datum.index, ['speed']] = speed


def filter_speed(data, groupby_column='unique_trial_id', **kwargs):
    """Filter the speed column in the data."""
    butter = define_filter(**kwargs)
    for uid, datum in data.groupby(groupby_column):
        speed = datum['speed']
        speed = signal.lfilter(*butter, speed)
        data.loc[datum.index, ['speed']] = speed


def zero_tframes(data):
    """Set the first tframe of each uid to zero."""
    for uid, datum in data.groupby('unique_trial_id'):
        first_tframe = datum.iloc[0]['tframe']
        tframes = datum['tframe'] - first_tframe
        data.loc[data['unique_trial_id'] == uid, ['tframe']] = tframes


def define_filter(low_freq=5, order=5):
    """Design a butterworth filter for speeds."""
    sample_frequency = 100
    nyq = sample_frequency / 2
    low = low_freq / nyq
    buttery = signal.butter(order, low)
    return buttery


def curvature_at_target(data, target='st_top', pad=5):
    """Find trajectories with abrupt changes of direction at target.

    Parameters
    pad : int
    Number of extra frames to use for the calculation, before and after those
    inside the target circle.

    """
    basket = utils.coords_from_svg(reference='mk_tr')
    uids = []
    parts = []
    id_moves = []
    curvatures = []
    trials = []
    for uid, datum in data.groupby('unique_trial_id'):
        traj = datum[['pos_x', 'pos_y']].values
        ix_intarget = basket[target](traj)
        if not ix_intarget.any():
            continue
        first_in, last_in = np.where(ix_intarget)[0][[0, -1]]
        first_out = np.where(ix_intarget[first_in:] == 0)[0][0] + first_in
        # ix_touse = np.arange(first_in - pad, last_in + pad, dtype=int)
        ix_touse = np.arange(first_in - pad, first_out + pad, dtype=int)
        ix_touse = ix_touse[ix_touse < len(traj)]
        intraj = traj[ix_touse]
        curvature = curvature_one_traj(intraj, datum.tframe.values[ix_touse])
        # ipdb.set_trace(cond=datum['id_move'].values[0] == 2)
        parts.append(datum['part'].values[0])
        id_moves.append(datum['id_move'].values[0])
        uids.append(uid)
        trials.append(datum['trial'].values[0])
        curvatures.append(np.max(np.abs(curvature)))
        # ipdb.set_trace()
    uids = np.array(uids, dtype=int)
    parts = np.array(parts, dtype=int)
    id_moves = np.array(id_moves, dtype=int)
    arrays = np.array([uids, parts, id_moves, trials])
    pandeaks = pd.DataFrame(arrays.T, columns=['unique_trial_id', 'part',
                                               'id_move', 'trial'])
    pandeaks.loc[:, ['curvature']] = curvatures
    return pandeaks


def curvature_one_traj(traj, tvec, cap=10):
    """Return the curvature of the trajectory."""
    dtraj = np.gradient(traj, tvec, axis=0)
    ddtraj = np.gradient(dtraj, tvec, axis=0)
    curvature = (dtraj * ddtraj[:, ::-1] * np.array([1, -1])).sum(axis=1) /\
        np.linalg.norm(dtraj, axis=1) ** 3
    # ipdb.set_trace(cond=np.isnan(curvature).any())
    return curvature[np.abs(curvature) < 10]


def curvatures_per_uid(data):
    """Calculate the curvature around primary and secondary target, per uid."""
    pandis = []
    for uid, datum in data.groupby('unique_trial_id'):
        # part, idmove, trial = datum.head(1)[['part', 'id_move', 'trial']].values[0]
        # if part == 401 and idmove == 2 and trial == 8:
        #     ipdb.set_trace()
        pandur_pt = curvature_at_target(datum, datum.second.values[0])
        pandur_st = curvature_at_target(datum, datum.third.values[0])
        pandur_pt.rename(columns={'curvature': 'curvature_pt'}, inplace=True)
        pandur_pt['curvature_st'] = pandur_st.curvature
        pandis.append(pandur_pt)
    return pd.concat(pandis)


def curvature_and_halfway(data, p_value=0.01):
    """Calculate, for each uid, curvature and halfway coarticulation.

    Additionally, label trajectories for which the halfway coarticulation is
    significantly different from zero ('halfway_sig' column).

    Returns
    -------
    pancur : pd.DataFrame
    Panda with columns unique_trial_id, part, id_move and curvature.

    pignificance

    """
    pancur = curvatures_per_uid(data)

    crosses = trajectory_variance(data, cross_fun='halfway_c2c')
    flip_crosses(crosses, rule=pc.flip_rule_half, inplace=True)
    *_, significance = classify_parts_signi(crosses)
    parts = []
    idmoves = []
    pvalues = []
    signi = []
    for key, value in significance.items():
        parts.append(key[0])
        idmoves.append(key[1])
        pvalues.append(value.pvalue)
        signi.append(value.pvalue <= p_value)
    avg_crosses = crosses.groupby(['part', 'id_move'])['cross'].mean()
    pignificance = pd.DataFrame(np.array([parts, idmoves, signi], dtype=int).T,
                                columns=['part', 'id_move', 'signi'])
    pignificance.loc[:, ['pvalue']] = pvalues
    chalf = pd.merge(pancur, pignificance, on=['part', 'id_move'], how='outer')
    chalf = pd.merge(chalf, avg_crosses, on=['part', 'id_move'], how='outer')
    chalf.rename(columns={'cross': 'avg_cross'}, inplace=True)
    chalf = pd.merge(chalf, crosses, on=['part', 'id_move', 'trial'],
                     how='outer')
    parts = []
    pars = {name: [] for name in data.columns
            if name.startswith('par_')}
    for part, datum in data.groupby('part'):
        parts.append(part)
        for name, listy in pars.items():
            listy.append(datum.iloc[0][name])
    mapanda = pd.DataFrame(parts, columns=['part'])
    for name, listy in pars.items():
        mapanda.loc[:, name] = listy
    allie = pd.merge(chalf, mapanda, on=['part'], how='inner')

    return pancur, pignificance, allie


def vpsoc3_curvature_vs_halfway(data=None):
    """Calculate curvature, halfway coart and all for vpsoc3.

    It calls curvature_and_halfway and adds the values of 'r' and 'h_...' to
    the resulting panda.

    """
    return_data_flag = 0
    if data is None:
        data = sims.vpsoc3_reference()
        imda.label_current_target(data)
        return_data_flag = 1
    curvature, psigni, chalf = curvature_and_halfway(data)
    parts = []
    rs = []
    hs = []
    h_labels = [col for col in data.columns if col.startswith('h_')]
    for part, datum in data.groupby('part'):
        parts.append(part)
        rs.append(datum.iloc[0]['r'])
        hs.append(datum.iloc[0][h_labels])
    mapanda = pd.DataFrame(hs, columns=h_labels)
    mapanda.loc[:, 'part'] = parts
    mapanda.loc[:, 'r'] = rs
    allie = pd.merge(chalf, mapanda, on=['part'], how='inner')
    if return_data_flag:
        return allie, data
    return allie


def hiseq_curvature_vs_halfway(data=None):
    """Calculate curvature, halfway coart and all for vpsoc3.

    It calls curvature_and_halfway and adds the values of 'r' and 'h_...' to
    the resulting panda.

    """
    return_data_flag = 0
    if data is None:
        data = sims.vpsoc3_reference()
        imda.label_current_target(data)
        return_data_flag = 1
    curvature, psigni, chalf = curvature_and_halfway(data)
    parts = []
    rs = []
    hs = []
    mondis = []
    for part, datum in data.groupby('part'):
        parts.append(part)
        rs.append(datum.iloc[0]['r'])
        hs.append(datum.iloc[0]['h'])
        mondis.append(datum.iloc[0]['mon_dis'])
    mapanda = pd.DataFrame(hs, columns=['h'])
    mapanda.loc[:, 'mon_dis'] = mondis
    mapanda.loc[:, 'part'] = parts
    mapanda.loc[:, 'r'] = rs
    allie = pd.merge(chalf, mapanda, on=['part'], how='inner')
    if return_data_flag:
        return allie, data
    return allie


def coart_vs_variability(crosses):
    """Calculate the correlation between hC(1) and variability in trajs."""
    gcrosses = crosses.groupby(['part', 'id_move'])
    means = gcrosses['cross'].mean().values
    stds = gcrosses['cross'].std().values
    # good_idx = means > 0
    # means = means[good_idx]
    # stds = stds[good_idx]
    pcorr = sp.stats.pearsonr(np.abs(means), stds, alternative='greater')
    regressy = sp.stats.linregress(np.abs(means), stds)
    return pcorr, regressy, means, stds


def envelope_per_r(chalf, xy_labels=None):
    """Find an envelope (min, max) for each value of r.

    Returns a function to interpolate the min, one for the max and a list of
    the min/max values found in --chalf--, in the format [[r, xmin, ymin, xmax,
    ymax], ...]. The returned min and max functions take x-values as input and
    return the interpolated y-values.

    Parameters
    ----------
    xy_labels : iterable, size=(2, )
    Labels in --chalf-- to be used as x-axis and y-axis in the data. Defaults
    to ['curvature_pt', 'curvature_st']

    """
    if xy_labels is None:
        xy_labels = ['curvature_pt', 'curvature_st']
    values = []  # (r, (minx, miny), (maxx, maxy))
    for c_r, datum in chalf.groupby('par_r'):
        curvies = datum[xy_labels].values
        idx_min = np.nanargmin(curvies[:, 1])
        idx_max = np.nanargmax(curvies[:, 1])
        values.append([c_r, *curvies[idx_min, :], *curvies[idx_max, :]])
    nvalues = np.array(values)
    min_coordies = nvalues[:, 1:3]
    min_ix_sorted = np.argsort(min_coordies[:, 0])
    max_coordies = nvalues[:, 3:]
    max_ix_sorted = np.argsort(max_coordies[:, 0])

    def minfun(x):
        """Return the minimum envelope evaluated at x."""
        return np.interp(x, *min_coordies[min_ix_sorted, ...].T)

    def maxfun(x):
        """Return the maximum envelope evaluated at x."""
        return np.interp(x, *max_coordies[max_ix_sorted, ...].T)

    return minfun, maxfun, nvalues


def model_dots_interpolate(data_chalf, model_chalf, cur_labels=None, vpsoc=False):
    """Find means in data_chalf that are not covered by model_chalf.

    For each mean in data_chalf, it finds its closest one in model_chalf and
    determines whether it's in the y-axis' 95% interval or not. If not, it's
    added to the list of dots to return.

    """
    if cur_labels is None:
        cur_labels = ['curvature_pt', 'curvature_st']
    baddies = []
    if vpsoc:
        infun_left, infun_right = vpsoc_area_interpolate(model_chalf)
    for (part, idmove), datum in data_chalf.groupby(['part', 'id_move']):
        modum = model_chalf.query('id_move == @idmove')
        if vpsoc:
            minfun = infun_left[idmove]
            maxfun = infun_right[idmove]
        else:
            minfun, maxfun, _ = envelope_per_r(modum, xy_labels=cur_labels)
        mean = datum[cur_labels].mean(axis=0).values
        interval = stats.bootstrap((datum[cur_labels[1]].values, ), np.mean,
                                   confidence_level=0.99)
        ci = interval.confidence_interval
        mod_int = (minfun(mean[0]), maxfun(mean[0]))
        cond1 = ci.low > mod_int[0] and ci.low < mod_int[1]
        cond2 = ci.high > mod_int[0] and ci.high < mod_int[1]
        if not (cond1 or cond2):
            baddies.append([part, idmove, mean])
    return baddies


def model_hc(data_chalf, model_chalf):
    """Calculate for each MoveID-part combo if hC is covered by the model."""
    baddies = []
    for (part, idmove), datum in data_chalf.groupby(['part', 'id_move']):
        modum = model_chalf.query('id_move == @idmove')
        hcs = modum['cross']
        mean = datum['cross'].mean()
        interval = stats.bootstrap((datum['cross'].values, ), np.mean,
                                   confidence_level=0.99)
        ci = interval.confidence_interval
        cond1 = ci.low > hcs.min() and ci.low < hcs.max()
        cond2 = ci.high > hcs.min() and ci.high < hcs.max()
        if not (cond1 or cond2):
            baddies.append([part, idmove, mean])
    return baddies


def blocky_interpolate(chalf, int_window=2, xy_labels=None):
    """Interpolate min and max using magic blocks."""
    if xy_labels is None:
        xy_labels = ['curvature_pt', 'curvature_st']
    xlab = xy_labels[0]
    data_min = chalf[xlab].min()
    data_max = chalf[xlab].max()
    bins = np.arange(-20, data_max, int_window)
    chalf.loc[:, 'bin'] = pd.cut(chalf[xlab], bins)
    xx = []
    ymin = []
    ymax = []
    for bin_edges, chalfum in chalf.groupby('bin'):
        curvies = chalfum[xy_labels[1]].values
        xx.append(bin_edges.left)
        xx.append(bin_edges.right)
        if len(chalfum) == 0:
            c_ymin = 0
            c_ymax = 0
        else:
            c_ymin = np.nanmin(curvies)
            c_ymax = np.nanmax(curvies)
        ymin.append(c_ymin)
        ymin.append(c_ymin)
        ymax.append(c_ymax)
        ymax.append(c_ymax)

    def minfun(x):
        """Return the minimum envelope evaluated at x."""
        return np.interp(x, xx, ymin)

    def maxfun(x):
        """Return the maximum envelope evaluated at x."""
        return np.interp(x, xx, ymax)

    return minfun, maxfun


def fit_chalf(chalf):
    """Fit a 1/x-type function to chalf, separated by id_move."""
    def fun(x, a1, b1):
        return a1 / (b1 + np.abs(x ** 2))  # + a2 / (b2 - x)
    popt = {}
    for idmove, chalfum in chalf.groupby('id_move'):
        x = chalfum['cross'].values
        y = chalfum['curvature_st'].values
        popt[idmove], *_ = sp.optimize.curve_fit(fun, x, y)
    return popt, fun


def distance_from_point_to_segment(px, py, ax, ay, bx, by):
    """Calculate the signed distance between points and a curve.

    Written by ChatGPT.
    """
    # Vector AB and AP
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay
    # Dot product of AB and AP
    AB_AP = ABx * APx + ABy * APy
    # Squared length of AB
    AB_AB = ABx * ABx + ABy * ABy
    # Projection factor of AP onto AB
    t = AB_AP / AB_AB
    # If t is outside the segment, clamp it to 0 or 1
    if t < 0:
        closest_x, closest_y = ax, ay
    elif t > 1:
        closest_x, closest_y = bx, by
    else:
        closest_x = ax + t * ABx
        closest_y = ay + t * ABy
    # Euclidean distance from (px, py) to the closest point on the segment
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    return dist


def cross_product(ax, ay, bx, by, px, py):
    """Calculate the cross product or sumetin.

    Written by ChatGPT.

    Cross product to determine the
    side (left or right) of the
    point relative to the segment

    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def furthest_points_from_curve_sorted(dots, curve):
    """Calculate a signed distance between dots and the curve.

    The sign refers to whether they're to the left (negative) or right
    (positive) of the curve (in the local coordinates of the curve, which vary
    depending on the order of the points used to define the curve).

    What's returned here is the left-most and the right-most point of each
    cluster, which means the most negative and the most positive,
    respectively. The clusters are defined in terms of the 'par_r' value.

    Started with ChatGPT, modified and adapted by me.


    Returns
    -------
    furthest_left(/right)_point : ndarray
    Left-most (right-most) point in --dots-- from the curve.

    """
    dists = []
    all_dots = []
    for dot in dots:
        px, py = dot

        for i in range(len(curve) - 1):
            ax, ay = curve[i]
            bx, by = curve[i + 1]
            c_dist = distance_from_point_to_segment(px, py, ax, ay, bx, by)
            c_cp = cross_product(ax, ay, bx, by, px, py)
            all_dots.append(dot)
            dists.append(c_dist * c_cp)
    dists_sorted = np.argsort(dists)
    furthest_left_point = all_dots[dists_sorted[0]]
    furthest_right_point = all_dots[dists_sorted[-1]]
    return furthest_left_point, furthest_right_point


def model_area_by_cluster_vpsoc(chalf):
    """Calculate the area of coverage of the model's chalf.

    This version fits the model's chalf to a 1/x curve and uses that as a
    reference line to then calculate a series of points that will minimally
    cover all the points in chalf.


    """
    popt, hyperbola = fit_chalf(chalf)

    xhyp = np.linspace(-30, 0, 100)
    lefts = {}
    rights = {}
    for (id_move, c_r), chalfum in chalf.groupby(['id_move', 'par_r']):
        x_range = (chalfum['cross'].min(), chalfum['cross'].max())
        x_mid = (x_range[0] + x_range[1]) / 2
        x_range_big = 2 * (x_range - x_mid) + x_mid
        xhyp = np.linspace(*x_range_big, 10)
        yhyp = hyperbola(xhyp, *popt[id_move])
        xyhyp = np.array([xhyp, yhyp])
        dots = chalfum[['cross', 'curvature_st']].values
        lefty, righty = furthest_points_from_curve_sorted(dots, xyhyp.T)
        lefty[1] = 0  # HACK!
        if id_move in lefts.keys():
            lefts[id_move].append(lefty)
            rights[id_move].append(righty)
        else:
            lefts[id_move] = [lefty]
            rights[id_move] = [righty]
    return lefts, rights


def interpolate_function(xvals):
    """Return a function to interpolate the curve created by xvals.

    NaNs are gotten rid of because they're yucky and nobody likes them.
    """
    ixy_nan = [np.any([xvalito is None for xvalito in xval])
               for xval in xvals]
    new_vals = np.array(xvals)[~np.array(ixy_nan)].astype(float)
    ix_sorted = np.argsort(new_vals[:, 0])
    sorted_vals = new_vals[ix_sorted]

    def infun(x):
        return np.interp(x, *sorted_vals.T)
    return infun


def vpsoc_area_interpolate(chalf, xy_labels=None):
    """Get functions to interpolate the curves for vpsoc chalf coverage.

    IOU explanation.

    """
    if xy_labels is None:
        xy_labels = ['cross', 'curvature_st']
    lefts, rights = model_area_by_cluster_vpsoc(chalf)
    infun_left = {}
    infun_right = {}
    for idmove, chalfum in chalf.groupby('id_move'):
        c_lefts = lefts[idmove]
        c_rights = rights[idmove]
        infun_left[idmove] = interpolate_function(c_lefts)
        infun_right[idmove] = interpolate_function(c_rights)
    return infun_left, infun_right


def model_parts(data_chalf, vpsoc_chalf, hiseq_chalf):
    """Find participant-id_move combinations not covered by the models.

    Returns
    -------
    Returns ((hiseq_hc_bad, hiseq_cc_bad, hiseq_cu_bad),
    (vpsoc_hc_bad, vpsoc_cc_bad, vpsoc_cu_bad)), where:

    XX_hc_bad : list
    Output of model_hc, for model XX.

    XX_cc_bad : list
    Output of model_dots_interpolate for hC1 vs curvature_st, for model
    XX.

    XX_cu_bad : list
    Output of model_dots_interpolate for curvature_pt vs curvature_st, for
    model XX.
    """
    # About hC1
    hiseq_hc_bad = model_hc(data_chalf, hiseq_chalf)
    vpsoc_hc_bad = model_hc(data_chalf, vpsoc_chalf)

    # About hC1 vs curv1
    cur_labels = ['cross', 'curvature_st']
    hiseq_cc_bad = model_dots_interpolate(data_chalf, hiseq_chalf,
                                          cur_labels=cur_labels,
                                          vpsoc=False)
    vpsoc_cc_bad = model_dots_interpolate(data_chalf, vpsoc_chalf,
                                          cur_labels=cur_labels,
                                          vpsoc=True)

    # About curv1 vs curv2
    hiseq_cu_bad = model_dots_interpolate(data_chalf, hiseq_chalf)
    vpsoc_cu_bad = model_dots_interpolate(data_chalf, vpsoc_chalf)

    return ((hiseq_hc_bad, hiseq_cc_bad, hiseq_cu_bad),
            (vpsoc_hc_bad, vpsoc_cc_bad, vpsoc_cu_bad))
