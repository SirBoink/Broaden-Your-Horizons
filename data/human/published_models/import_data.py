"""Multiple functions to import the behavioral data.

Some other functions to transform the data are included.
"""
import os
import glob
import re

import pandas as pd
import numpy as np

from src import utils
from src.model import single

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath('./data/')

SPEED_THRESHOLD = 50  # mm/s
MAINTAIN = 10  # frames
DT = 0.01  # framerate from QTM.

DATA_FILE = './all_data.csv'


def load_data():
    """Load the csv file with all the data.

    The only advantage of this is speed. This csv file has to have been created
    manually and might not be up-to-date. Use import_data() instead if you want
    to import from scratch.
    """
    return pd.read_csv(DATA_FILE, index_col=0)


def save_data(data):
    """Save a csv file with --data--.

    Use this to save the data from import_data(), for future fast loading with
    load_data(). Speed is the only advantage.

    """
    data.to_csv(DATA_FILE)


def import_data(participants=None, conditions=None, include_failed=False,
                add_motors=False, norm_frames=False, motor_kwargs=None,
                data_folder=None, fix_frames=True, add_ctarget=True,
                interpolate=True):
    """Import the requested data files into a big happy panda.

    Parameters
    ----------
    participants : iterable[int]
    Participant numbers to take.

    conditions : list[str]
    List containing 'single' and/or 'sequential'. If None, both are imported.

    include_failed : bool
    Whether to include failed trials in the data.

    add_motors : bool
    Whether to infer motor commands from the data using an arm model. This
    calls motor_from_datum, and its parameters can be passed with
    --motor_kwargs--. The motor commands are added as the columns ux_t and
    ux_y.  Defaults to True.

    fix_frames : bool
    Infers how many QTM frames (and thus time) passed between consecutive rows
    in the data, to better estimate the speed and averages. See
    normalize_time() for more info. Adds 'dt' column, and modifies the 'frames'
    column.

    norm_frames : bool
    Whether to normalize time by setting the frame to zero when the speed has
    reached a minimum threshold. If True, normalize_time() is called with
    reasonable default parameters. If you want to change parameters, set this
    to False and run the function separately.

    motor_kwargs : dict
    Parameters for the function motor_from_datum. Used if --add_motors-- is
    True.

    add_ctarget : bool
    If True, label_current_target() is called on the data to add to each frame
    the current target (circle).

    interpolate : bool
    If True, the data is interpolated between tframe.min to tframe.max, for
    each trial separately. This is useful e.g. for running low-pass filters on
    the speed. It takes about a minute to do on the full data set.

    Returns
    -------
    pandata : pd.DataFrame
    Contains the data for all requested participants and conditions. Each row
    is one measurement and the columns are all those in the data. Nothing is
    added here except 'part', which is the participant number.

    """
    if motor_kwargs is None:
        motor_kwargs = {}
    if data_folder is None:
        data_folder = DATA_FOLDER
    if participants is None:
        participants = find_participants(data_folder=data_folder)
        if len(participants) == 0:
            messy = f'No participant data found in {data_folder}'
            raise ValueError(messy)
    pandota = []
    for part in participants:
        filename = data_folder + '/data_{}.csv'
        pandita = pd.read_csv(filename.format(part))
        pandita['part'] = part
        if part < 300:
            change_labels(pandita)
        else:
            fix_condition(pandita)
        if conditions is not None:
            pandita = pandita.query('condition in @conditions')
        if add_motors:
            pandita = motor_from_datum(pandita, **motor_kwargs)
        pandota.append(pandita)
    pandota = pd.concat(pandota, ignore_index=True)
    pandota['unique_trial_id'] = unique_run_identifier(pandota)
    if not include_failed:
        pandota = pandota.query('success')
    if norm_frames:
        normalize_time(pandota, speed_threshold=200, maintain=10)
    if fix_frames:
        do_fix_frames(pandota, inplace=True)
    if add_ctarget:
        label_current_target(pandota)
    if interpolate:
        interpolate_data(pandota)
    return pandota


def do_fix_frames(pandata, inplace=True, force_unique_id=False):
    """Fix the problem with not recording dt in the data.

    The data points recorded in psychopy are not always at even intervals due
    to programming error. This leads to the speed seemingly doubling from one
    row to the next in the pandas.

    To fix this, a new column dt is created with integers, where the value for
    each row is how many frames this measurement was held, with 1 being the
    default. For any row idx, id[idx] is a number (typically 1 or 2) such that
    the right-speed is calculated as (pos[idx + 1] - pos[idx]) / dt[idx].

    Additionally, for compatibility with proj_switch and proj_hands, a column
    tframe is created. This works as a timestamp in seconds, starting at 0 for
    each participant and running until the end of the recording for this
    participant. It respects the newly-calculated dt.

    This function assumes that the panda already has the 'unique_trial_id'
    column. If --force_unique_id-- is True, unique_run_identifier will be
    called to add it. Otherwise, an error is thrown.

    """
    if not inplace:
        pandata = pandata.copy()
    if 'unique_trial_id' not in pandata.columns:
        if force_unique_id:
            unique_run_identifier(pandata, append=True)
        else:
            raise ValueError('The panda provided is missing the '
                             '"unique_trial_id" column. Run the '
                             'unique_run_identifier function to add it.')
    pandata.loc[:, ['dt']] = 1.0
    for unique_id in pd.unique(pandata['unique_trial_id']):
        datum = pandata.query('unique_trial_id == @unique_id')
        c_dt = _fix_frames_one(datum)
        pandata.loc[datum.index, 'dt'] = c_dt
        pandata.loc[datum.index, 'frame'] += np.cumsum(c_dt - 1)

    # Add tframe
    pandata.loc[:, ['tframe']] = 0
    for part, datum in pandata.groupby('part'):
        tframe = np.roll(np.cumsum(datum['dt'].values) * DT, 1)
        tframe[0] = 0
        pandata.loc[datum.index, ['tframe']] = tframe
    if not inplace:
        return pandata


def _fix_frames_one(pandatum):
    """Calculate the new dt column for one run.

    Hack(1): The first two positions of each trial are exactly the same, for
    some weird reason. To avoid warnings about division by zero, I added a
    +1e-5 to speeds, which affects nothing (I hope). Need to fix this
    elsewhere.

    """
    diff = np.diff(pandatum.loc[:, ['pos_x', 'pos_y']], axis=0)
    c_dt = pandatum['dt'][:-1].values
    qspeed = np.linalg.norm(diff, axis=1) + 1e-5  # Hack(1)!
    if (qspeed == 0).any():
        ipdb.set_trace()
    ix_test = 0
    double = np.array([True])  # changes size later in loop
    while double.any():
        if ix_test > 10:
            break
        c_speed = qspeed / c_dt
        ratio = c_speed[1:] / c_speed[:-1]
        double = np.abs(ratio - 2) < 0.4
        c_dt[np.where(double)] = 0.5
        # ipdb.set_trace()
        ix_test += 1
    c_dt = np.concatenate([c_dt, [1]])
    return (c_dt * 2).astype(int)


def fix_condition(pandata, inplace=True):
    """Fix the problem with all conditions being sequential in 400 series.

    Somehow, the experiment records all trials as being sequential in the 400
    series of the experiment. This function fixes those labels, making all
    trials without a third target a single-target trial.

    Works in-place unless otherwise specified.

    """
    if not inplace:
        pandata = pandata.copy()

    single = pandata['third'].isnull()
    if not single.any():
        single = pandata['third'] == 'none'
    condition = np.where(single, 'single', 'sequential')
    pandata['condition'] = condition
    if not inplace:
        return pandata


def normalize_time(data, speed_threshold=None, maintain=None):
    """Eliminate all frames before movement starts.

    The normalization is done for each participant and trial independently.

    All frames for which the speed is lower than --speed_threshold-- will be
    eliminated and the first one that remains will be marked as the zero-th,
    and all others adjusted accordingly. This is done to be able to average
    trajectories and not have nonsensical curves.

    To avoid being tricked by insignificant yanks, the speed needs to be above
    the threshold for at least --maintain-- frames.

    """
    if speed_threshold is None:
        speed_threshold = SPEED_THRESHOLD
    if maintain is None:
        maintain = MAINTAIN
    unique_ids = pd.unique(data['unique_trial_id'])
    data.loc[:, ['elim']] = 1
    for unid in unique_ids:
        datum = data.query('unique_trial_id == @unid')
        flag_found, c_index = _normalize_time_one(datum,
                                                  speed_threshold,
                                                  maintain)
        if not flag_found:
            message = ('The speed was never maintained above threshold '
                       'for trial {}. Aborting.')
            raise ValueError(message.format(unid))
        data.loc[c_index, ['elim']] = 0
        data.loc[c_index, ['frame']] = np.arange(len(c_index))
        data.loc[c_index, ['tframe']] -= datum.loc[c_index[0], ['tframe']]
    data.drop(data.loc[data['elim'] == 1].index, inplace=True)


def _normalize_time_one(data, speed_threshold=None, maintain=None):
    """Eliminate all frames before movement starts.

    Assumes that only one run is included (i.e. one participant, one id_move,
    one trial).

    """
    # diffy = np.diff(data.loc[:, ['pos_x', 'pos_y']].values, axis=0)
    # speed = np.concatenate([[0], np.linalg.norm(diffy * 100, axis=1)])
    grady = np.gradient(data.loc[:, ['pos_x', 'pos_y']].values, axis=0)
    speed = np.linalg.norm(grady * 100, axis=1)
    thresed = speed >= speed_threshold
    # kept = np.diff(thresed, maintain)
    # kept_high = kept * thresed[maintain:]
    flag_found = 0
    found = np.where(np.diff(np.where(thresed == False)[0]) >= maintain)
    if len(found[0]) > 0:
        idx = found[0][0]
        flag_found = 1
    # ipdb.set_trace()
    c_index = data.index[idx:]
    if not flag_found:
        raise ValueError()
    return flag_found, c_index


def unique_run_identifier(data, append=True):
    """Return a numerical identifier for each trial.

    This number is uniquely determined by the participant number, condition,
    id_move and trial number using the following logic:

    All numbers start with a 1, to avoid python freaking out about leading
    zeroes in an int. From there: The first four digits are the participant
    numbers (with added leading zeroes, if needed). The fifth digit is
    condition (0 for single-target, 1 for sequential). The sixth and seventh
    are id_move (with leading zero, if needed) and the last three digits are
    trial number (with leading zeroes, if needed).

    """
    part = data['part']
    cond = data['condition'] == 'sequential'
    idmove = data['id_move']
    trial = data['trial']

    def formy(x, n):
        temp = f'{{:0{n}d}}'
        return temp.format(x)
    that = []
    for this, n in zip([part, cond, idmove, trial], [4, 1, 2, 3]):
        that.append(this.map(lambda x: formy(x, n)).values)
    final = '1' + np.sum(that, axis=0)
    finalint = final.astype(int)
    if append:
        data['unique_trial_id'] = finalint
    return finalint


def reverse_uid(uid):
    """Return the information in the uid."""
    uid = str(uid)
    part = int(uid[1:5])
    condition = 'sequential' if int(uid[5]) else 'single'
    id_move = int(uid[6:8])
    trial = int(uid[8:])
    return part, condition, id_move, trial


def populate_uid_info(data):
    """Populate part, condition, trial and id_move given the uid.

    The function adds the columns (part, ...) and the corresponding information
    to each row.

    Parameters
    ----------
    data : pd.DataFrame
    Must have a column called unique_trial_id.

    """
    for uid, datum in data.groupby('unique_trial_id'):
        info = reverse_uid(uid)
        for col, value in zip(['part', 'condition', 'id_move', 'trial'], info):
            data.loc[datum.index, [col]] = value


def find_participants(data_folder=None):
    """Return a list of participants for which the experimental data exists."""
    if data_folder is None:
        data_folder = DATA_FOLDER
    files = glob.glob(data_folder + '/*.csv')
    participants = []
    for c_file in files:
        try:
            participants.append(int(re.search(r'[0-9]+', c_file)[0]))
        except TypeError:
            pass

    return sorted(participants)


def motor_from_datum(pandata, arm_model=None, inplace=False):
    """Extract the motor commands from the data in --pandata--.

    A copy of --pandata-- is made unless --inplace-- is True, and eventually
    returned.

    It assumes that each dimension in the movement is controlled independently
    by a linear model.

    The resulting motor commands are added to the panda as ux_t and uy_t.


    Parameters
    ----------
    pandata : pd.DataFrame
    Contains thsime trajectories from a participant or simulations.

    Returns
    -------
    pandata : pd.DataFrame
    Modified pandata with added motor commands. A copy of the original,
    if --inplace-- is False.

    """
    if arm_model is None:
        arm_model = single.Arm(num_dim=2)
    if not inplace:
        pandata = pandata.copy()
    basket = utils.coords_from_svg(reference='mk_tr')

    data = pandata.query('condition == "single" and success == True')
    x51_labels = [f'x51_{idx}' for idx in range(5)]
    x52_labels = [f'x52_{idx}' for idx in range(5)]
    id_moves = np.unique(data['id_move'])
    for id_move in id_moves:
        datum = data.query('id_move == @id_move')
        last = datum.iloc[0]['second']
        x_end = basket[last].center
        for trial in np.unique(datum['trial']):
            datumtum = datum.loc[datum['trial'] == trial, ['pos_x', 'pos_y']]
            poses = datumtum.to_numpy()
            x_t, u_t = arm_model.inv_dynamics(poses)
            pandata.loc[datumtum.index, ['ux_t', 'uy_t']] = u_t
            x_t[:, 0, -1] = x_end[0]
            x_t[:, 1, -1] = x_end[1]
            pandata.loc[datumtum.index, x51_labels] = x_t[:, 0, :]
            pandata.loc[datumtum.index, x52_labels] = x_t[:, 1, :]
    return pandata


def normalize_trajectories(pandata, trim_range=None, inplace=False):
    """Trajectory normalization for inference.

    For each trajectory, it calculates the mid-point (max speed) and sets
    that as the middle point in time as well, and cuts the tails off.

    """
    if not inplace:
        pandata = pandata.copy()
    pandata['frame'] = 0
    data = pandata.query('condition == "single" and success == True')
    id_moves = np.unique(pandata['id_move'])
    for id_move in id_moves:
        datum = data.query('id_move == @id_move')
        for trial in np.unique(datum['trial']):
            datumtum = datum.loc[datum['trial'] == trial]
            poses = datumtum.loc[:, ['pos_x', 'pos_y']]
            speeds = np.diff(poses, axis=0)
            idx_max = np.argmax(np.linalg.norm(speeds, axis=1))
            frames = np.arange(len(datumtum))
            frames += 35 - idx_max
            pandata.loc[datumtum.index, ['frame']] = frames
    if trim_range is not None:
        pandata.drop(pandata.loc[pandata['frame'] >= trim_range[1]].index,
                     inplace=True)
        pandata.drop(pandata.loc[pandata['frame'] < trim_range[0]].index,
                     inplace=True)
    return pandata


def change_labels(pandata, inplace=True):
    """Change the labels from the old labeling system to the new.

    The changes the color-based labels of the circles to the new
    position-based ones in the following way:
    pt_blue -> pt_left
    pt_red -> pt_right
    st_olive -> st_left
    st_lila -> st_top
    st_pink -> st_right

    Changes are done in-place unless otherwise specified.
    """
    if not inplace:
        pandata = pandata.copy()
    old_labels = ['pt_blue', 'pt_red', 'st_olive', 'st_lila', 'st_pink']
    new_labels = ['pt_left', 'pt_right', 'st_left', 'st_top', 'st_right']
    for old, new in zip(old_labels, new_labels):
        for thing in ['first', 'second', 'third']:
            condition = pandata[thing] == old
            if len(condition) != 0:
                pandata.loc[condition, thing] = new
    if not inplace:
        return pandata


def label_current_target(data):
    """Infer and label the current target in the trajectory.

    Given  each  trial (unique_trial_id),  use  some  heuristics to  label  the
    current target circle for each frame in the data.

    """
    unique_ids = pd.unique(data['unique_trial_id'])
    basket = utils.coords_from_svg(reference='mk_tr')
    ordinals = [ordi for ordi in ['second', 'third', 'fourth']
                if ordi in data.columns]
    for unique_id in unique_ids:
        # ipdb.set_trace()
        condi = data['unique_trial_id'] == unique_id
        datum = data.loc[condi]
        c_target = np.empty(len(datum), dtype=object)
        targets = datum.iloc[0][ordinals].values
        p_cutoff = 0
        for target in targets:
            # ipdb.set_trace()
            if pd.isnull(target):
                break
            # last_target = target  # For later
            circle = basket[target]
            traj = datum.loc[:, ['pos_x', 'pos_y']].values
            ixs_found = circle(traj)
            found = np.where(ixs_found)[0]
            try:
                found_first = found[found > p_cutoff][0]
            except IndexError:
                # Target never reached.
                c_target[p_cutoff:] = target
                break
            exit_all = np.where(ixs_found == 0)[0]
            if exit_all.max() < found_first:
                cutoff = len(datum)
            else:
                exit_first = exit_all[exit_all > found_first][0]
                cutoff = int((found_first + exit_first) / 2)
            c_target[p_cutoff:cutoff] = target
            data.loc[condi, ['ctarget']] = c_target
            # ipdb.set_trace()
            p_cutoff = cutoff


def interpolate_data(pandata):
    """Interpolate values missing in tframe.

    It assumes a dt of 0.01, and for each uid, it makes sure that all values of
    tframe between its minimum and maximum are present, i.e. that tframe[i+1] -
    tframe[i] = 0.01, for all i. Missing data is linearly interpolated.

    """
    cols_extra = ['pos_x', 'pos_y', 'tframe']
    cols_same = [col for col in pandata.columns
                 if col not in cols_extra]
    all_pandas = []
    for uid, datum in pandata.groupby('unique_trial_id'):
        tframe_min = datum['tframe'].min()
        tframe_max = datum['tframe'].max()
        all_tframes = np.arange(tframe_min * 100, tframe_max * 100) / 100
        new_df = pd.DataFrame({'tframe': all_tframes})
        datum.tframe = np.round(datum.tframe * 1000).astype(int)
        new_df.tframe = np.round(new_df.tframe * 1000).astype(int)
        new_datum = pd.merge(new_df, datum[cols_extra], on=[
                             'tframe'], how='left')
        new_datum.tframe = new_datum.tframe / 1000
        new_datum.interpolate(method='from_derivatives', inplace=True)
        for col in cols_same:
            new_datum.loc[:, [col]] = datum.iloc[0][col]
        all_pandas.append(new_datum)
    return pd.concat(all_pandas, ignore_index=True)


def test_fix_speed_one(x, mask):
    """Calculate the new dt column for one run."""
    pos = np.concatenate([x[None, :], x[None, :]]).T
    diff = np.diff(pos, axis=0) * mask
    # diff = np.concatenate([diff_1[None, :], diff_1[None, :]]).T
    c_dt = np.ones(len(diff))
    qspeed = np.linalg.norm(diff, axis=1)
    ix_test = 0
    double = np.array([True])  # changes size later in loop
    while double.any():
        if ix_test > 5:
            print('oh no!')
            break
        c_speed = qspeed / c_dt
        ratio = c_speed[1:] / c_speed[:-1]
        double = np.abs(ratio - 2) < 0.3
        c_dt[np.where(double)] = 0.5
        # ipdb.set_trace()
        ix_test += 1
        print(c_speed)
        print(c_dt)
    c_dt = np.concatenate([c_dt, [1]])
    return c_dt


def detect_failed_trials(data):
    """Detect and mark trials in which targets were not reached.

    This is mostly done for simulations (though it can be used with
    experimental data). For each trial, it checks whether all targets were
    reached. If not, it sets the 'success' column to False.

    Note that this creates and/or writes on the column 'success'. If it already
    exists, its values will be overwritten and lost forever.

    Parameters
    ----------
    data : pd.DataFrame
    Assumed to be like participant data. The necessary columns are 'unique_trial_id',
    'first, 'second, 'third', 'pos_x' and 'pos_y'.

    """
    basket = utils.coords_from_svg('mk_tr')
    for uid, datum in data.groupby(['unique_trial_id']):
        traj = datum[['pos_x', 'pos_y']].values
        flag_success = True
        for target in datum.head(1)[['first', 'second', 'third']].values[0]:
            if not basket[target](traj).any():
                flag_success = False
                break
        data.loc[datum.index, ['success']] = flag_success


def prune_failed_parts(data, threshold=1, reset_part=False):
    """Get rid of parts who had at least one failed trial.

    Note that this needs the column 'success' in --data-- to be meaningful,
    which is not the case by default in simulated data.

    Alternatively, you can use the ratio to determine which get pruned. So,
    participants are kept if the failed/total ratio is lower than
    --threshold--.

    If --reset_part-- is True, the participant numbers are reset to range(0,
    N), where N is the number of surviving parts. The order is preserved.

    """
    succ_sims = data.groupby('part')[['success']].mean()
    above = succ_sims['success'] >= threshold
    good_idxs = succ_sims.reset_index().loc[above, 'part'].values
    data = data.query('part in @good_idxs')
    if reset_part:
        unique_survivors = np.unique(data.part)
        relabel_dict = {survy: idx for idx,
                        survy in enumerate(unique_survivors)}
        data.part = data['part'].map(lambda x: relabel_dict[x]).values
    return data
