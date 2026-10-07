"""Different tools for simulations using the agents and stuff."""
import logging
from itertools import product

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.model import hierarchical as hie
from src import utils
from src.model import single
from src import import_data as imda
from src import paper_conf as pc


def multiple_runs_monitor_seq(agent, t_end, ix_seq, num_runs, x0=None,
                              background_noise=0.2, max_retry=1000,
                              dyn_kwargs=None, **kwargs):
    """Simulate multiple runs with sequential movements.

    Has the --agent-- do multiple runs of the simulation using the monitor
     to trigger the next element in the sequence, and returns it all
    in one beautiful panda.

    **kwargs is there for compatibility with multiple_runs_monitor_single

    Parameters
    ----------
    max_retry : int
    Maximum number of retries for integration. This helps with the stupid SHS
    which keeps becoming infinite for no good reason. Note that this number
    accumulates throughout all runs.

    dyn_kwargs : dict
    Parameters for the dynamics() method in SOC and its siblings. Use this to
    simulate trajectories without noise, even if the controller itself
    considered noise. Handy.

    """
    pandata = pd.DataFrame()
    if x0 is None:
        x0 = agent.set_initial_conditions(agent.sequences[ix_seq],
                                          background_noise=background_noise)
        x0[:agent.phys_dim] = agent.centers[agent.sequences[ix_seq][0]]
    agent.choose_seq(ix_seq)
    all_x = []
    all_run = []
    all_t = []
    all_frames = []
    retry = 0
    for ix_run in range(num_runs):
        while retry <= max_retry:
            try:
                datum = agent.move(x_t=x0, t_ini=0, t_end=t_end,
                                   id_stop='pt_start', dyn_kwargs=dyn_kwargs)
                break
            except TypeError as err:
                logging.warning(f'Integration crashed in {ix_run}. Retrying')
                retry += 1
                if retry > max_retry:
                    logging.warning('Number of retries exceeded max.')
                    raise err
        all_x.append(datum[0])
        all_run.append(ix_run * np.ones(len(datum[0])))
        all_t.append(datum[1])
        all_frames.append(np.arange(len(datum[0])))
    all_x = np.concatenate(all_x)
    # transitions = agent.transitions[seq_elms]
    # first = agent.labels[transitions[1][0]]
    # second = agent.labels[transitions[1][1]]
    # third = agent.labels[transitions[2][1]]
    pandata['angle_1'] = all_x[..., 0]
    pandata['angle_2'] = all_x[..., 1]
    seq_elms = agent.sequences[ix_seq]
    ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']
    for key, element in zip(ordinals, seq_elms):
        pandata[key] = agent.labels[agent.transitions[element][-1]]
    # pandata['first'] = first
    # pandata['second'] = second
    # pandata['third'] = third
    pandata['pos_x'], pandata['pos_y'] = agent.angles_to_xy(all_x[..., 0],
                                                            all_x[..., 1])
    pandata['t'] = np.concatenate(all_t)
    pandata['frame'] = np.concatenate(all_frames)
    pandata['tframe'] = pandata['frame'] * 0.01  # frame * dt
    pandata['trial'] = np.concatenate(all_run).astype(int)
    pandata['id_move'] = ix_seq
    pandata['success'] = True
    pandata['condition'] = 'sequential'
    pandata['part'] = 0
    imda.unique_run_identifier(pandata)
    num_shs = all_x.shape[-1] - 2
    for ix_shs in range(num_shs):
        pandata[f'shs_{ix_shs}'] = all_x[..., 2 + ix_shs]
    return pandata


def multiple_runs_monitor_single(agent, t_end, ix_single, num_runs, x0=None,
                                 **kwargs):
    """Simulate multiple runs with single-target movements.

    Has the --agent-- do multiple runs of the simulation for single-target
    movements.

    **kwargs is there for compatibility with multiple_runs_monitor_seq
      and is ignored.

    """
    pandata = pd.DataFrame()
    if x0 is None:
        x0 = agent.set_initial_conditions(agent.sequences[0])
        x0[:agent.phys_dim] = agent.centers[agent.transitions[ix_single][0]]
    all_x = []
    all_run = []
    all_t = []
    all_frames = []
    for ix_run in range(num_runs):
        datum = agent.move_single(x_t=x0, ix_single=ix_single, t_ini=0,
                                  t_end=t_end)
        all_x.append(datum[0])
        all_run.append(ix_run * np.ones(len(datum[0])))
        all_t.append(datum[1])
        all_frames.append(np.arange(len(datum[0])))
    all_x = np.concatenate(all_x)
    pandata['angle_1'] = all_x[..., 0]
    pandata['angle_2'] = all_x[..., 1]
    transitions = agent.transitions[ix_single + 1]
    first = agent.labels[transitions[0]]
    second = agent.labels[transitions[1]]
    third = None
    pandata['first'] = first
    pandata['second'] = second
    pandata['third'] = third
    pandata['pos_x'], pandata['pos_y'] = agent.angles_to_xy(all_x[..., 0],
                                                            all_x[..., 1])
    pandata['t'] = np.concatenate(all_t)
    pandata['frame'] = np.concatenate(all_frames)
    pandata['tframe'] = pandata['frame'] * 0.01
    pandata['trial'] = np.concatenate(all_run).astype(int)
    pandata['id_move'] = ix_single
    pandata['success'] = True
    pandata['condition'] = 'single'
    pandata['part'] = 1
    imda.unique_run_identifier(pandata)
    num_shs = all_x.shape[-1] - 2
    for ix_shs in range(num_shs):
        pandata[f'shs_{ix_shs}'] = all_x[..., 2 + ix_shs]
    return pandata


def multiple_runs_monitor(condition='sequential', **kwargs):
    """Simulate multiple runs with either single or seq.

    Wrapper that selects between the single- or sequential-movement
    simulations.
    """
    if condition == 'sequential':
        return multiple_runs_monitor_seq(**kwargs)

    try:
        kwargs['ix_single'] = kwargs['ix_seq']
        del kwargs['ix_seq']
    except:
        pass
    return multiple_runs_monitor_single(**kwargs)


def all_seqs(agent=None, num_runs=10, t_end=0.7, **kwargs):
    """Simulate the agent with all sequences.

    Parameters
    ----------
    *args and **kwargs are sent to multiple_runs_monitor.
    """
    if agent is None:
        soc_pars = {'sigma_c': 1}
        agent = hie.SimpleSOC(t_ends=t_end, soc_pars=soc_pars)
        agent.monitor_distance = 30
        agent.monitor_sd = 5
        agent.tau23 = 3
        agent.tau1 = 0.7
    num_moves = agent.sequences.shape[0]
    ix_seqs = np.arange(num_moves, dtype='int')
    all_pandas = []
    for c_seq in ix_seqs:
        pandita = multiple_runs_monitor(condition='sequential', agent=agent,
                                        t_end=t_end * 3.5, ix_seq=c_seq,
                                        num_runs=num_runs, **kwargs)
        all_pandas.append(pandita)
    return pd.concat(all_pandas), agent


def all_sings(agent=None, ix_sings=None, num_runs=20, t_end=0.7, **kwargs):
    """Run all single-target movements."""
    if agent is None:
        soc_pars = None
        agent = hie.SimpleSOC(t_ends=t_end, soc_pars=soc_pars)
        agent.monitor_distance = 1
        agent.monitor_sd = 0
        agent.tau23 = 5.5
    if ix_sings is None:
        # num_moves = agent.transitions.shape[0] - 1  # to account for [0, 0]
        ix_sings = [0, 1]  # np.arange(num_moves, dtype='int')
    all_pandas = []
    for ix_sing in ix_sings:
        pandita = multiple_runs_monitor(condition='single', agent=agent,
                                        t_end=t_end * 2.5, ix_single=ix_sing,
                                        num_runs=num_runs, **kwargs)
        pandita['id_move'] = ix_sing  # to account for [0, 0]
        all_pandas.append(pandita)
    return pd.concat(all_pandas), agent


def simulate_agent(agent=None, **kwargs):
    """Simulate one experimental session with the agent.

    The resulting data
    should be indistinguishable from a real participant.

    """
    single, _ = all_sings(agent=agent, num_runs=20, **kwargs)
    sequen, _ = all_seqs(agent=agent, num_runs=20, **kwargs)
    pandout = pd.concat([single, sequen], axis='index')
    pandout.set_index(np.arange(len(pandout)), drop=True, inplace=True)
    return pandout


def vpsoc3_reference(add_target=True, try_load=False,
                     noise=True):
    """Simulate agents with vpSOC(3) for combinations of r and h.

    The idea is to show that in vpSOC(3), for any combination of r and h, as
    hC1 approaches zero from the left, the maximum curvature around the second
    target increases.

    Returns
    -------
    data : pd.DataFrame
    Simulations in a happy panda. The columns are as in the experimental data
    (see imda.import_data()). 'ctarget' is also added, and 'part' reflects
    unique combinations of the parameters r and h. Additional columns r and h_x
    (where x is 0, 1, 2, 3) are added.

    """
    if try_load:
        try:
            data = pd.read_csv('./vpsoc_data.csv', index_col=0)
            return data
        except FileNotFoundError:
            logging.info('vpSOC(3) sims not found. Recalculating.')
    mod_pars = pc.vpsoc3_pars
    all_r = mod_pars['r']
    all_h = mod_pars['h']
    total = len(all_r) * len(all_h)
    data_list = []
    h_labels = [f'par_h_{idx}' for idx in range(len(all_h[0]))]
    generator = enumerate(product(all_r, all_h))
    if noise:
        dyn_kwargs = {}
        num_runs = 20
    else:
        dyn_kwargs = {'motor_noise': False, 'obs_noise': False,
                      'add_noise': False}
        num_runs = 1
    for part, (r, h) in tqdm(generator, total=total):
        agent = hie.vpSOC(t_ends=1.5, h_scaling=h,
                          soc_pars={'r': r, 'add_noise': 10000})
        datum, _ = all_seqs(agent=agent, num_runs=num_runs, t_end=1.5,
                            dyn_kwargs=dyn_kwargs)
        datum.loc[:, 'par_r'] = r
        datum.loc[:, h_labels] = h
        datum.loc[:, 'part'] = part
        datum.loc[:, 'noise'] = True
        data_list.append(datum)
    data = pd.concat(data_list, ignore_index=True)
    imda.label_current_target(data)
    imda.unique_run_identifier(data, append=True)
    imda.detect_failed_trials(data)
    if add_target:
        imda.label_current_target(data)
    return data


def pars_hiseq_reference_hc():
    """Return the pars for hiseq_reference."""
    h_second = 10
    all_h = [[1, h_second], [2, h_second], [10, h_second],
             [30, h_second]]
    all_r = [1e-7, 1e-6, 1e-5, 3e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    all_mondis = [0.01, 0.05, 0.1, 0.4, 0.5, 0.7, 1, 1.5]
    total = len(all_h) * len(all_r) * len(all_mondis)
    parprod = enumerate(product(all_h, all_r, all_mondis))
    return parprod, total


def hiseq_reference(add_target=True, try_load=False,
                    noise=True):
    """FFS."""
    if try_load:
        try:
            data = pd.read_csv('./hiseq_data.csv', index_col=0)
            return data
        except FileNotFoundError:
            logging.info('Hiseq sims not found. Recalculating.')
    dt = 0.01
    t_end = 1.7
    add_noise = 5000
    all_data = []
    parprod = pc.hiseq_parprod
    total = pc.hiseq_total
    if noise:
        dyn_kwargs = {}
        num_runs = 20
    else:
        dyn_kwargs = {'motor_noise': False, 'obs_noise': False,
                      'add_noise': False}
        num_runs = 1
    for idx, (h, r, mondis) in tqdm(parprod, total=total):
        mondis_abs = False
        soc_pars = {'sigma_c': 0.3, 'add_noise': add_noise, 'r': r, 'omega_nu': 0.6,
                    'omega_f': 0.6}
        masiso = hie.HiSeq(t_ends=t_end, h_scaling=h,
                           soc_pars=soc_pars,
                           dt=dt, mondis_abs=mondis_abs)
        masiso.delta_t_monitor = 0.0
        masiso.monitor_distance = mondis
        masiso.monitor_sd = 3
        masiso.tau23 = 5
        pandata, _ = all_seqs(masiso, num_runs=num_runs, t_end=t_end * 2,
                              background_noise=0, dyn_kwargs=dyn_kwargs)
        pandata['par_r'] = r
        pandata['par_h'] = h[1][0]
        pandata['par_mondis'] = mondis
        pandata['part'] = idx
        pandata['noise'] = noise
        all_data.append(pandata)
    data = pd.concat(all_data, ignore_index=True)
    imda.unique_run_identifier(data, append=True)
    imda.detect_failed_trials(data)
    targets_from_idmove(data)
    if add_target:
        imda.label_current_target(data)
    return data


def calc_and_save_hiseq(filename, **kwargs):
    """Run hiseq_reference and save nicely."""
    data = hiseq_reference(**kwargs)
    data.to_csv(filename)
    return data


def hiseq_allsoc():
    """Simulate a hiseq agent with only soc segments."""
    tau23 = 4
    monitor_sd = 1
    monitor_distance = 0.5
    mondis_abs = False
    transitions = [[0, 0], [0, 1], [1, 3], [3, 6]]
    sequences = [[0, 1, 2, 3]]
    t_ends = np.ones(len(transitions))
    # One value per target in sequences (pt_start, pt_xx, st_xx, pt_start)
    h_scaling = [1] * 4
    agent = hie.HiSeq(transitions=transitions, sequences=sequences,
                      t_ends=t_ends, h_scaling=h_scaling, tau23=tau23,
                      monitor_sd=monitor_sd, monitor_distance=monitor_distance,
                      mondis_abs=mondis_abs)
    data = multiple_runs_monitor(agent=agent, t_end=2.5, ix_seq=0, num_runs=20)
    imda.label_current_target(data)
    return data


def targets_from_idmove(data, inplace=True):
    """Rewrite the target columns in --data-- using id_move.

    Rewrite the "first", "second", etc. columns with values based on their
    id_move, using the experimental setup. For example, id_move of 0 has
    targets [pt_start, pt_left, st_left, pt_start].

    """
    if not inplace:
        data = data.copy()

    # Like it's done in experiment/parameters.py:
    *_, seq_labels = utils.get_labels()

    # HiSeq sims might lack a 'fourth' column:
    if 'fourth' not in data.columns:
        data['fourth'] = 'dummy'
    ordinals = ['first', 'second', 'third', 'fourth']
    for id_move, datum in data.groupby('id_move'):
        targets = seq_labels[id_move] + ('pt_start', )
        data.loc[datum.index, ordinals] = targets

    if not inplace:
        return data


def merge_simulations(sims):
    """Merge, sort and re-label simulations."""
    data = pd.concat(sims, ignore_index=True)
    par_labels = [col for col in data.columns if col.startswith('par_')]
    for part, (dummy, datum) in enumerate(data.groupby(par_labels)):
        data.loc[datum.index, ['part']] = part
    return data
