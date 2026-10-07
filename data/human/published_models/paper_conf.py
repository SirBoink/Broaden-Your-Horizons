"""Configuration parameters for the manuscript.

The first section comprises general parameters for fonts and colors, followed
by the sign-flipping rules for halfway and transitional coarticulations.

The second section are the figue-specific parameters for simulations with vpSOC
and HiSeq. Here we list only those parameters that are crucial for the
figures. All others are left as the default parameters that can be found in
each model:
1. The base class for SOC can be found in models/single.py: SOC. Its default
parameters are:
def_pars = {'sigma_c': 0.3,
            'p_scaling': 1,
            'm': 1.0,
            'omega_nu': 0.2,  # Adapted for longer movements
            'omega_f': 0.3,  # Adapted for longer movements
            'N': 5,
            'M': 3,
            'Nc': 1,
            'tau1': 0.04,
            'tau2': 0.04,
            'delta': 0.01,
            'sigma_s': 0.5,
            'r': 1e-5,  # Weight of the energy term for cost
            'obs_noises': [0.02, 0.02, 1],
            'dt': 0.01,
            't_end': 1,
            'add_noise': 1000,  # Additive noise for dynamics
            }
2. The class for vpSOC can be found in models/hierarchical.py:vpSOC. Besides
parameters pertaining to the experimental task (targets, sequences, etc.), as
well as those from SOC (using the same default values), it introduces
h_scaling. h_scaling and r are systematically changed for simulations and
defined differently for each figure.

3. The HiSeq class can be found in models/hierarchical.py:HiSeq. It uses the
parameters from SOC and vpSOC, with the same defaults. In addition, it
introduces a number of parameters for the dynamical system (Lotka-Volterra
equations) which are left unchanged throughout simulations. It also introduces
the monitor_distance parameter, which determines how close to the target (the
center of the target circle) a trajectory has to get before the next element of
the sequence is triggered. Related, the monitor_sd parameter controls the speed
with which new elements are activated. These are varied throughout simulations.

"""
from itertools import product

import numpy as np

from src.model import hierarchical as hie

####################################################
# ------- General configuration for plots    -------
####################################################


# Data stuff
all_parts = np.arange(401, 421)
hiseq_parts = np.array([402, 403, 404, 418, 413])


# Plotting stuff
subplot_label_font = {'size': 12}

colors_seq = np.array([[150, 150, 255], [150, 0, 255], [0, 150, 255],
                       [255, 150, 150], [255, 150, 0], [255, 0, 150]]) / 256
colors_sin = np.array([[0, 0, 255], [255, 0, 0]]) / 256


# Analysis parameters

# Depending on the function (trajectory_variance vs trajvar_all_segments),
# different rules apply.
# Flipping sign for trajectory_variance
flip_rule_trans = {(0, 'sequential'): -1,
                   (3, 'sequential'): -1,
                   (5, 'sequential'): -1}
flip_rule_half = {(1, 'sequential'): -1,
                  (2, 'sequential'): -1,
                  (4, 'sequential'): -1}  # Flips crosses for readability

# Flipping signs for trajvar_all_segments
flip_rule_all = {(0, 'sequential', 0): -1,
                 (4, 'sequential', 0): -1,
                 (0, 'sequential', 1): -1,
                 (1, 'sequential', 1): -1,
                 (2, 'sequential', 1): -1,
                 (3, 'sequential', 1): -1,
                 (4, 'sequential', 1): -1,
                 (5, 'sequential', 1): -1,
                 (5, 'sequential', 2): -1}

# Models parameters
ref_vpsoc = {'soc_pars': {'add_noise': 10_000},
             't_end': 0.7,
             'h_scaling': 1
             }
ref_sims = {}


####################################################
# ------- For Figures 2 and 3, with data     -------
####################################################
good_part = 404  # Participant A
bad_part = 406  # Participant B


####################################################
# --------- For Figure 4 and 5, classification -----
####################################################

# Selected participants for the classification of trajectories:
parts_hiseq_class = [412, 414]


# Model parameters for vpSOC(3)
_r_values = [1e-8, 5e-8, 1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 8e-7, 1e-6,
             1.2e-6, 2e-6, 3e-6, 5e-6, 8e-6, 1e-5, 5e-5]
# _r_values = [1e-8, 1e-7]
_h1 = [0.6, 1, 1.3, 1.7, 2, 18]
_h2 = [1, 2, 18]
_h3 = [5]
vpsoc3_pars = {'r': _r_values,
               'h': list(product(_h1, _h2, _h3))}


# Model parameters for HiSeq simulations
dd = 10  # Dummy value; unused by models but may be expected
h_sec = 10
hs = [1.1, 1.2, 1.5, 1.8]
all_h = [np.array([[dd, dd], *[[h_prot, 20, 0]] * 6, *[[h_sec, dd]] * 3],
                  dtype=object)
         for h_prot in hs]
all_r = [1.0e-07, 5.0e-07, 7.0e-07, 9.0e-07, 1.0e-06, 1.2e-06, 1.5e-06,
         1.8e-06, 2.0e-06, 2.2e-06, 2.5e-06, 3.0e-06, 4.0e-06, 5.0e-06,
         6.0e-06, 7.0e-06, 8.0e-06, 1.0e-05, 2.0e-05, 3.0e-05, 5.0e-05,
         1.0e-04, 5.0e-04, 1.0e-03]
all_r = all_r[-2:]
all_mondis = [0.1, 0.4, 0.5, 0.7, 1, 1.5, 2]
# all_mondis = [0.05, 0.5, 2]
hiseq_total = len(all_h) * len(all_r) * len(all_mondis)
hiseq_parprod = enumerate(product(all_h, all_r, all_mondis))


# # Other (fixed) Model parameters for HiSeq simulations
# tau23 = 3
# monitor_sd = 10
# monitor_distance = 1.2
# mondis_abs = False
# transitions = [[0, 0], [0, 1], [1, 3], [3, 6],  # all SOC
#                [0, 1, 3],   # vpSOC(2) + SOC
#                [1, 3, 6]]  # SOC + vpSOC(2)

# # All seqs. are idmove 0, with three "models", repeated to make 6:
# sequences = np.array([[0, 1, 2, 3], [0, 4, 3], [0, 1, 5]] * 2, dtype=object)
# t_ends = 1.7  # [1, 1, 1, 1, 2, 2]
# r = [1e-4]
# soc_pars = {'r': r, 'add_noise': 10}
# h_scalings = []
# for ch1 in [10, 30, 50]:
#     h_scalings.append([[1], [1], [1], [1], [ch1, 1], [ch1, _h3[0]]])
# hiseq_pars = {'tau23': tau23, 'monitor_sd': monitor_sd,
#               'monitor_distance': monitor_distance, 'transitions': transitions,
#               'sequences': sequences, 't_ends': t_ends, 'r': r,
#               'h': h_scalings, 'soc_pars': soc_pars,
#               'mondis_abs': mondis_abs}


####################################################
####################################################
############# Supplementary figures ################
####################################################
####################################################

# Note that some of these are no longer used.

####################################################
# --------- For Supp. Figure, cost of planning------
####################################################

cost_num_targets = np.arange(1, 5)
cost_durations = np.arange(0.5, 20, 0.5)
cost_reps = 1000


####################################################
# ------- For Figure 5, with vpSOC simulations -----
####################################################
r_low_high = [4e-6, 8e-6]

# For vpSOC(3):
hs_first = np.array([0.9, 10])
hs_second = np.array([1.5, 18])
hs_third = np.array([15])
hs_vpsoc3 = list(product(hs_first, hs_second, hs_third))

# For vpSOC(2):
hs_first = np.array([0.4, 3])
hs_second = np.array([3])
hs_vpsoc2 = list(product(hs_first, hs_second))


####################################################
# ------- For Figure 8, with data and models -------
####################################################

# Models and parameters
hiseq_ts = np.array([1, 2, 1]) * 1
tau23 = 3
monitor_sd = 5
monitor_distance = 30
r_high = 2e-5
r_low = 2e-7
mods = {'vpSOC(3)+': (hie.vpSOC,
                      [{'h_scaling': [15, 15, 6],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 0,
                                     'r': r_high}},
                       {'h_scaling': [6, 10, 15],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 0,
                                     'r': r_high}},
                       {'h_scaling': [2, 8, 15],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 0,
                                     'r': r_high}},
                       ]
                      ),
        'vpSOC(3)-': [hie.vpSOC,
                      [{'h_scaling': [1, 7, 15],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 8_000,
                                     'r': r_low}
                        },
                       {'h_scaling': [1, 7, 15],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 0,
                                     'r': r_low}
                        },
                       {'h_scaling': [1, 7, 15],
                        't_ends': 1.6,
                        'soc_pars': {'add_noise': 0,
                                     'r': r_low}
                        },
                       ]
                      ],
        'HiSeq (vpSOC(2)+vpSOC(1))': [hie.HiSeq,
                                      [{'transitions': [[0, 0], [0, 1, 3], [3, 6]],
                                        'sequences': [[0, 1, 2]] * 6,
                                        't_ends': hiseq_ts,  # TODO: fix this
                                        'h_scaling': 5,
                                        'tau23': tau23,
                                        'monitor_sd': monitor_sd,
                                        'monitor_distance': monitor_distance,
                                        'soc_pars': {'add_noise': 90,
                                                     'r': 5e-5}},
                                       {'transitions': [[0, 0], [0, 1, 4], [4, 6]],
                                        'sequences': [[0, 1, 2]] * 6,
                                        't_ends': hiseq_ts,  # TODO: fix this
                                        'h_scaling': np.array([[1, 1], [15, 3],
                                                               [2, 1]]),
                                        'tau23': tau23,
                                        'monitor_sd': monitor_sd,
                                        'monitor_distance': monitor_distance,
                                        'soc_pars': {'add_noise': 90,
                                                     'r': 5e-5}},
                                       {'transitions': [[0, 0], [0, 2, 3], [3, 6]],
                                        'sequences': [[0, 1, 2]] * 6,
                                        't_ends': hiseq_ts,  # TODO: fix this
                                        'h_scaling': np.array([10, 5]),
                                        'tau23': tau23,
                                        'monitor_sd': monitor_sd,
                                        'monitor_distance': monitor_distance,
                                        'soc_pars': {'add_noise': 90,
                                                     'r': 5e-5}},
                                       ]
                                      ]
        }
# Selected participant data (part, id_move)
example_trajs = {'vpSOC(3)+': [(406, 0),
                               (414, 2),
                               (420, 3)],
                 'vpSOC(3)-': [(413, 0),
                               (413, 2),
                               (413, 3)],
                 'HiSeq (vpSOC(2)+vpSOC(1))': [(418, 0),
                                               (403, 2),
                                               (412, 3)]
                 }
