"""Various hierarchical models for hand movements."""

from itertools import product

import numpy as np
from scipy.integrate import solve_ivp

from src import utils
from src.model import single


def get_crho(num_neurons, seq_length, clusters):
    """Set the values for the interneuron connections.

    The sequences to be generated will comprise the clusters in the order
    given by --clusters--, cut into sequences of length --seq_length--. For
    example, if --seq_length-- is 3, then the first sequence is going to be
    the first three elements of --clusters--, the second sequence from the
    fourth to the sixth (inclusive), etc.

    Parameters
    ----------
    num_neurons : int
    Number of neurons in the population

    seq_length : int
    Number of clusters per sequence

    clusters : ndarray
    Matrix containing all the clusters in the sequences. Each row is taken to
    be a the collection of neurons belonging to each cluster.
    """
    num_clusters = clusters.shape[0]

    rho = 2 * np.ones((num_neurons, num_neurons))  # High all-to-all inhibition

    for ix_cluster in np.arange(num_clusters):
        frac = 1 / len(clusters[ix_cluster, clusters[ix_cluster, :] != 0])
        in_pc = 2 * frac  # inhibition to previous cluster
        in_cc = frac  # inhibition within current cluster
        in_nc = frac / 10  # inhibition to next cluster

        for ix_neuron in np.arange(len(clusters[ix_cluster,
                                                clusters[ix_cluster] != 0])):
            if ix_cluster % seq_length > 0:
                rho[clusters[ix_cluster - 1, clusters[ix_cluster - 1, :] != 0],
                    clusters[ix_cluster, ix_neuron]] = in_pc
            rho[clusters[ix_cluster, clusters[ix_cluster, :] != 0],
                clusters[ix_cluster, ix_neuron]] = in_cc
            if (ix_cluster + 1) % seq_length > 0:
                rho[clusters[ix_cluster + 1, clusters[ix_cluster + 1, :] != 0],
                    clusters[ix_cluster, ix_neuron]] = in_nc
    return rho


def generate_clusters(N, C, M):
    """Generate the clusters for the cshs.

    Each cluster contains --clus_size-- neurons, and they're set so that each
    cluster will excite another cluster, thus creating a sequence. A
    proto-connectivity matrix is created, which, for each element, contains a
    list of the elements it excites.

    Parameters
    ----------
    N : int
    Number of neurons in space

    C : int
    Number of desired clusters

    M : int
    Number of elements per cluster

    """
    clusters = - np.ones((C, M), dtype=np.int64)
    pool = np.random.permutation(N)
    already_seen = np.zeros(N, dtype=np.int64)

    clusters[0, :] = pool[:M]
    already_seen[clusters[0, :]] += 1
    for ix_c in np.arange(1, C):
        sorted_index = np.argsort(already_seen)
        sorted_as = already_seen[sorted_index]
        partition = - np.ones(N, dtype=np.int64)
        partition[0] = 0
        c_par = 1
        for ix_par in np.arange(N - 1):
            if sorted_as[ix_par] != sorted_as[ix_par + 1]:
                partition[c_par] = ix_par + 1
                c_par += 1
        partition = partition[partition != -1]
        for ix_part, part in enumerate(partition[:-1]):
            aux_shuffle = np.random.permutation(partition[ix_part + 1] - part)
            aux_sorted = sorted_index[part:partition[ix_part + 1]]
            sorted_index[part:partition[ix_part + 1]] = aux_sorted[aux_shuffle]
        clusters[ix_c, :] = sorted_index[:M]
        flag1 = 1
        flag2 = 0
        add_k = 1
        change_n = 1
        while flag1:
            for ix_comp in np.arange(ix_c):
                if np.all(sorted(clusters[ix_c, :]) == sorted(clusters[ix_comp,
                                                                       :])):
                    flag2 = 1
                    break
            if flag2:
                clusters[ix_c, change_n] = sorted_index[M + add_k]
                add_k += 1
                if M + add_k == len(sorted_index):
                    if change_n == N:
                        RuntimeError('Too much cluster repetition led to bad'
                                     ' results. Try again.')
                    change_n += 1
                    add_k = 1
                flag2 = 0
            else:
                flag1 = 0
        already_seen[clusters[ix_c, :]] += 1
    return clusters


def clusters_two_pops(num_neurons, num_clusters, size_clusters, loc='end',
                      new_neurons=None):
    """Generate the clusters from two populations of neurons.

    One of them, of size --num_neurons--, makes the bulk of the clusters. A
    second population is created, of size --num_clusters--, from which one
    neuron belongs to each of the clusters.

    The second population is numbered starting from --num_neurons--, and in
    total, the two populations have --num_neurons-- + --num_clusters-- neurons.
    This can be inverted with --loc--.

    Parameters
    ----------
    num_neurons : int
    Size of the neuronal population from which the clusters is created.

    num_clusters : int
    Number of clusters to create. This is overriden if --new_neurons-- is
    provided.

    size_clusters : int
    Number of neurons per cluster. Note that an extra neuron is added to
    each cluster.

    new_neurons : ndarray
    Array of the new neurons to use. A cluster will be created for each of
    these neurons, and the neuron placed in the cluster. If not None, the
    number of clusters is set to len(new_neurons), regardless of
    --num_clusters--. If providing --new_neurons-- and --loc--=='end', it is
    the user's responsability to make sure that the neurons in --new_neurons--
    do not include any numbers between 0 and num_neurons, inclusive; otherwise,
    the clusters will have repeated neurons and it all goes to hell fast.

    loc : str
    If 'end', the second population is at the end of the first. If 'start',
    it's at the beginning. This refers to indices, not to the position in
    clusters.

    """
    if new_neurons is None:
        if loc == 'end':
            new_pop = np.arange(num_neurons, num_clusters + num_neurons)[:,
                                                                         None]
        elif loc == 'start':
            new_pop = np.arange(num_clusters)[:, None]
        else:
            raise ValueError('--loc-- should be either "start" or "end", '
                             'was {}'.format(loc))
        offset = num_clusters
    else:
        new_pop = new_neurons[:, None]
        offset = max(new_neurons)
        num_clusters = len(new_neurons)
        print('Ignored --num_neurons-- in favor of --new_neurons--')
    clusters = generate_clusters(num_neurons, num_clusters, size_clusters)
    if loc == 'end':
        clusters = clusters + offset
    clusters = np.concatenate([clusters, new_pop], axis=1)
    return clusters


class Simple():
    """Simplified version of the five-level monstrosity.

    Model which only has three levels, instead of five; the model consists
    of the physical level, an SHS level and the Goal level.

    The positions of the attractors are set as in the experiment. The physical
    layer is set to the global attractor, as an example.

    """

    def_pars = {'tau': 1,
                'tau23': 1.1,
                'tau1': 1,
                'monitor_sd': 1,
                'monitor_distance': 200,
                'delta_t_mon itor': 0.1,
                'dt': 0.01,
                }
    name = 'Simple'

    def __init__(self, sigma=None, random_seed=None, sequences=None,
                 transitions=None, mondis_abs=True, **pars):
        """Initialize all matrices, sequences, goals, snacks."""
        self.mondis_abs = mondis_abs
        self.def_pars.update(pars)
        for key, value in self.def_pars.items():
            setattr(self, key, value)
        self.random = np.random.default_rng(random_seed)
        self.offset_origin = 0  # Legacy; dummy value. Delete later
        if transitions is None:
            transitions = np.array([[0, 0], [0, 1], [0, 2], [1, 3], [1, 4],
                                    [1, 5], [2, 3], [2, 4], [2, 5],
                                    [3, 6], [4, 6], [5, 6]])
        self.transitions = transitions
        if sequences is None:
            sequences = np.array([[0, 1, 3, 9], [0, 1, 5, 10], [0, 1, 4, 11],
                                  [0, 2, 6, 9], [0, 2, 8, 10], [0, 2, 7, 11]])
        self.sequences = sequences
        self.sequence = sequences[0]
        self.num_seqs = len(sequences)
        self.num_moves = len(transitions)
        self.phys_dim = 2
        self.Ns = self.num_moves
        self.N = self.Nx = self.num_seqs + self.Ns + 2
        rhos = np.zeros((len(sequences), self.Ns, self.Ns))
        for ix_seq, sequence in enumerate(sequences):
            sigma = self._set_sigmas(sigma, sequence, set_sigma=False,
                                     ret_sigma=True)
            c_rho = self._set_rhos(sequence, sigma, ret_rho=True,
                                   set_rho=False)
            rhos[ix_seq, ...] = c_rho
        self.sigma = sigma
        self.rhos = rhos
        ix_seq = 0
        self.rho = rhos[ix_seq, ...]  # Just in case
        basket = utils.coords_from_svg(reference='mk_tr')
        self.basket = basket
        circles = ['pt_start', 'pt_left', 'pt_right', 'st_left', 'st_top',
                   'st_right', 'pt_start']
        self.labels = circles
        self.centers = np.array([self.transform_circles(*basket[key].center)
                                 for key in circles])
        self.radii = np.array([basket[key].radius for key in circles])
        # transitions = np.array([[0, 0], [0, 1], [0, 2], [1, 3], [1, 4], [1, 5],
        #                         [2, 3], [2, 4], [2, 5]])
        self.moves = self.fixed_models()

        self._map_neurons_to_space()

    def transform_circles(self, x, y, **params):
        """Perform coordinate transformation.

        Transforms the coordinates from the (x,y) system in the experimental
        design to some other coordinate system.

        This is a placeholder meant to be overwritten in subclasses.

        """
        return x, y

    def _set_sigmas(self, sigma=None, sequence=None, ret_sigma=False,
                    set_sigma=True):
        r"""Set random values for sigma.

        Ensures that the sigma[i_0] >= sigma[j] for all j. If --sigma-- is
        None, random values are generated.

        Sets
        ----
        sigma : 1darray size == [N,]
        Random values of the sigma parameters.

        """
        if sequence is None:
            sequence = self.sequence
        if sigma is None:
            sigma = 10 + 5 * self.random.random(self.Ns)
        old_zero = sigma[sequence[0]]
        ix_max = sigma.argmax()
        sigma[sequence[0]] = sigma[ix_max]
        sigma[ix_max] = old_zero
        if set_sigma:
            self.sigma = sigma
        if ret_sigma:
            return sigma

    def _set_rhos(self, sequence=None, sigma=None, ret_rho=False,
                  set_rho=True):
        """Set the connectivity matrix for the N neurons.

        Parameters
        ----------
        sigma : ndarray size==[N, ]
        ODE parameters

        sequence : ndarray size <= N
        Desired sequence to be written into the ODE. Each neuron can be
        included at most once.

        Returns ---- rho : ndarray size == [N, N] Connection strength between
        all neurons. If --set_rho-- is True, self.rho is set.

        """
        if sigma is None:
            sigma = self.sigma
        if sequence is None:
            sequence = self.sequence
        rho = 2 * np.ones([self.Ns, self.Ns], dtype=float)

        # Condition 41 AZR
        for p_seq, c_seq in zip(sequence, sequence[1:]):
            rho[p_seq, c_seq] = sigma[p_seq] / sigma[c_seq] + 0.51

        np.fill_diagonal(rho, 1)

        # Condition 42 AZR
        for c_seq, n_seq in zip(sequence, sequence[1:]):
            rho[n_seq, c_seq] = sigma[n_seq] / sigma[c_seq] - 0.5

        for ixx in range(self.Ns):
            for ixy in range(1, len(sequence)):
                cond_1 = ixx != sequence[ixy - 1]
                cond_2 = ixy == len(sequence) - 1
                if cond_2:
                    cond_3 = True  # Dummy value
                else:
                    cond_3 = ixx != sequence[ixy + 1]
                if cond_1 and (cond_2 or cond_3):
                    sum_1 = rho[sequence[ixy - 1], sequence[ixy]]
                    sum_2 = (sigma[ixx] - sigma[sequence[ixy - 1]])
                    div = sigma[sequence[ixy]]
                    rho[ixx, sequence[ixy]] = sum_1 + sum_2 / div + 2
        np.fill_diagonal(rho, 1)
        if set_rho:
            self.rho = rho
        if ret_rho:
            return rho

    def set_initial_conditions(self, sequence=None, background_noise=0.0):
        """Set initial conditions in the right shape."""
        if sequence is None:
            sequence = self.sequence
        x_init = background_noise * np.ones(self.N)
        x_init[sequence[0]] = 0.95 * self.sigma[sequence[0]]
        x_init[sequence[1]] = 0.05 * self.sigma[sequence[1]]

        x_ini = np.concatenate([[0], [0], x_init[:self.Ns]])
        return x_ini

    def monitor(self, x_t):
        """Monitor for subgoals. Similar to hierarchical_agent.monitor."""
        c_goal = self. transitions[np.argmax(x_t[..., self.phys_dim:])][-1]
        if c_goal == 0:
            return False
        attractor = self.centers[c_goal]
        distance = np.linalg.norm(attractor - x_t[..., :self.phys_dim],
                                  axis=-1)[..., None]
        if self.mondis_abs:
            if np.ndim(self.monitor_distance) > 0:
                reference = self.monitor_distance[c_goal]
            else:
                reference = self.monitor_distance
            # reference = self.monitor_distance
        else:
            reference = self.monitor_distance * self.radii[c_goal]
            # ipdb.set_trace(cond=c_goal==3 and reference > distance)
        flag = distance < reference
        return flag

    def _map_neurons_to_space(self, ):
        """Create mapping between physical space and SHS.

        Set the mapping between the physical space of movements and the
        neurons on the shs level of the model, assuming that the number
        of neurons is a square number.

        It assumes the physical space is a square (or cube) of size 1x1 and
        distributes the neurons of the shs level evenly, with the first
        and last neurons at the edges.

        Results are saved to self.mapping

        """
        # if np.sqrt(self.Ns) % 1 != 0:
        #     raise ValueError('The number of neurons --num_neurons-- '
        #                      'must be the square of an integer')
        side = np.sqrt(self.Ns).astype(int)
        places = np.linspace(0, 1, side, endpoint=True)
        mapping = np.array(list(product(places, places)))
        self.mapping = mapping[self.random.permutation(mapping.shape[0]), :]

    def fixed_models(self, ):
        """Create all the attractor models for the physical layer.

        One model
        for every combination of start-pt, and pt-st. Uses the coordinates
        from the svg file.

        Returns all these models. The order of the models returned is
        consistent with the way the sequences of clusters are created in the
        class; using the svg's nomenclature, the order of the models is:
        1. pt_start -> pt_blue -> st_olive
        2. pt_start -> pt_blue -> st_lila
        ...
        8. pt_start -> pt_red -> st_pink

        """
        transitions = self.transitions
        centers = self.centers
        moves = [[]] * self.num_moves
        for ix_model in range(self.num_moves):
            c_trans = transitions[ix_model]
            pos_ini, pos_end = centers[c_trans]
            if self.offset_origin:
                pos_ini -= (pos_end - pos_ini) * self.offset_origin
            moves[ix_model] = self._build_attractor(pos_ini, pos_end)
        return moves

    def choose_seq(self, ix_seq, set_init_cond=True):
        """Set the current sequence to be executed.

        Set self.rho to be that consistent with the chosen sequence
        as per self.sequences[ix_seq].

        Additionally, if set_init_cond is True, the initial conditions
        are set as self.x_t to facilitate the chosen sequence.

        If ix_seq is an array of size=(self.sequences.shape[0], ),
        it is normalized and taken as a distribution over goals. It is
        then used to create a weighted average of the rhos.

        """
        if isinstance(ix_seq, np.ndarray):
            if ix_seq.shape[-1] != self.sequences.shape[0]:
                raise ValueError('ix_seq is of bad size')
            self.rho = np.sum(self.rhos.T.dot(ix_seq.T).T, axis=0) / \
                ix_seq.sum()  # In case ix_seq isn't normalized
        else:
            self.rho = self.rhos[ix_seq, ...]
        if set_init_cond:
            self.x_t = self.set_initial_conditions(
                sequence=self.sequences[ix_seq])

    def _build_attractor(self, pos_ini, pos_end):
        """Define point attractor.

        Define the attractor using the simple point attractor for direction
        and a nice field for speed.

        """
        def mafu(t, x):
            return pos_end - x

        return mafu

    def x_dot(self, t, x_t, pars=None):
        """Return the RHS of the differential equation.

        Replaces the parent's x_dot method to account for this class's way
        of doing the attractors
        """
        # y = x_t[..., :self.phys_dim]
        x = x_t[..., self.phys_dim:]
        y_dot = self._x1_dot(t, x_t) * self.tau1
        x_dot = self.tau23 * x * (self.sigma - self.rho.dot(x.T).T)
        return np.concatenate([y_dot, x_dot], axis=-1)

    def move(self, x_t, t_ini, t_end, x_noise=0, id_stop=None):
        """Run a simulation of the system with the given parameters.

        Divides integration into intervals of --monitor_delta_t-- length,
        after each of which the monitor is called to see if the current
        element of the sequence has been reached.

        Parameters
        ----------
        x_t : ndarray

        t_ini, t_end : float
        Initial and final times for integration.

        x_noise : ndarray
        Noise to add to the dynamics. self.x_dot decides how to do this.

        id_stop : str
        If provided, the simulation will end as soon as the circle indexed by
        id_stop has been reached. For this, self.Basket is used.

        Returns
        -------
        all_x : ndarray
        Values for the vector x_t for each trial +1. The first element (in the
        leading dimension) is the parameter --x_t--

        all_t : ndarray
        All times

        """
        if t_end - t_ini == 0:
            return x_t[None, ...], np.array([t_ini, t_end])
        noise = self.monitor_sd * self.monitor(x_t)

        def x_dot_noise(t, x):
            return self.x_dot(t, x) + noise
        t_eval = np.arange(t_ini, t_end, self.dt) + self.dt
        intout = solve_ivp(x_dot_noise, (t_ini, t_end + self.dt), x_t,
                           t_eval=t_eval, method='RK45')
        return (np.concatenate([x_t[None, :], intout.y.T]),
                np.concatenate([[t_ini], intout.t]))

    def _x1_dot(self, t=None, x_t=None):
        """Attractor-level dynamical system."""
        if x_t is None:
            x_t = self.x_t
        x_two = x_t[2:(self.num_moves + 2)]
        weights = x_two / self.sigma
        x_one = x_t[:2]
        out = np.sum([move(t, x_one) * weights[idx]
                      for idx, move in enumerate(self.moves)], axis=0)
        return out


class SimpleCubic(Simple):
    """Version of the Simple model with the Cubic system as physical space."""

    name = 'Simple Cubic'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau1 = 3 / 10 ** 3
        self.tau23 = 1

    def _build_attractor(self, pos_ini, pos_end):
        """Cubic!"""
        diff = pos_end - pos_ini

        def mafu(x):
            xdot = np.zeros_like(x)
            for idx, diffy in enumerate(diff):
                if diffy:
                    xdot[idx] = (x[idx] - pos_ini[idx]) ** 2 * (
                        pos_end[idx] - x[idx])
                else:
                    xdot[idx] = (pos_ini - x)[idx]
            return xdot

        return mafu


class SimpleCubicScaled(Simple):
    """Version of the Simple model with the Cubic system for the physical space"""

    name = 'Simple Cubic Scaled'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau1 = 1 / 10 ** 1
        self.tau23 = 1

    def _build_attractor(self, pos_ini, pos_end):
        """Cubic!"""
        diff = pos_end - pos_ini

        def mafu(x):
            xdot = np.zeros_like(x)
            for idx, diffy in enumerate(diff):
                if diffy:
                    xdot[idx] = (x[idx] - pos_ini[idx]) ** 2 * (
                        pos_end[idx] - x[idx]) / np.abs(diff[idx])
                else:
                    xdot[idx] = (pos_ini - x)[idx]
            return xdot

        return mafu


class SimpleTodorov(Simple):
    """Version of the Simple model that contains the linear model for limb
    movements from the Todorov paper (see movemod.LinearTodorov for more info
    on this).

    """
    name = 'Simple Todorov'

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.arm = single.LinTodSimple()

    def move(self, x_t, t_ini, t_end, x_noise=0):
        """Overrides the parent's move method to include the linear model."""
        num_future_steps = 50
        if t_end - t_ini == 0:
            return x_t[None, ...], np.array([t_ini, t_end])
        if np.ndim(x_noise) == 0:
            x_noise = x_noise * np.ones_like(x_t)
        t_vec = np.arange(t_ini, t_end, self.dt)
        t_vec = np.concatenate([t_vec, [t_end]])
        all_x = [x_t, ]
        all_t = [t_ini]
        all_u = []
        c_devs = np.array([[0, 0, 0], [0, 0, 0]])
        x5_t = np.hstack([x_t[:self.phys_dim][:, None], c_devs, [[1], [1]]])
        flaggy = -10
        c_t = t_ini
        noise_vec = np.ones_like(x_t)
        noise_vec[:self.phys_dim] = 0
        for ix_t, c_t in enumerate(t_vec):
            candidates = self._generate_next(x_t, c_t, 4)
            if ix_t < 4:
                c_last = candidates[ix_t, ...]
            else:
                c_last = candidates[-1, ...]
            c_u = self._generate_motor(c_last, x5_t)
            x5_t = self.arm.dynamics(x5_t, c_u, with_noise=True)
            oth_x, _ = super().move(x_t, c_t, c_t + self.dt)
            x_t = oth_x[-1, ...]
            x_t[:self.phys_dim] = x5_t[..., 0]
            all_x.append(x_t)
            all_t.append(c_t)
            all_u.append(c_u)
        all_u.append(0)
        return np.array(all_x), np.array(all_t), np.array(all_u)

    def _calculate_devs(self, all_x):
        """Generates the derivatives at time t. If not enough time points to
        calculate the nth derivative, it will be set to zero."""
        x = np.array(all_x)
        if x.shape[0] == 0:
            return np.zeros((self.phys_dim, 3), dtype=float)
        missing_time = 4 - x.shape[0]
        if missing_time > 0:
            fill = np.tile(x[0, ...], (missing_time, 1))
            x = np.vstack([x, fill])
        vel = np.diff(x, axis=0)[-1, :self.phys_dim]
        acc = np.diff(x, 2, axis=0)[-1, :self.phys_dim]
        con = np.diff(x, 3, axis=0)[-1, :self.phys_dim]
        return np.vstack([vel, acc, con]).T

    def _generate_next(self, x_t, t_ini, predict_ahead):
        """Generates the next four steps using the attractor"""
        num_future_steps = predict_ahead
        t_vec = np.arange(num_future_steps) * self.dt + t_ini
        all_x = np.zeros((num_future_steps, x_t.shape[0]))
        for idx, t in enumerate(t_vec):
            x_t = x_t + self.x_dot(t, x_t) * self.dt
            all_x[idx, ...] = x_t
        return all_x

    def _generate_motor(self, last_x, x5_t):
        """Generates the motor commands to achieve the x_t values."""
        ut = np.zeros(self.phys_dim)
        for ix_dim in range(self.phys_dim):
            ut[ix_dim] = self.arm.inv_dynamics(last_x[ix_dim],
                                               x5_t[ix_dim, ...])
        return ut

    def x_dot(self, *args, **kwargs):
        xdot = super().x_dot(*args, **kwargs)
        # xdot[:self.phys_dim] = (-50, 50)
        return xdot


class SimpleSOC(Simple):
    """Implementation of the Simple model with SOC internal models."""

    name = 'Simple SOC'

    def __init__(self, t_ends=0.5, soc_pars=None, *args, **kwargs):
        """Initialize all matrices and internal models.

        SHS-based sequential movements with SOC controllers for
        single-target movements.

        Parameters
        ----------
        t_ends : ndarray, float
        Duration of each movement. If a float, the same duration is used
        for all movements. If an ndarray, its size should equal the
        number of movements.

        soc_pars : dict
        Sent to class single.SOC

        *args and **kwargs are sent to class Simple

        """
        if soc_pars is None:
            soc_pars = {}
        self.shoulder_anchor = np.array([-100, -450])  # Roughly shoulder
        self.a1 = 300
        self.a2 = 420
        super().__init__(*args, **kwargs)
        self.soc_pars = soc_pars
        if np.ndim(t_ends) == 0:
            t_ends = t_ends * np.ones(self.num_moves)
        self.t_ends = t_ends
        self.socs = self.build_socs()
        self.threshold = 0.2
        self.arm = single.SOC(dt=self.dt, **soc_pars)
        # self.arm.C = 5 * self.arm.C

    def angles_to_xy(self, a1, a2, *args, **kwargs):
        """Return inputs; for compatibility."""
        return a1, a2

    def build_socs(self):
        """Build a SOC model for each movement."""
        transitions = self.transitions
        centers = self.centers
        moves = [[]] * self.num_moves
        for ix_model in range(self.num_moves):
            t_end = self.t_ends[ix_model]
            c_trans = transitions[ix_model]
            pos_ini, pos_end = centers[c_trans]
            moves[ix_model] = self._build_soc(pos_ini, pos_end, t_end)
        return moves

    def _build_soc(self, pos_ini, pos_end, t_end):
        """Use the SOC model instead of attractors."""
        LNt = [[]] * self.phys_dim
        KNt = [[]] * self.phys_dim
        socky = single.SOC(t_end=t_end, dt=self.dt, **self.soc_pars)
        self.socky = socky
        for ix_dim in range(self.phys_dim):
            x5_ini = np.array((pos_ini[ix_dim], 0, 0, 0, pos_end[ix_dim]))
            LNt[ix_dim], KNt[ix_dim] = socky.optimal_controler(x5_ini)

        if (pos_ini == pos_end).all():
            KNt = [np.zeros(KNt[0].shape), np.zeros(KNt[1].shape)]
            LNt = [np.zeros(LNt[0].shape), np.zeros(LNt[1].shape)]

        def dynamic(t, x5N_t, y5N_t, x5Nhat_t, dyn_kwargs=None):
            """Return x5[t + 1] given x5_t.

            Parameters
            ----------
            x5N_t : iterable
            Contains the x5_t vectors, with as many elements as there are
            physical dimensions (2D for the surface on a table). Each element
            is an ndarray with 5 elements (x5_t; position, speed, etc).

            """
            if dyn_kwargs is None:
                dyn_kwargs = {}
            x5N_t[:, -1] = pos_end
            ct = int(t / self.dt)
            x5N_tp1 = []
            uN_tp1 = []
            for Kt, Lt, x5hat_t, y5_t in zip(KNt, LNt, x5Nhat_t, y5N_t):
                xhat_tp1, u_tp1 = socky.estimate_and_control(x5hat_t, y5_t,
                                                             Kt[ct], Lt[ct])
                x5N_tp1.append(xhat_tp1)
                uN_tp1.append(u_tp1)
            x5N_tp1 = []
            y5N_tp1 = []
            for u_t in uN_tp1:
                x_t, y_t = socky.dynamics(x5N_t, u_t, **dyn_kwargs)
                x5N_tp1.append(x_t)
                y5N_tp1.append(y_t)
            return (np.array(x5N_tp1), np.array(uN_tp1), np.array(x5N_tp1),
                    np.array(y5N_tp1))

        return dynamic

    def mix_socs(self, x5_t, x_t, t):
        """Mix SOCs for the attractor-level dynamical system."""
        x_two = x_t[2:(self.num_moves + 2)]
        weights = x_two / self.sigma
        x5_tp1 = 0
        u_t = 0
        for idx, soc in enumerate(self.socs):
            out = soc(t, x5_t)
            x5_tp1 += out[0] * weights[idx]
            u_t += out[1] * weights[idx]
        return x5_tp1, u_t

    def gen_mix_socs(self):
        """Build a generator to mix SOCs online."""
        times = np.zeros(len(self.socs))
        active_socs = np.zeros(len(self.socs), dtype=bool)
        blacklist_socs = np.zeros(len(self.socs), dtype=bool)
        x5_t, y5_t, x_t, u_t, t = yield
        while 1:
            x_two = x_t[2:(self.num_moves + 2)]
            weights = x_two / self.sigma
            new_active = weights >= self.threshold
            active_socs[new_active * (1 - blacklist_socs).astype(bool)] = 1
            weights[~active_socs] = 0
            if weights.sum() == 0:
                weights = np.zeros_like(weights)
            else:
                weights /= weights.sum()
            # ipdb.set_trace()
            blacklist_socs[new_active] = 1
            times[active_socs] = times[active_socs] + 1
            for idx, time in enumerate(times):
                if time >= int(self.t_ends[idx] / self.dt):
                    times[idx] = 0
                    active_socs[idx] = 0
            x5_tp1 = np.zeros_like(x5_t)
            u_tp1 = np.zeros(self.phys_dim)
            for idx, soc in enumerate(self.socs):
                if not active_socs[idx]:
                    continue
                # ipdb.set_trace(cond=not active_socs[0] and np.sum(active_socs) > 1)
                # ipdb.set_trace(cond=times[idx] > 170)
                out = soc(times[idx] * self.dt, x5_t, y5_t, u_t)
                x5_tp1 += out[0][:, :5] * weights[idx]  # HACK!
                u_tp1 += np.squeeze(out[1] * weights[idx])
                # ipdb.set_trace(cond=idx > 0)
            x5_t, y5_t, x_t, u_t, t = yield x5_tp1, u_tp1

    def arm_dynamics_many(self, x5N_t, uN_t, dyn_kwargs=None):
        """Move the arm in all dimensions given motor command.

        Call self.arm.dynamics for each element of the first dimension
        of x5N_t and u_t.

        """
        if dyn_kwargs is None:
            dyn_kwargs = {}
        x5N_tp1 = []
        yN_tp1 = []
        for x5_t, u_t in zip(x5N_t, uN_t):
            x5_tp1, y_tp1 = self.arm.dynamics(x5_t, u_t, **dyn_kwargs)
            x5N_tp1.append(x5_tp1)
            yN_tp1.append(y_tp1)
        return np.array(x5N_tp1), np.array(yN_tp1)

    def move(self, x_t, t_ini, t_end, x_noise=0, id_stop=None,
             dyn_kwargs=None):
        """Override parent to make SOCs go.

            Divides integration into intervals of --monitor_delta_t-- length,
        after each of which the monitor is called to see if the current
        element of the sequence has been reached.

        Parameters
        ----------
        x_t : ndarray

        t_ini, t_end : float
        Initial and final times for integration.

        x_noise : ndarray
        Noise to add to the dynamics. self.x_dot decides how to do this.

        id_stop : str
        If provided, the simulation will end as soon as the circle indexed by
        id_stop has been reached. For this, self.Basket is used.

        Returns
        -------
        all_x : ndarray
        Values for the vector x_t after each call to the monitor.
        """
        if t_end - t_ini == 0:
            return x_t[None, ...], np.array([t_ini, t_end])
        if np.ndim(x_noise) == 0:
            x_noise = x_noise * np.ones_like(x_t)
        if self.delta_t_monitor == 0:
            t_vec = [np.arange(t_ini, t_end, self.dt)]
        else:
            tstep = np.arange(t_ini, t_end, self.delta_t_monitor)
            t_vec = [np.arange(c_t, n_t, self.dt)
                     for c_t, n_t in zip(tstep, tstep[1:])]
        all_x = [x_t, ]
        all_t = [0, ]
        all_u = []
        c_devs = np.array([[0, 0, 0], [0, 0, 0]])
        x5_t = np.hstack([x_t[:self.phys_dim][:, None], c_devs, [[1], [1]]])
        xhat_t = x5_t
        all_xhat = [x5_t, ]
        y5_t = np.vstack([self.arm.H.dot(x_one) for x_one in x5_t])
        noise_vec = np.ones_like(x_t)
        noise_vec[:self.phys_dim] = 0
        mamix = self.gen_mix_socs()
        mamix.send(None)
        for c_tvec in t_vec:
            c_mon = self.monitor(x_t)
            noise = self.monitor_sd * c_mon
            for c_t in c_tvec:
                xhat_t, u_t = mamix.send((x5_t, y5_t, x_t, u_tc_t))
                x5_t, y5_t = self.arm_dynamics_many(x5_t, u_t,
                                                    dyn_kwargs=dyn_kwargs)
                # x_t = super().move(x_t, c_t, c_t + self.dt)[0][-1, ...]
                x_t = x_t + (self.x_dot(c_t, x_t) + noise ** 2) * self.dt
                if not c_mon:
                    x_t[x_t < 0.05] = 0
                x_t[:self.phys_dim] = x5_t[..., 0]
                all_x.append(x_t)
                all_t.append(c_t)
                all_u.append(u_t)
                all_xhat.append(xhat_t)
                if not (id_stop is None):
                    continue
                    cart_x = self.angles_to_xy(
                        *(x_t[:self.phys_dim] * np.array([1, -1])))
                    if self.basket[id_stop](*cart_x):
                        # logging.info(
                        #     'You have reached your destination. Stopping sim')
                        break
        # To make it the same size as x_t
        all_u.append(np.zeros_like(all_u[-1]))
        return np.array(all_x), np.array(all_t), np.array(all_xhat), np.array(all_u)

    def move_single(self, ix_single, **kwargs):
        """Move the physical space using the SOC indicated by ix_single.

        Note the SHS dimensions of --x_t-- are ignored in order to force a single
        SOC to take over.

        Parameters
        ----------
        ix_single : int
        Indexes the single-target movement, as per self.transitions.

        """
        old_monitor = self.monitor_sd
        self.monitor_sd = 0
        if ix_single == 0:
            ix_single_eff = 1
            self.choose_seq(0)
        elif ix_single == 1:
            self.choose_seq(3)
            ix_single_eff = 2
        x0 = np.zeros(self.Ns + self.phys_dim)
        x0[ix_single_eff + self.phys_dim] = self.sigma[ix_single_eff]
        x0[:self.phys_dim] = kwargs['x_t'][:self.phys_dim]
        kwargs['x_t'] = x0
        outie = self.move(**kwargs)
        self.monitor_sd = old_monitor
        return outie


class SimpleSOCAngles(SimpleSOC):
    """Version with angles instead of Descartes."""

    name = 'Simple SOC angles'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_circles(self, x, y, a1=None, a2=None):
        """Transforms the coordinates on the experimental design into angles,
        with a two-link arm (lenghts a1 and a2), with a shoulder anchored at
        self.shoulder_anchor.
        """
        if a1 is None:
            a1 = self.a1
        if a2 is None:
            a2 = self.a2
        x, y = np.array([x, y]) - self.shoulder_anchor
        q2 = np.arccos((x ** 2 + y ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2))
        q1 = np.arctan2(y, x) - np.arctan2(a2 *
                                           np.sin(q2), a1 + a2 * np.cos(q2))
        return q1, q2

    def angles_to_xy(self, q1, q2, a1=None, a2=None):
        """Inverse transformation to transform_circles()"""
        if a1 is None:
            a1 = self.a1
        if a2 is None:
            a2 = self.a2
        x = a1 * np.cos(q1) + a2 * np.cos(q1 + q2)
        y = a2 * np.sin(q1 + q2) + a1 * np.sin(q1)
        return np.array([x, y]) + self.shoulder_anchor[:, None]


class SimpleTodorovOriginal(Simple):
    """Uses the dynamical system to create target trajectories and then uses
    TodorovLinear to select the motor commands and execute the movement.

    """

    name = 'Simple Todorov with Linear'

    def __init__(self, *args, **kwargs):
        self.arm = single.LinearTodorov()
        self.monitor_time = 0.5
        super().__init__(*args, **kwargs)

    def move(self, x_t, t_ini, t_end, x_noise=0):
        """Does a full forward simulation and then calculates motor commands
        to match it."""
        monitor_times = np.arange(t_ini, t_end, self.monitor_time)
        all_x = []
        noise_vec = np.ones_like(x_t)
        noise_vec[:self.phys_dim] = 0
        all_x, all_t = super().move(x_t, t_ini, t_end)
        all_u = []
        x5_t, all_u = self.arm.inv_dynamics(all_x[:, :self.phys_dim])
        all_x5 = self.arm.move(x5_t[0, ...], all_u, with_noise=True)
        all_x[:, :self.phys_dim] = all_x5[..., 0]
        return all_x, all_t, all_u


class vpSOC():
    """Bundle of multiple SOCSeqs with equilibrium points as in the experiment.

    In a way, as naive as it is magical, it's similar to Simple and its
    derivatives, but the sequential part of the movement is taken care of by
    the SOCSeqs.

    """

    name = 'vpSOC'

    def __init__(self, t_ends=1, h_scaling=None, timings=None,
                 soc_pars=None, full_circle=True, ix_seqs=None, **kwargs):
        """Instanciate the vpSOC model with the requested parameters.

        Parameters
        ----------
        timings : ndarray, size=(3, )
        Timings for each one of the targets in the sequence, in absolute terms
        (i.e. in seconds). One element per target, starting with the first
        target (i.e. not pt_start). If None, they are evenly distributed in
        time.

        ix_seqs : list
        If a list, only the sequences with the requested indices are used. This
        can be used to speed up the initialization of the controllers, but will
        return an "incomplete" agent, which doesn't have all the sequences in
        the experiment. Aditionally, when requesting a certain sequence
        (e.g. self.move(..., ix_seq=N)), that indexing will no longer reflect
        that of the experiment, but of ix_seqs. For example, if ix_seqs=[0, 3],
        when asking the agent to perform sequence 1 (i.e. self.move(...,
        ix_seq=1)), it will perform the sequence of index 3 in the experiment.

        """
        self.full_circle = full_circle
        self.h_scaling = h_scaling
        self.timings = timings
        if soc_pars is None:
            soc_pars = {}
        self.soc_pars = soc_pars
        self.dt = 0.01
        self.phys_dim = 2
        basket = utils.coords_from_svg(reference='mk_tr')
        self.basket = basket
        circles = ['pt_start', 'pt_left', 'pt_right', 'st_left', 'st_top',
                   'st_right']
        self.labels = circles
        self.a1 = self.a2 = 300
        self.shoulder_anchor = np.array([-100, -400])  # Roughly shoulder
        self.centers = np.array([self.transform_circles(*basket[key].center)
                                 for key in circles])
        sequences = np.array([[0, 1, 3, 9], [0, 1, 5, 10], [0, 1, 4, 11],
                              [0, 2, 6, 9], [0, 2, 8, 10], [0, 2, 7, 11]])
        if not self.full_circle:
            sequences = sequences[:, :-1]
        if ix_seqs is not None:
            sequences = sequences[ix_seqs]
        self.sequences = sequences
        transitions = np.array([[0, 0], [0, 1], [0, 2], [1, 3], [1, 4], [1, 5],
                                [2, 3], [2, 4], [2, 5], [3, 0], [4, 0], [5, 0]])
        self.transitions = transitions
        if np.ndim(t_ends) == 0:
            t_ends = t_ends * np.ones(len(self.sequences))
        self.t_ends = t_ends
        # self.arm = single.SOCSeq(*args, **kwargs)
        self.socs = self.build_socseqs()
        self.socsins = self.build_socsins()
        self.c_seq = 0

        self.parameters = self.set_parameter_dict()

    def set_parameter_dict(self, ):
        """Set up the parameter dictionary for databases."""
        par_list = ['h_scaling', 'timings', 't_ends', 'timings', 'full_circle',
                    'phys_dim', 'a1', 'a2', 'shoulder_anchor']
        pardict = {key: getattr(self, key) for key in par_list}
        pardict.update(self.soc_pars)
        return pardict

    def build_socseqs(self,):
        """Build all the multiple-target socs."""
        all_socs = []
        self.socseq_agents = []
        for ix_seq, sequence in enumerate(self.sequences):
            t_end = self.t_ends[ix_seq]
            targets = [self.centers[self.transitions[c_seq, 1]]
                       for c_seq in sequence]
            all_socs.append(self._build_socseq(targets[0],
                                               targets[1:],
                                               t_end))
        return all_socs

    def build_socsins(self, ):
        """Build all the single-target socs."""
        all_socs = []
        self.socsin_agents = []
        pos_ini = self.basket['pt_start'].center
        t_end = self.t_ends[0]  # HACK
        for ix_seq, second in enumerate(['pt_left', 'pt_right']):
            target_center = self.basket[second].center
            all_socs.append(self._build_sinsoc(pos_ini, target_center, t_end))
        return all_socs

    def _build_sinsoc(self, pos_ini, pos_end, t_end):
        """Use the SOC model instead of attractors."""
        LNt = [[]] * self.phys_dim
        KNt = [[]] * self.phys_dim
        socky = single.SOC(t_end=t_end, dt=self.dt, **self.soc_pars)
        self.socky = socky
        self.socsin_agents.append(socky)
        # print(t_end)
        for ix_dim in range(self.phys_dim):
            x5_ini = np.array((pos_ini[ix_dim], 0, 0, 0, pos_end[ix_dim]))
            LNt[ix_dim], KNt[ix_dim] = socky.optimal_controler(x5_ini)

        if (pos_ini == pos_end).all():
            KNt = [np.zeros(KNt[0].shape), np.zeros(KNt[1].shape)]
            LNt = [np.zeros(LNt[0].shape), np.zeros(LNt[1].shape)]

        def dynamic(x_ini, dyn_kwargs=None):
            """Return x5[t + 1] given x5_t.

            Parameters
            ----------
            x5N_t : iterable
            Contains the x5_t vectors, with as many elements as there are
            physical dimensions (2D for the surface on a table). Each element
            is an ndarray with 5 elements (x5_t; position, speed, etc).

            """
            x5N_t = np.hstack([x_ini[:, None],
                               np.array([[0, 0, 0], [0, 0, 0]]),
                               pos_end[:, None]])
            all_x = []
            all_u = []
            for Kt, Lt, x_one in zip(KNt, LNt, x5N_t):
                xhat, x, y, u = socky.move(x_one, Kt=Kt, Lt=Lt,
                                           dyn_kwargs=dyn_kwargs)
                all_x.append(x)
                all_u.append(u)
            return all_x, all_u
        return dynamic

    def _build_socseq(self, pos_ini, goals, t_end):
        """Build one socseq."""
        LNt = [[]] * self.phys_dim
        KNt = [[]] * self.phys_dim
        extra_dims = self.sequences.shape[1] - 2
        socky = single.SOCSeq(t_end=t_end, dt=self.dt, timings=self.timings,
                              h_scaling=self.h_scaling, extra_dims=extra_dims,
                              **self.soc_pars)
        self.socseq_agents.append(socky)
        for ix_dim in range(self.phys_dim):
            c_goals = [c_goal[ix_dim] for c_goal in goals]
            x5_ini = np.array((pos_ini[ix_dim], 0, 0, 0, *c_goals))
            LNt[ix_dim], KNt[ix_dim] = socky.optimal_controler(x5_ini)

        def dynamic(x_ini, dyn_kwargs=None):
            """Return x5[t + 1] given x5_t.

            Parameters
            ----------
            x5N_t : iterable
            Contains the x5_t vectors, with as many elements as there are
            physical dimensions (2D for the surface on a table). Each element
            is an ndarray with 5 elements (x5_t; position, speed, etc).

            """
            e_goals = [goal[:, None] for goal in goals]
            x6N_t = np.hstack([x_ini[:, None],
                               np.array([[0, 0, 0], [0, 0, 0]]), *e_goals])
            all_x = []
            all_u = []
            for Kt, Lt, x_one in zip(KNt, LNt, x6N_t):
                xhat, x, y, u = socky.move(x_one, Kt=Kt, Lt=Lt,
                                           dyn_kwargs=dyn_kwargs)
                all_x.append(x)
                all_u.append(u)
            return all_x, all_u
        return dynamic

    def transform_circles(self, x, y, *args, **kwargs):
        """Hold the place."""
        return x, y

    def angles_to_xy(self, q1, q2, *args, **kwargs):
        """Hold the place."""
        return q1, q2

    def set_initial_conditions(self, *args, **kwargs):
        """Dummy function.

        Doesn't even have the decency to pretend that it's doing anything. What
        a bum.

        Exists for compatibility purposes.

        *args and **kwargs are there for compatibility with other models.

        """
        return np.zeros(self.phys_dim)

    def choose_seq(self, ix_seq, *args, **kwargs):
        """Exists for compatibility purposes.

        Helps with making self.move
        compatible with the corresponding method of the Simple class.

        *args and **kwargs are there for compatibility with other models.
        """
        self.c_seq = ix_seq

    def move_single(self, ix_single, *args, **kwargs):
        """Move single-target hands."""
        kwargs['ix_seq'] = ix_single
        return self.move(*args, seq=False, **kwargs)

    def move(self, x_t, ix_seq=None, seq=True, dyn_kwargs=None, **kwargs):
        """Move.

        dyn_kwargs are sent to dynamics()

        *args and **kwargs are there for compatibility with other models.
        """
        c_socs = self.socs if seq else self.socsins
        if ix_seq is None:
            ix_seq = self.c_seq
        all_x, all_u = c_socs[ix_seq](x_t, dyn_kwargs=dyn_kwargs)
        all_x = np.moveaxis(np.array(all_x), [0, 1, 2], [1, 0, 2])[:, :, 0]
        all_u = np.array(all_u).squeeze().T
        all_xhat = all_x  # compat with SimpleSOC
        all_t = np.arange(0, self.t_ends[ix_seq] + self.dt, self.dt)
        return all_x, all_t, all_xhat, all_u


def HieSOCSeqAngles(HieSOCSeq):
    """Version with angles instead of DeCartes."""

    def angles_to_xy(self, q1, q2, a1=None, a2=None):
        """Inverse transformation to transform_circles()."""
        if a1 is None:
            a1 = self.a1
        if a2 is None:
            a2 = self.a2
        x = a1 * np.cos(q1) + a2 * np.cos(q1 + q2)
        y = a2 * np.sin(q1 + q2) + a1 * np.sin(q1)
        return np.array([x, y]) + self.shoulder_anchor[:, None]

    def transform_circles(self, x, y, a1=None, a2=None):
        """Transform the coordinates on the experimental design into angles.

        Assumes a two-link arm (lenghts a1 and a2), with a shoulder anchored at
        self.shoulder_anchor.
        """
        if a1 is None:
            a1 = self.a1
        if a2 is None:
            a2 = self.a2
        x, y = np.array([x, y]) - self.shoulder_anchor
        q2 = np.arccos((x ** 2 + y ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2))
        q1 = np.arctan2(y, x) - np.arctan2(a2 *
                                           np.sin(q2), a1 + a2 * np.cos(q2))
        return q1, q2


class HiSeq(SimpleSOC):
    """Model specifically for some parts of the paper.

    This child hijacks SimpleSOC's build_socs functions to enable the
    possibility of mixing vpSOCs and SOCs in one trajectory. To do this, it
    generalizes the transitions to any number of elements, and if it's bigger
    than 2, it calls a vpSOC instead.

    """

    name = 'HiSeq(vpsOC(2)+SOC)'  # Can get overwritten in __init__

    def __init__(self, ix_seqs=None, name=None, **kwargs):
        """Cookies and SOC-based milk.

        Parameters
        ----------
        ix_seqs : list
        If a list, only the sequences with the requested indices are used. This
        can be used to speed up the initialization of the controllers, but will
        return an "incomplete" agent, which doesn't have all the sequences in
        the experiment. Aditionally, when requesting a certain sequence
        (e.g. self.move(..., ix_seq=N)), that indexing will no longer reflect
        that of the experiment, but of ix_seqs. For example, if ix_seqs=[0, 3],
        when asking the agent to perform sequence 1 (i.e. self.move(...,
        ix_seq=1)), it will perform the sequence of index 3 in the experiment.

        name : str
        Name to give to the model. By default, this model is called HiSeq, and
        will be saved so in the databases. If you provide custom sequences or
        transitions, you might want to provide also a new name (such as
        HiSeq(SOC+SOC+SOC)).

        """
        if 'transitions' not in kwargs:
            transitions = [[0, 0],
                           [0, 1, 3], [0, 1, 5], [0, 1, 4],
                           [0, 2, 3], [0, 2, 5], [0, 2, 4],
                           [3, 6], [4, 6], [5, 6]]
            self.name = 'HiSeq(vpSOC(2)+SOC)'
            kwargs['transitions'] = transitions
            kwargs['t_ends'] = np.array([1, 2, 2, 2, 2, 2, 2, 1, 1, 1])
        if 'sequences' not in kwargs:
            sequences = np.array([[0, 1, 7], [0, 2, 9], [0, 3, 8],
                                  [0, 4, 7], [0, 5, 9], [0, 6, 8]], dtype=object)
            if ix_seqs is not None:
                sequences = sequences[ix_seqs]
            kwargs['sequences'] = sequences
        if 'h_scaling' not in kwargs:
            self.h_scaling = 1
        else:
            self.h_scaling = kwargs.pop('h_scaling')
        self.delta_t_monitor = 0
        super().__init__(**kwargs)
        self.parameters = self.set_parameter_dict()

    def set_parameter_dict(self, ):
        """Set up the parameter dictionary for databases."""
        par_list = ['h_scaling', 't_ends', 'phys_dim', 'a1', 'a2',
                    'shoulder_anchor']
        pardict = {key: getattr(self, key) for key in par_list}
        pardict.update(self.def_pars)
        pardict.update(self.soc_pars)
        return pardict

    def fixed_models(self,):
        """Return dummy functions for compatibility."""
        def dummy(*args, **kwargs):
            return np.zeros(self.phys_dim)
        return [dummy] * self.num_moves

    def build_socs(self,):
        """Build SOC or SeqSOCS, depending on the wind."""
        transitions = self.transitions
        centers = self.centers
        moves = [[]] * self.num_moves
        h_scalings_all = np.array(self.h_scaling, dtype=object)
        # if np.ndim(h_scalings_all) == 1:
        #     h_scalings_all = np.tile(h_scalings_all[None, :], (self.num_moves, 1))
        # elif np.ndim(h_scalings_all) == 0:
        #     h_scalings_all = h_scalings_all * np.ones(self.num_moves)

        for ix_model in range(self.num_moves):
            t_end = self.t_ends[ix_model]
            c_trans = transitions[ix_model]
            pos_ini = centers[c_trans[0]]
            targets = [centers[target] for target in c_trans[1:]]
            moves[ix_model] = self._build_socseq(pos_ini, targets,
                                                 t_end,
                                                 h_scalings_all[ix_model])
        return moves

    def _build_socseq(self, pos_ini, goals, t_end, h_scaling):
        """Build one socseq."""
        LNt = [[]] * self.phys_dim
        KNt = [[]] * self.phys_dim
        extra_dims = len(goals) - 1
        socky = single.SOCSeq(t_end=t_end, dt=self.dt,
                              h_scaling=h_scaling, extra_dims=extra_dims,
                              **self.soc_pars)
        for ix_dim in range(self.phys_dim):
            c_goals = [c_goal[ix_dim] for c_goal in goals]
            x5_ini = np.array((pos_ini[ix_dim], 0, 0, 0, *c_goals))
            LNt[ix_dim], KNt[ix_dim] = socky.optimal_controler(x5_ini)

        def dynamic(t, x5Nhat_t, y5N_tp1, uN_t):
            """Return x5[t + 1] given x5_t.

            Parameters
            ----------
            x5N_t : iterable
            Contains the x5_t vectors, with as many elements as there are
            physical dimensions (2D for the surface on a table). Each element
            is an ndarray with 5 elements (x5_t; position, speed, etc).

            """
            x5Nhat_t = np.hstack([x5Nhat_t[:, :4], np.array(goals).T])
            # x5N_t[:, -1] = pos_end
            ct = int(t / self.dt)
            x5N_tp1 = []
            uN_tp1 = []
            zippy = zip(KNt, LNt, x5Nhat_t, y5N_tp1, uN_t)
            for Kt, Lt, x5hat_t, y5_tp1, u_t in zippy:
                xhat_tp1, u_tp1 = socky.estimate_and_control(x5hat_t, y5_tp1,
                                                             u_t, Kt[ct],
                                                             Lt[ct])
                x5N_tp1.append(xhat_tp1)
                uN_tp1.append(u_tp1)
            return np.array(x5N_tp1), np.array(uN_tp1)

        return dynamic

    def move(self, x_t, t_ini, t_end, x_noise=0, id_stop=None, dyn_kwargs=None):
        """Override parent to make SOCs go.

            Divides integration into intervals of --monitor_delta_t-- length,
        after each of which the monitor is called to see if the current
        element of the sequence has been reached.

        Parameters
        ----------
        x_t : ndarray
        Initial conditions in a space of dimension 2 + (number of transitions),
        where a transition refers from the movement from one circle to another.
        By default, there are 12 transitions: from the starting point to each
        primary target (2), from each primary target to each secondary target (6),
        from each secondary target back to the initial position (3) and a dummy
        transition (technically from initial position to initial position) to
        model lack of movement at the beginning of a trial.

        t_ini, t_end : float
        Initial and final times for integration.

        x_noise : ndarray
        Noise to add to the dynamics. self.x_dot decides how to do this.

        id_stop : str
        If provided, the simulation will end as soon as the circle indexed by
        id_stop has been reached. For this, self.Basket is used.

        Returns
        -------
        all_x : ndarray
        Values for the vector x_t after each call to the monitor.
        """
        if t_end - t_ini == 0:
            return x_t[None, ...], np.array([t_ini, t_end])
        if np.ndim(x_noise) == 0:
            x_noise = x_noise * np.ones_like(x_t)
        if self.delta_t_monitor == 0:
            t_vec = [np.arange(t_ini, t_end, self.dt)]
        else:
            tstep = np.arange(t_ini, t_end, self.delta_t_monitor)
            t_vec = [np.arange(c_t, n_t, self.dt)
                     for c_t, n_t in zip(tstep, tstep[1:])]
        all_x = [x_t, ]
        all_t = [0, ]
        all_u = []
        c_devs = np.array([[0, 0, 0], [0, 0, 0]])
        x5_t = np.hstack([x_t[:self.phys_dim][:, None], c_devs, [[1], [1]]])
        xhat_t = x5_t
        all_xhat = [x5_t, ]
        y5_t = np.vstack([self.arm.H.dot(x_one) for x_one in x5_t])
        noise_vec = np.ones_like(x_t)
        noise_vec[:self.phys_dim] = 0
        mamix = self.gen_mix_socs()
        mamix.send(None)
        cart_x = self.angles_to_xy(*(x_t[:self.phys_dim]))
        u_t = np.zeros(self.phys_dim)
        if id_stop is not None:
            flag_start_end = self.basket[id_stop](cart_x)
        for c_tvec in t_vec:
            for c_t in c_tvec:
                c_mon = self.monitor(x_t)
                noise = self.monitor_sd * c_mon
                shc_noise = 0 * np.random.randn(*x5_t.shape)
                shc_noise[:, :2] = 0
                xhat_t, u_t = mamix.send(
                    (xhat_t + shc_noise, y5_t, x_t, u_t, c_t))
                x5_t, y5_t = self.arm_dynamics_many(x5_t, u_t,
                                                    dyn_kwargs=dyn_kwargs)
                # x_t = super().move(x_t, c_t, c_t + self.dt)[0][-1, ...]
                x_t = x_t + (self.x_dot(c_t, x_t) + noise ** 2) * self.dt
                x_t[:self.phys_dim] = x5_t[:, 0]
                if not c_mon:
                    x_t[x_t < 0.5] = 0
                x_t[:self.phys_dim] = x5_t[..., 0]
                all_x.append(x_t)
                all_t.append(c_t)
                all_u.append(u_t)
                all_xhat.append(xhat_t)
                if not (id_stop is None):
                    # continue
                    cart_x = self.angles_to_xy(
                        *(x_t[:self.phys_dim]))
                    if self.basket[id_stop](cart_x):
                        # logging.info(
                        #     'You have reached your destination. Stopping sim')
                        if not flag_start_end:
                            break
                    else:
                        flag_start_end = False
        # To make it the same size as x_t
        all_u.append(np.zeros_like(all_u[-1]))
        return np.array(all_x), np.array(all_t), np.array(all_xhat), np.array(all_u)
