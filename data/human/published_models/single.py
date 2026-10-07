"""Single-movement models from the literature, implemented for your pleasure.
"""
import logging

from numpy.linalg import multi_dot as mdot
from scipy.linalg import block_diag
import numpy as np


class LinearTodorov():
    """Model for hand movements.

    Implements the linear model of "hand movements" as presented in:

    Todorov, Emanuel. “Stochastic Optimal Control and Estimation Methods
    Adapted to the Noise Characteristics of the Sensorimotor System.” Neural
    Computation 17, no. 5 (May 1, 2005):
    1084–1108. https://doi.org/10.1162/0899766053491887.

    To extend this to N-dimensional movement, N copies of the 1-dimensional
    system are used, which means that the dimensions are identical and
    independent from each other.

    """

    name = 'Linear Todorov'

    def __init__(self, x_ends=None, random_seed=None):
        """Does nothing much for now"""
        self.random = np.random.default_ring(random_seed)
        if x_ends is None:
            x_ends = np.array([[0, 0], [1, 1]])
        self.N = x_ends.shape[1]
        self.num_splines = 5
        self.tau1 = 0.04
        self.tau2 = 0.04
        self.delta = 0.01
        self.m = 1
        self.sigma_c = 0.1
        self.kappa = 1.5  # Control-dependent noise parameter
        A, B, C, H = self.set_matrices()
        self.A = A
        self.B = B
        self.C = C  # Currently not used!
        self.H = H
        self.obs_noise = 0.1  # Couldn't find this in the Todorov 1998 paper.

    def _dynamics_1d(self, x_t, u_t, with_noise=False):
        """Dynamics model for 1D. Returns x_{t+1} given x_t and u_t. This is
        the model described in Todorov (1998).

        Note that, as done in Todorov (1998), the control signal is a scalar,
        which is then multiplied by the vector C. This noise term is used
        only if --with_noise-- is True.

        """
        if with_noise:
            noise = self.random.normal() * self.B.dot(u_t) * self.sigma_c
        else:
            noise = 0
        return self.A.dot(x_t) + self.B.dot(u_t) + noise

    def dynamics(self, x_t, u_t, with_noise=False):
        """Dynamics model for N-dimensional movements. """
        new_x = np.zeros_like(x_t)
        for idx in range(self.N):
            new_x[idx] = self._dynamics_1d(x_t[idx, ...], u_t[idx],
                                           with_noise=with_noise)
        return new_x

    def dynamics_many(self, x_t, u_all, with_noise=False):
        """Propagates --x_t-- through the dynamics of the arm as many times as
        --u_all-- has elements.

        Parameters
        ----------
        u_all : ndarray[float, float]
        Control signals for all trials in the future. The state --x_t-- will
        be propagated through the dynamics as many times as u_all.shape[0].

        """
        num_steps = len(u_all)
        all_x = [x_t]
        for idx in range(num_steps):
            x_t = self.dynamics(x_t, u_all[idx], with_noise=with_noise)
            all_x.append(x_t)
        return np.array(all_x)

    def move(self, x_ini, u_t, **kwargs):
        """Makes the thing GO for many trials at a time.

        Parameters
        ----------
        x_ini : ndarray
        Initial position. Size (self.N, 5).

        u_t : ndarray
        Control states for all trials. Size (num_trials, self.N).

        **kwargs are sent to self.dynamics

        """
        num_trials = u_t.shape[0]
        x_t = np.zeros((num_trials, *x_ini.shape))
        c_x = x_ini
        for trial, u_tau in enumerate(u_t):
            x_t[trial, ...] = self.dynamics(c_x, u_t[trial, :], **kwargs)
            c_x = x_t[trial, ...]
        return x_t

    def inv_dynamics(self, all_x, x_targets=(0, 1), dx_targets=(0, 0),):
        """Returns the necessary control signal u_t to go from x_t to x_tp1 in
        one time step, for all the steps in --all_x--. Note that, by necessity,
        it reconstructs the entire X vector for each time point; this vector is
        also returned, for shits and giggles.

        Parameters
        ----------
        all_x : ndarray
        Size = (num_trials, self.N)

        """
        if np.ndim(all_x) == 1:
            all_x = all_x[:, None]
        num_trials = all_x.shape[0]
        x_t = np.zeros((num_trials, self.N, 5))
        u_t = np.zeros((num_trials, self.N))
        x_t[..., -1] = 1
        x_t[..., 0] = all_x
        x_t[:-1, :, 1] = np.diff(all_x, axis=0) / self.delta
        x_t[:-2, :, 2] = np.diff(x_t[:-1, :, 1], axis=0) / self.delta * self.m
        shifted_f = x_t[:-3, :, 2] * (1 - self.delta / self.tau2)
        unshifted_f = x_t[1:-2, :, 2]
        x_t[:-3, :, 3] = (unshifted_f - shifted_f) / self.delta * self.tau2
        shifted_g = x_t[:-4, :, 3] * (1 - self.delta / self.tau1)
        unshifted_g = x_t[1:-3, :, 3]
        u_t[:-4, ...] = (unshifted_g - shifted_g) / self.delta * self.tau1
        return x_t, u_t

    def observations(self, x_t, with_noise=False):
        """Observation function with or without noise."""
        if with_noise:
            noise = self.noise_obs * self.random.normal()
        else:
            noise = 0
        return self.H.dot(x_t) + noise

    def set_matrices(self,):
        """Sets the matrices for the dynamics model as defined in Todorov
        (1998)."""
        tau1 = self.tau1
        tau2 = self.tau2
        delta = self.delta
        m = self.m
        sigma_c = self.sigma_c  # 0.5
        A = np.array([[1, delta, 0, 0, 0],
                      [0, 1, delta / m, 0, 0],
                      [0, 0, 1 - delta / tau2, delta / tau2, 0],
                      [0, 0, 0, 1 - delta / tau1, 0],
                      [0, 0, 0, 0, 1]])
        B = np.array([0, 0, 0, delta / tau1, 0])
        H = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
        C = sigma_c * B
        return A, B, C, H

    def covariance(self, u_t):
        """Returns the covariance at the final point X_T given the control
        states u_t.

        As with everything in life, all the dimensions are handled
        independently and the results are returned as a set of covariance
        matrices, i.e. size = (self.N, *A.shape)

        """
        final_t = len(u_t)
        A = self.A
        B = self.B
        k = self.kappa
        all_covs = np.zeros((self.N, final_t, *A.shape))
        for idx in np.arange(self.N):
            cov = np.zeros((final_t, *A.shape))
            for i in np.arange(final_t):
                thing = np.linalg.matrix_power(A, final_t - i - 1).dot(B)
                cov[i, ...] = thing.dot(thing.T) * u_t[i, idx] ** 2
                all_covs[idx, i, ...] = k * np.sum(cov[:i + 1, ...], axis=0)
        return all_covs


class SOC:
    """Implement the Stochastic Optimal Control model.

    The implementation comes from:
    Todorov E. Stochastic Optimal Control and Estimation Methods Adapted to the
    Noise Characteristics of the Sensorimotor System. Neural Computation.
    2005 May 1;17(5):1084–108.

    """

    name = 'SOC'

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

    def __init__(self, random_seed=None, random_generator=None, **pars):
        """Initialize all default and modified parameters and matrices.

        All parameters have a default value defined in self.def_pars.

        """
        if random_generator is None:
            self.random = np.random.default_rng(random_seed)
        else:
            self.random = random_generator
        def_pars = self.def_pars.copy()
        def_pars.update(pars)
        self.pars = def_pars
        for key, value in def_pars.items():
            setattr(self, key, value)
        self.T = int(self.t_end / self.dt)

        A, B, C, H, D = self.define_forward_model()
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        self.D = D
        Qt, R = self.define_cost_matrices()
        self.Qt = Qt
        self.R = R
        self.add_noise = self.pars['add_noise']
        noises = self.define_noise_covariances()
        self.Omega_xi, self.Omega_w, self.Omega_eta, self.Sigma_1 = noises

    def __eq__(self, other):
        """Compare all important attributes to determine equality."""
        atts = ['A', 'B', 'C', 'D', 'H', 'Qt', 'R', 'Omega_xi', 'Omega_w',
                'Omega_eta', 'Sigma_1']
        all_atts = atts + list(self.def_pars.keys())
        for att in all_atts:
            try:
                comp = getattr(self, att) == getattr(other, att)
            except AttributeError:
                return False
            try:
                comp = comp.all()
            except AttributeError:
                pass
            if not comp:
                return False
        return True

    def define_forward_model(self):
        """Set up the matrices for the forward model.

        This is for a one-dimensional system.
        """
        tau1 = self.tau1
        tau2 = self.tau2
        delta = self.delta
        m = self.m
        sigma_c = self.sigma_c  # 0.5
        A = np.array([[1, delta, 0, 0, 0],
                      [0, 1, delta / m, 0, 0],
                      [0, 0, 1 - delta / tau2, delta / tau2, 0],
                      [0, 0, 0, 1 - delta / tau1, 0],
                      [0, 0, 0, 0, 1]])
        B = np.array([0, 0, 0, delta / tau1, 0])[:, None]
        H = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
        C = sigma_c * B
        D = np.zeros((self.M, self.N))  # Observation noise matrix

        return A, B, C, H, D

    def dynamics(self, x_t, u_t, motor_noise=True, obs_noise=True,
                 add_noise=True):
        """Return the next state and obs given current state and control."""
        mnoise = motor_noise * \
            self.C.dot(np.squeeze(u_t)) * self.random.normal()
        jnoise = add_noise * self.add_noise * self.random.normal() * self.B ** 2
        onoise = obs_noise * self.D.dot(x_t) * self.random.normal()
        x_tp1 = self.A.dot(x_t) + np.squeeze(self.B.dot(u_t)) + np.squeeze(
            mnoise) + np.squeeze(jnoise)
        y_tp1 = self.H.dot(x_tp1) + np.squeeze(onoise)
        return x_tp1, y_tp1

    def estimate_and_control(self, xhat_t, y_tp1, u_t, Kt, Lt):
        """Estimate the next state xhat_tp1 and issue motor command u_t."""
        A = self.A
        B = self.B
        H = self.H
        xhat_tp1_t = A.dot(xhat_t) + np.squeeze(B.dot(u_t))
        xhat_tp1 = xhat_tp1_t + Kt.dot(y_tp1 - H.dot(xhat_tp1_t))
        u_tp1 = -Lt.dot(xhat_tp1)
        return xhat_tp1, u_tp1

    def define_cost_matrices(self):
        """Define the cost matrices Q_t and R."""
        Qt = np.zeros((self.T, self.N, self.N))
        p = self.p_scaling * np.array([1, 0, 0, 0, -1])[:, None]
        v = np.array([0, self.omega_nu, 0, 0, 0])[:, None]
        f = np.array([0, 0, self.omega_f, 0, 0])[:, None]
        Qt[-1, ...] = p.dot(p.T) + v.dot(v.T) + f.dot(f.T)

        R = np.eye(self.Nc) * self.r
        return Qt, R

    def define_noise_covariances(self, ):
        """Noise covariance matrices."""
        Omega_xi = self.add_noise * np.ones((self.N, 1))
        Omega_eta = Sigma_1 = Omega_xi
        Omega_w = (self.sigma_s * np.diag(self.obs_noises)) ** 2
        return Omega_xi, Omega_w, Omega_eta, Sigma_1

    def initiate_gains(self):
        """Initiate the kalman gains with uninformative values."""
        K = np.zeros((self.T, self.N, self.M))
        return K

    def controller_backward_pass(self, Kt):
        """Do a backward pass of the controller using --Kt--."""
        R = self.R
        B = self.B
        C = self.C
        A = self.A
        D = self.D
        H = self.H
        Qt = self.Qt
        Omega_w = self.Omega_w
        Omega_xi = self.Omega_xi
        Omega_eta = self.Omega_eta
        Sxt = np.zeros((self.T, self.N, self.N))
        Set = np.zeros((self.T, self.N, self.N))
        Lt = np.zeros((self.T, self.Nc, self.N))
        st = np.zeros(self.T)
        Sxt[-1, ...] = self.Qt[-1, ...]
        Set[-1, ...] = 0
        st[-1] = 0
        time_vec = np.arange(self.T)[::-1]
        for t in time_vec[1:]:
            inv = R + mdot((B.T, Sxt[t + 1], B)) + mdot(
                (C.T, Sxt[t + 1] + Set[t + 1], C))
            outie = mdot((B.T, Sxt[t + 1], A))
            # Lt[t, ...] = np.linalg.inv(inv).dot(outie)
            Lt[t, ...] = np.linalg.solve(inv, outie)
            Sxt[t] = Qt[t] + mdot((A.T, Sxt[t + 1], A - B.dot(Lt[t]))) + \
                mdot((D.T, Kt[t].T, Set[t + 1], Kt[t], D))
            Set[t] = mdot((A.T, Sxt[t + 1], B, Lt[t])) + \
                mdot(((A - Kt[t].dot(H)).T, Set[t + 1], A - Kt[t].dot(H)))
            st[t] = np.trace(Sxt[t + 1, ...].dot(Omega_xi + Set[t + 1, ...].dot(
                Omega_xi + Omega_eta + Kt[t, ...].dot(
                    Omega_w.dot(Kt[t, ...].T))) + st[t + 1]))
        return Lt

    def estimator_forward_pass(self, Lt, xhat_zero):
        """Do a forward pass of the estimator."""
        Omega_w = self.Omega_w
        Omega_eta = self.Omega_eta
        Omega_xi = self.Omega_xi
        Sigma_1 = self.Sigma_1
        A = self.A
        B = self.B
        C = self.C
        H = self.H
        D = self.D

        Kt = np.zeros((self.T, self.N, self.M))
        Zet = np.zeros((self.T, self.N, self.N))
        Zxt = np.zeros((self.T, self.N, self.N))
        Zxet = np.zeros((self.T, self.N, self.N))
        xhat = np.zeros((self.T, self.N))

        xhat[0, ...] = xhat_zero
        Zet[0, ...] = Sigma_1
        Zxt[0, ...] = xhat_zero[:, None].dot(xhat_zero[None, :])
        Zxet[0, ...] = 0
        Zext = np.transpose(Zxet, (0, 2, 1))

        time_vec = np.arange(self.T)
        for t in time_vec[:-1]:
            reppy = A - B.dot(Lt[t])
            inny = mdot((H, Zet[t], H.T)) + Omega_w + \
                mdot((D, Zet[t] + Zxt[t] + Zxet[t] + Zext[t], D.T))
            Kt[t, ...] = mdot((A, Zet[t], H.T, np.linalg.inv(inny)))
            Zet[t + 1] = Omega_xi + Omega_eta + mdot(
                (A - Kt[t].dot(H), Zet[t], A.T)) + \
                mdot((C, Lt[t], Zxt[t], Lt[t].T, C.T))
            Zxt[t + 1] = Omega_eta + mdot((Kt[t], H, Zet[t], A.T)) + \
                mdot((reppy, Zxt[t], reppy.T)) + \
                mdot((reppy, Zxet[t], H.T, Kt[t].T)) + \
                mdot((Kt[t], H, Zext[t], reppy.T))
            Zxet[t + 1] = mdot(
                (reppy, Zxet[t], (A - Kt[t].dot(H)).T)) - Omega_eta
        return Kt

    def optimal_controler(self, xhat_zero, store=True):
        """Calculate the optimal controler and estimator.

        It iterates the controler backwards and the estimator forward.
        """
        Kt = np.zeros((self.T, self.N, self.M))
        for i in range(5):
            Lt = self.controller_backward_pass(Kt)
            Kt = self.estimator_forward_pass(Lt, xhat_zero)
        if store:
            self.Lt = Lt
            self.Kt = Kt
        return Lt, Kt

    def move(self, x_ini, Kt=None, Lt=None, key=None, dyn_kwargs=None):
        """Propagate the system using the optimal controler to make choices."""
        if dyn_kwargs is None:
            dyn_kwargs = {}
        if Lt is None or Kt is None:
            logging.info(
                'Kt or Lt not provided; calculating the optimal controler')
            Lt, Kt = self.optimal_controler(x_ini)
        xhat_t = x_ini
        x_t = x_ini
        y_t = self.H.dot(x_ini)
        all_x = np.zeros((self.T + 1, self.N))
        all_xhat = np.zeros((self.T + 1, self.N))
        all_y = np.zeros((self.T + 1, self.M))
        all_u = np.zeros((self.T + 1, self.Nc))
        u_t = -Lt[0].dot(xhat_t)
        all_u[0] = u_t
        all_x[0] = x_t
        all_y[0] = y_t
        all_xhat[0] = xhat_t
        x_t, y_t = self.dynamics(x_t, u_t, **dyn_kwargs)
        xhat_t = self.A.dot(xhat_t) + self.B.dot(u_t) + \
            Kt[0].dot(y_t - self.H.dot(xhat_t))
        for t in range(1, self.T):
            all_xhat[t] = xhat_t
            all_x[t] = x_t
            all_y[t] = y_t
            u_t = -Lt[t].dot(xhat_t)
            xhat_t = self.A.dot(xhat_t) + self.B.dot(u_t) + \
                Kt[t].dot(y_t - self.H.dot(xhat_t))
            all_u[t] = u_t
            x_t, y_t = self.dynamics(x_t, u_t, **dyn_kwargs)
        all_xhat[-1] = xhat_t
        all_x[-1] = x_t
        all_y[-1] = y_t
        return all_xhat, all_x, all_y, all_u


class SOCQian(SOC):
    """Infinite-time implementation of SOC by Qian 2013."""

    name = 'SOC + Qian'

    def __init__(self, epsilon=10, timeout=500, *args, **kwargs):
        """Do nothing, say nothing, take all the cake.

        Parameters
        ----------
        epsilon : float
        Euclidian distance to target to end simulations.

        timeout : int
        Time out (in time steps) before simulations are stopped before someone
        gets hurt.

        """
        self.epsilon = epsilon
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def define_noise_covariances(self, ):
        """Re-define Qt to remove time dependency."""
        matrices = super().define_noise_covariances()
        last_matrices = matrices[-1][0, ...]
        return *matrices[:-1], last_matrices

    def estimator_control(self, ):
        r"""Obtain the estimator and control matrices.

        This follows equations 2.1 - 2.17 in Qian 2013, with the following
        changes to the nomenclature to match the SOC class (SOC->SOCQian):
        C -> Y
        add_noise * B^2 -> G
        H -> C
        0 -> F  (state-dependent noise, zero in SOC)
        D -> 0  (state-dependent obs. noise, zero in SOCQian)
        0 -> D  (unbiased obs. noise, zero in SOC)
        Omega_xi -> G
        Additionally, hA in this function represents \hat{A} in Qian 2013.

        """
        R = self.R
        B = self.B
        C = self.C
        A = self.A
        # D = self.D
        H = self.H
        Q = self.Qt[0, ...]
        F = np.zeros_like(A)
        U = np.diag((1, 0.1, 0.01, 0, 0))  # Qian 2014
        D = np.array(self.obs_noises)  # np.ones(self.M,) * self.obs_noise

        # G = 0.1 * self.B ** 2
        G = self.Omega_xi

        R = np.eye(self.Nc) * self.r

        K = np.zeros((self.N, self.M))
        L = np.zeros((self.Nc, self.N))
        for _ in range(10):
            hA = np.kron(np.array([[0, 1], [0, 0]]), B.dot(L))\
                + block_diag(A - B.dot(L), A - K.dot(D))
            hF = np.kron(np.array([[1, 0], [1, 0]]), F)
            hY = np.kron(np.array([[-1, 1], [-1, 1]]), C.dot(L))
            hG = block_diag(G, -K.dot(D)[:, None])\
                + np.kron(np.array([[0, 0], [1, 0]]), G)
            Ws = np.kron(hA.T, hA.T) + np.kron(hF, hF).T + np.kron(hY, hY).T
            V = block_diag(Q + mdot((L.T, R, L)), mdot((L.T, R, L)) + U)\
                + np.kron(np.array([[0, -1], [-1, 0]]), -mdot((L.T, R, L)))
            vecS = -np.linalg.inv(Ws).dot(np.reshape(V, (-1, 1), order='f'))
            Wp = np.kron(hA, hA) + np.kron(hF, hF) + np.kron(hY, hY)
            vecP = - \
                np.linalg.inv(Wp).dot(np.reshape(
                    hG.dot(hG.T), (-1, 1), order='f'))
            N2 = 2 * self.N
            S = vecS.reshape((N2, N2))
            P = vecP.reshape((N2, N2))
            K = mdot((P[self.N:, self.N:], H.T)) / (D.dot(D.T))
            inny = R + \
                mdot((C.T, S[self.N:, self.N:] + S[:self.N, :self.N], C))
            L = mdot((np.linalg.inv(inny), B.T, S[self.N:, self.N:]))
            # ipdb.set_trace()
        return K, L

    def move(self, x_ini, Kt=None, Lt=None, key=None, **dyn_kwargs):
        """Propagate the system using the optimal controler to make choices.

        In contrast to SOC, the movement goes on until the system is within
        self.epsilon of the target, or self.timeout time steps have passed.

        Parameters
        ----------
        x_ini : ndarray, size=(5, )
        Initial position of the plant: (x, dx, ddx, dddx, filter, target).

        Kt : ndarray, size=(self.N, self.M)
        Kalman gain. Does not depend on time in this model.

        Lt : ndarray, size=(self.N, self.Nc)
        Control matrix. Does not depend on time in this model.

        key : unknown
        Key to my heart. Darkness took it, stored outside the reach of my
        hands, my eyes, my soul. Is it lost forever? Will I ever see it again?
        Or is the door forever shut, its hinges rusting away, beyond it a
        wilting flower, yearning for the sun. Seriously, I don't know what this
        is or why it was included in SOC.

        **dyn_kwargs are sent to self.dynamics.

        """
        if Lt is None or Kt is None:
            logging.info(
                'Kt or Lt not provided; calculating the optimal controler')
            Lt, Kt = self.optimal_controler(x_ini)
        xhat_t = x_ini
        x_t = x_ini
        y_t = self.H.dot(x_ini)
        all_x = []  # np.zeros((self.T + 1, self.N))
        all_xhat = []  # np.zeros((self.T + 1, self.N))
        all_y = []  # np.zeros((self.T + 1, self.M))
        all_u = []  # np.zeros((self.T + 1, self.Nc))
        u_t = -Lt.dot(xhat_t)
        all_u.append(u_t)
        all_x.append(x_t)
        all_y.append(y_t)
        all_xhat.append(xhat_t)
        x_t, y_t = self.dynamics(x_t, u_t, **dyn_kwargs)
        xhat_t = self.A.dot(xhat_t) + self.B.dot(u_t) + \
            Kt[0].dot(y_t - self.H.dot(xhat_t))
        for t in range(1, self.timeout):
            all_xhat.append(xhat_t)
            all_x.append(x_t)
            all_y.append(y_t)
            u_t = -Lt.dot(xhat_t)
            xhat_t = self.A.dot(xhat_t) + self.B.dot(u_t) + \
                Kt.dot(y_t - self.H.dot(xhat_t))
            all_u.append(u_t)
            x_t, y_t = self.dynamics(x_t, u_t, **dyn_kwargs)
            if np.linalg.norm(xhat_t[0] - xhat_t[-1]) < self.epsilon:
                break
        all_xhat.append(xhat_t)
        all_x.append(x_t)
        all_y.append(y_t)
        return (np.array(all_xhat), np.array(all_x), np.array(all_y),
                np.array(all_u))


class SOCSeq(SOC):
    """Sequential version of SOC.

    The model is made to go through two goals within the alloted time by
    putting via-point targets along the way.

    Two things are changed from SOC to SOCSeq:
    1. The arm model is given an extra dimension at the end which represents
       the final goal.
    2. There are now two Qs that are different from zero, one for each goal.
       It is assumed that Qs(T / 2) is where the intermediary goal is reached.

    """

    name = 'Sequential SOC'

    def __init__(self, extra_dims=1, h_scaling=None, timings=None, *args,
                 **kwargs):
        """Start the instrumentation process. Liquification will take us.

        Parameters
        ----------
        extra_dims : int
        Number of (extra) targets for sequential movements. These are in
        addition to the final target.

        h_scaling : float or ndarray
        Importance given to targets. If a float, all targets in the sequence
        are given the same importance in the cost function. If an ndarray, each
        target will have its own, and the size of h_scaling must equal
        extra_dims + 1. Note that h_scaling[n] is the importance of the n-th
        goal, regardless of the identity of the circle.

        timings : ndarray, size=(3, )
        Timings for each one of the targets in the sequence, in relative terms
        (i.e. normalized to 1). One element per target, starting with the first
        target (i.e. not pt_start). If None, they are evenly distributed in
        time.

        """
        self.extra_dims = extra_dims
        self.num_targets = extra_dims + 1
        if h_scaling is None:
            h_scaling = 1
        if np.size(h_scaling) > 1:
            self.h_scalings = h_scaling
        else:
            self.h_scalings = np.ones(extra_dims + 1) * h_scaling
        self.timings = timings
        # if h_scaling is not None:
        #     self.h_scaling = h_scaling
        # else:
        #     self.h_scaling = 1
        N = self.def_pars['N'] + extra_dims
        kwargs['N'] = N
        super().__init__(*args, **kwargs)

    def define_forward_model(self):
        """Overturn the theocracy and behead its leaders."""
        matrices = super().define_forward_model()
        A, B, C, H, D = matrices
        A = block_diag(A, np.eye(self.extra_dims))
        B = np.vstack((B, np.zeros((self.extra_dims, 1))))
        H = np.hstack((H, np.zeros((len(H), self.extra_dims))))
        C = self.sigma_c * B
        return A, B, C, H, D

    def define_cost_matrices(self):
        """Define the cost matrices Q_t and R."""
        Qt = np.zeros((self.T, self.N, self.N))
        # p = self.p_scaling * np.array([1, 0, 0, 0, 0, -1])[:, None]
        v = np.array([0, self.omega_nu, 0, 0, *[0]
                     * self.num_targets])[:, None]
        f = np.array([0, 0, self.omega_f, 0, *[0] * self.num_targets])[:, None]

        if self.timings is None:
            t_tars = np.linspace(0, self.T - 1, self.num_targets + 1,
                                 endpoint=True, dtype=int)[1:]
        else:
            t_tars = (self.t_end * self.timings / self.dt).astype(int)
        for ix_tar, t_tar in enumerate(t_tars):
            if t_tar != t_tars[-1]:
                qq = 0.1
            else:
                qq = 1
            h_scaling = self.h_scalings[ix_tar]
            pt = np.zeros(self.N)
            pt[0] = 1
            pt[4 + ix_tar] = -1
            pt = pt[:, None] * h_scaling
            Qt[t_tar, ...] = pt.dot(pt.T) + qq * v.dot(v.T) + qq * f.dot(f.T)
        # Qt[-1, ...] += v.dot(v.T) + f.dot(f.T)

        # Qt[-1, ...] = p.dot(p.T) + v.dot(v.T) + f.dot(f.T)
        # phalf = self.h_scaling * np.array([1, 0, 0, 0, -1, 0])[:, None]
        # halft = int(self.T / 2)
        # Qt[halft, ...] = phalf.dot(phalf.T) + v.dot(v.T) + f.dot(f.T)

        R = np.eye(self.Nc) * self.r
        return Qt, R


class Arm:
    """Implement the linear model of hand movements.

    The implementation follows:

    Todorov, Emanuel. “Stochastic Optimal Control and Estimation Methods
    Adapted to the Noise Characteristics of the Sensorimotor System.” Neural
    Computation 17, no. 5 (May 1, 2005):
    1084–1108. https://doi.org/10.1162/0899766053491887.

    To extend this to N-dimensional movement, N copies of the 1-dimensional
    system are used, which means that the dimensions are identical and
    independent from each other.

    """

    name = 'Linear arm model'

    def __init__(self, x_ends=None, sigma_c=0.5, tau1=0.04, tau2=0.04,
                 delta=0.01, m=1, kappa=1.5, obs_noise=0.1, num_dim=1,
                 random_seed=None, random_generator=None):
        """Initialize all parameters and matrices."""
        if random_generator is None:
            self.random = np.random.default_rng(random_seed)
        else:
            self.random = random_generator
        self.N = num_dim
        self.tau1 = tau1
        self.tau2 = tau2
        self.delta = delta
        self.m = m
        self.sigma_c = sigma_c
        self.kappa = kappa  # Control-dependent noise parameter
        A, B, C, H = self.set_matrices()
        self.A = A
        self.B = B
        self.C = C
        self.H = H
        # Couldn't find this in the Todorov 1998 paper.
        self.obs_noise = obs_noise

    def _dynamics_1d(self, x_t, u_t, with_noise=False):
        """Dynamics model for 1D.

        Returns x_{t+1} given x_t and u_t. This is
        the model described in Todorov (1998).

        Note that, as done in Todorov (1998), the control signal is a scalar,
        which is then multiplied by the vector C. This noise term is used
        only if --with_noise-- is True.

        """
        if with_noise:
            noise = self.random.normal() * self.C.dot(u_t)
        else:
            noise = 0
        return self.A.dot(x_t) + self.B.dot(u_t) + noise

    def dynamics(self, x_t, u_t, with_noise=False):
        """Dynamics model for N-dimensional movements."""
        new_x = np.zeros_like(x_t)
        for idx in range(self.N):
            new_x[idx] = self._dynamics_1d(x_t[idx, :], u_t[idx],
                                           with_noise=with_noise)
        return new_x

    def move(self, x_ini, u_t, **kwargs):
        """Make the thing GO for many trials at a time.

        Parameters
        ----------
        x_ini : ndarray
        Initial position. Size (self.N, 5).

        u_t : ndarray
        Control states for all trials. Size (num_trials, self.N).

        **kwargs are sent to self.dynamics

        Returns
        x_t : ndarray
        All visited states. Note that it contains one more element (in the
        leading dimension) than the number of motor commands.

        """
        num_trials = u_t.shape[0]
        x_t = np.zeros((num_trials + 1, *x_ini.shape))
        x_t[0, ...] = x_ini
        for trial, u_tau in enumerate(u_t):
            x_t[trial +
                1, ...] = self.dynamics(x_t[trial, ...], u_t[trial, :], **kwargs)
        return x_t

    def inv_dynamics(self, all_x):
        """Reconstruct motor commands for the given trajectory.

        Returns the necessary control signal u_t to go from x_t to x_tp1 in
        one time step, for all the steps in --all_x--. Note that, by necessity,
        it reconstructs the entire X vector for each time point; this vector is
        also returned, for shits and giggles.

        Parameters
        ----------
        all_x : ndarray
        Size = (num_trials, self.N)

        Returns
        -------
        x_t : ndarray size=(num_trials, self.N, 5)
        Reconstructed 5-dimensional states.

        u_t : ndarray size=(num_trials, self.N)
        Reconstructed motor commands.

        """
        num_trials = all_x.shape[0]
        x_t = np.zeros((num_trials, self.N, 5))
        u_t = np.zeros((num_trials, self.N))
        x_t[..., -1] = 1
        x_t[..., 0] = all_x
        x_t[:-1, :, 1] = np.diff(all_x, axis=0) / self.delta
        x_t[:-2, :, 2] = np.diff(x_t[:-1, :, 1], axis=0) / self.delta * self.m
        shifted_f = x_t[:-3, :, 2] * (1 - self.delta / self.tau2)
        unshifted_f = x_t[1:-2, :, 2]
        x_t[:-3, :, 3] = (unshifted_f - shifted_f) / self.delta * self.tau2
        shifted_g = x_t[:-4, :, 3] * (1 - self.delta / self.tau1)
        unshifted_g = x_t[1:-3, :, 3]
        u_t[:-4, ...] = (unshifted_g - shifted_g) / self.delta * self.tau1
        return x_t, u_t

    def observations(self, x_t, with_noise=False):
        """Observation function with or without noise."""
        if with_noise:
            noise = self.noise_obs * self.random.normal()
        else:
            noise = 0
        return self.H.dot(x_t) + noise

    def set_matrices(self, ):
        """Set the matrices for the dynamics model."""
        tau1 = self.tau1
        tau2 = self.tau2
        delta = self.delta
        m = self.m
        sigma_c = self.sigma_c  # 0.5
        A = np.array([[1, delta, 0, 0, 0],
                      [0, 1, delta / m, 0, 0],
                      [0, 0, 1 - delta / tau2, delta / tau2, 0],
                      [0, 0, 0, 1 - delta / tau1, 0],
                      [0, 0, 0, 0, 1]])
        B = np.array([0, 0, 0, delta / tau1, 0])
        H = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
        C = sigma_c * B
        return A, B, C, H

    def covariance(self, u_t):
        """Return the covariance at the final point X_T given u_t.

        As with everything in life, all the dimensions are handled
        independently and the results are returned as a set of covariance
        matrices, i.e. size = (self.N, *A.shape)

        """
        final_t = len(u_t)
        A = self.A
        B = self.B
        k = self.kappa
        all_covs = np.zeros((self.N, final_t, *A.shape))
        for idx in np.arange(self.N):
            cov = np.zeros((final_t, *A.shape))
            for i in np.arange(final_t):
                thing = np.linalg.matrix_power(A, final_t - i - 1).dot(B)
                cov[i, ...] = thing.dot(thing.T) * u_t[i, idx] ** 2
                all_covs[idx, i, ...] = k * np.sum(cov[:i + 1, ...], axis=0)
        return all_covs


class LinTodSimple(Arm):
    """Similar to Todorov, but the inverse dynamics are replaced to select
    only one motor command to match the prescriptions made by the physical
    attractor level in the Simple class.

    """

    name = 'Linear Todorov Simple'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A4 = np.linalg.matrix_power(self.A, 4)
        self.A3B = np.linalg.matrix_power(self.A, 3).dot(self.B)

    def inv_dynamics(self, last_x, x5_t):
        """Returns the motor command at time t such that the last (fourth)
        element of --all_x-- is reached.

        Parameters
        ----------
        last_x : ndarray[float]
        Last position (at t+4) to match, as prescribed by the attractor
        dynamics of the physical layer. Note that only the last position
        can be matched by these inverse dynamics.

        x5_t : ndarray[float]
        Position and its derivatives for the arm model. size = (5, ).

        """
        M = self.A3B[0]
        return 1 / M * (last_x - self.A4.dot(x5_t)[0])


def test_inv_dynamics():
    """Test the inv_dynamics method of the Arm class.

    Some simple data is generated such that the motor commands should be 1
    always. If inv_dynamics does not predict this, the assersion fails.

    """
    arm = Arm(sigma_c=0, tau1=1, tau2=1, delta=1, m=1, kappa=1, obs_noise=0)
    traj = np.concatenate(((0, 0, 0), np.arange(5),
                           np.arange(4)[::-1]))[:, None]
    x_t, u_t = arm.inv_dynamics(traj)

    cond_x = (x_t[..., 0] == traj).all()
    cond_u = u_t[0] == 1 and u_t[4] == -2
    message = ('The motors obtained from inv_dynamics do not match the ones '
               'in the data.')
    assert bool(cond_x and cond_u), message


def test_move():
    """Test the dynamics method of Arm.

    A set of motor commands for which the trajectory is known is sent to the
    method; if the returned trajectory matches the known one, True is
    returned. False otherwise.

    """
    arm = Arm(sigma_c=0, tau1=1, tau2=1, delta=1, m=1, kappa=1, obs_noise=0)
    u_t = np.array([1, -2, 1, 0, 0, 0])[:, None]
    x5_ini = np.array([0, 0, 0, 0, 1])[None, :]
    x_t = arm.move(x5_ini, u_t)

    expected = np.array([0, 0, 0, 1, 0, 0])
    return (expected == x_t[1:, 0, 0]).all()
