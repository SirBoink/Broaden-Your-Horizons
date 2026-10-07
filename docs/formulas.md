# Equations and conventions

Time is in seconds, position in metres, angles in radians, force in newtons and torque in newton-metres. Recorded commands use Δt = 0.01 s. Numerical integration may use smaller physical steps without changing command timing.

## Motor prediction and discovery

The observed discovery feature is

$$\phi_t=[x_t,y_t,v_{x,t},v_{y,t},a_{1,t},\ldots,a_{6,t}]^\top,
\qquad \widetilde\phi_{t,d}=(\phi_{t,d}-\mu_d)/\sigma_d.$$

Finite recorded successor returns and their backward recursion are

$$\psi_\gamma(t)=\sum_{k=0}^{T-1-t}\gamma^k\widetilde\phi_{t+k},
\qquad \psi_\gamma(t)=\widetilde\phi_t+\gamma\psi_\gamma(t+1).$$

With $s=-\log\gamma$, the continuous decay rate is $\lambda=s/\Delta t$ and the nominal discount horizon is $\tau=-\Delta t/\log\gamma$. Forty discounts form 39 adjacent bands:

$$B_i(t)=\frac{\psi_{\gamma_{i+1}}(t)-\psi_{\gamma_i}(t)}{|s_{i+1}-s_i|}
\approx-\partial_s\psi_s(t),\qquad
-\partial_s\psi_s(t)=\sum_k k e^{-sk}\widetilde\phi_{t+k}.$$

The grid uses geometrically spaced derivative peak times $p_i$ between 0.01 and 1.8 s. For ratio $r=p_{i+1}/p_i$,

$$\tau_i=\frac{\Delta t(r-1)}{r\log r}r^i,\quad i=0,\ldots,39,
\qquad\gamma_i=e^{-\Delta t/\tau_i}.$$

Adjacent-time directional change is

$$C_i(t)=1-\frac{B_i(t-1)^\top B_i(t)}{\|B_i(t-1)\|\|B_i(t)\|}.$$

Zero-norm pairs are invalid, not zero-change observations. For a sorted reference distribution, within-band percentiles use the midpoint of left and right insertion ranks divided by its length. Components exceeding the 90th percentile must span three consecutive bands. The representative maximizes percentile, then prefers earlier time and lower band. Changes between samples $k$ and $k+1$ belong to the latter state.

For $a=\gamma_i$, $b=\gamma_{i+1}$ and $n$ remaining samples, finite-tail missing mass is

$$m_i(n)=\frac{b^n/(1-b)-a^n/(1-a)}{1/(1-b)-1/(1-a)}.$$

Bands with $m_i(n)>0.005$ are excluded. Single-head tails use $\gamma^n$. Candidate target exclusion and separation use Euclidean distance. The executable implementations are in `formulas.py`.

## Option control and rewards

Normalized SAC actions become muscle excitation $u=(\operatorname{clip}(a,-1,1)+1)/2$. The selector acts on a 30-dimensional observation with one base action and five option actions.

For distance $d$ to the evaluated target and excess travel $w$,

$$w_t=\max(\|x_{t+1}-x_t\|-(d_t-d_{t+1}),0),$$
$$r_t=\alpha_{\rm progress}\frac{d_t-d_{t+1}}{\epsilon}
-\alpha_{\rm distance}\frac{d_{t+1}}{R}
-\alpha_{\rm path}\frac{w_t}{\epsilon}
+c_{\rm target}I_t+c_{\rm sequence}F_t.$$

Here $R$ is workspace radius, $\epsilon$ the progress scale, $I$ target acquisition and $F$ sequence completion. Weights and environment defaults are in `config/parameters.json`; historical checkpoint metadata may omit weights.

For option duration $D$, deliberation cost $c$ and terminal indicator $z$,

$$R_k=\sum_{j=0}^{D-1}\gamma^j r_{t+j}-c,
\quad o^*=\arg\max_{o\in\mathcal A(s')}Q_\theta(s',o),
\quad y=R_k+(1-z)\gamma^D Q_{\bar\theta}(s',o^*).$$

Masks exclude unavailable options. Discounted command rewards and Double DQN targets are implemented in `formulas.py`; state transitions and option termination remain in the environments.

## Learned successor features and readouts

The residual SF network receives 24 physical-state variables plus six actions. It predicts 23×10 SF values, a distinct grid from the original 40-band discovery calculation. Targets and inputs use saved normalization.

$$\mathcal L=\operatorname{MSE}(\widehat\psi,\psi^{\rm MC})
+\beta\operatorname{MSE}(\widehat\psi,\widetilde\phi+\gamma(1-z)\widehat\psi_{\rm target}(s')).$$

Loss comparisons use the same target standardization. Target-network exponential averaging is $\theta_{\rm target}\leftarrow(1-\eta)\theta_{\rm target}+\eta\theta$.

Ridge readouts forecast the same future hand displacement from current state, single-head SF or multiscale SF:

$$\widehat W=(X^\top X+\rho I)^{-1}X^\top Y,
\qquad\operatorname{MSE}=\frac1N\sum_j\|Y_j-\widehat Y_j\|^2.$$

The implementation uses calibration-only principal components and an unpenalized intercept. Ridge strength and the single head are selected on validation data. Relative error reduction is $100(E_{\rm control}-E_{\rm multi})/E_{\rm control}$.

## Biomechanics and integration

$$v=J(q)\dot q,\qquad R_{jm}=\partial\ell_m/\partial q_j,
\qquad \tau_{jm}=-R_{jm}F_m,$$
$$M(q)\ddot q+c(q,\dot q)=-R(q)F_m-D\dot q+\tau_{\rm joint}+J(q)^\top f_{\rm endpoint}.$$

Muscle force depends on activation, fibre length and velocity. Native activation follows $\dot a=(u-a)/\tau(u,a)$. MotorNet supplies its piecewise activation constants, Hill force curves, moment arms and arm inertia; exact source excerpts are in `docs/references/motornet/`. Activation alone does not specify net torque.

Historical Euler uses $s_{n+1}=s_n+h f(s_n,u)$. The finer solver evaluates full state, force and geometry at each classical RK4 stage:

$$k_1=f(s_n),\quad k_2=f(s_n+hk_1/2),\quad k_3=f(s_n+hk_2/2),\quad k_4=f(s_n+hk_3),$$
$$s_{n+1}=s_n+h(k_1+2k_2+2k_3+k_4)/6.$$

## Kinematics, human profiles and statistics

$$\kappa(t)=\frac{|\dot x\ddot y-\dot y\ddot x|}{(\dot x^2+\dot y^2)^{3/2}},
\qquad r=\frac{\sum_j(p_j-\bar p)(h_j-\bar h)}{\sqrt{\sum_j(p_j-\bar p)^2\sum_j(h_j-\bar h)^2}}.$$

The paper-specific curvature routine uses native timestamps and millimetre coordinates; inverse millimetres convert to inverse metres by multiplication by 1000. Its regional maxima use first entry, 1 mm tolerance and five-frame padding. Other geometry diagnostics explicitly use their own smoothing and speed floor; the estimands are not interchangeable.

Completion is successful trials divided by all trials. Restricted completion time assigns failures the full 8 s budget. Local benefit is chosen-branch progress minus base-branch progress at 500 ms. Progress is initial distance minus current distance, capped at acquisition.

For independent group summaries, Student intervals are $\bar x\pm t_{.975,n-1}s/\sqrt n$. Exploratory paired transfer intervals resample complete orders with replacement and take the 2.5th and 97.5th percentiles of mean differences. Endpoint measurements first average repeated invocations within episodes. Segment share counts each option invocation once and compresses uninterrupted base commands into one segment; command share counts elapsed option command steps divided by all command steps.
