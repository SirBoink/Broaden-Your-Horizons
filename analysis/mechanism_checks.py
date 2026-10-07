"""Short scientific/API regression checks. No training or artifact mutation."""
import json
import numpy as np
import formulas as math


def run_checks(legacy_root=None):
    from . import mechanism_experiment as experiment
    from .mechanism_physics import RK4Arm26
    # y'=y has an independent analytic solution. Incorrect accumulated RK stages
    # do not attain fourth-order accuracy, even when their outputs stay finite.
    class Exponential:
        _rungekutta4 = RK4Arm26._rungekutta4

        def __init__(self, dt):
            self.minidt, self.half_minidt = dt, dt/2
            self.states = {'value': 1.}

        def ode(self, action, states, endpoint_load, joint_load):
            return states.copy()

        def integration_step(self, dt, derivative, states):
            return {'value': states['value']+dt*derivative['value']}

        def _set_state(self, states):
            self.states = states

    errors = []
    for dt in (.2, .1):
        solver = Exponential(dt)
        solver._rungekutta4(None, None, None)
        errors.append(abs(solver.states['value']-np.exp(dt)))
    assert errors[1] < 1e-7 and errors[0]/errors[1] > 25
    gamma = np.array([.5, .8, .9])
    phi = np.arange(120, dtype=float).reshape(40, 3) / 100
    result = math.returns(phi, gamma)
    np.testing.assert_allclose(result[:, :-1], phi[:-1] + gamma[:, None, None]*result[:, 1:])
    np.testing.assert_allclose(result[:, -1], np.broadcast_to(phi[-1], (3, 3)))
    constant = math.returns(np.ones((40, 1)), gamma)[:, 0, 0]
    np.testing.assert_allclose(constant, (1-gamma**40)/(1-gamma))
    remaining = np.array([1, 20, 40])
    explicit = np.array([[(sum(b**k-a**k for k in range(n, 1000))) /
                         (1/(1-b)-1/(1-a)) for n in remaining]
                         for a, b in zip(gamma[:-1], gamma[1:])])
    np.testing.assert_allclose(math.missing_mass(gamma, remaining), explicit)
    assert np.all(np.diff(math.discount_grid()) > 0)
    assert np.isnan(math.cosine_change(np.zeros((2, 3, 4)))).all()
    # The convention attributes a change between k and k+1 to the latter state.
    percentile = np.zeros((6, 10)); percentile[1:4, 2] = .99
    events = math.events(percentile, np.ones((11, 2)), 10, np.zeros((1, 2)), 0, 0)
    assert len(events) == 1 and events[0]['step'] == 3
    # Independently reset rows must match the same rows integrated in a batch.
    starts = [[45., 90.], [46., 89.]]
    batched, obs = experiment.make_task((0, 3), postures=starts)
    assert obs.shape == (2, 30) and experiment.features(batched).shape == (2, 10)
    command = np.array([[.12, .22, .18, .32, .21, .14], [.32, .13, .19, .12, .23, .18]], dtype=np.float32)
    singles = [experiment.make_task((0, 3), postures=[p])[0] for p in starts]
    for _ in range(3):
        batched.step(command, deterministic=True)
        for row, single in enumerate(singles):
            single.step(command[row:row+1], deterministic=True)
    np.testing.assert_allclose(experiment.features(batched), np.concatenate([experiment.features(s) for s in singles]), atol=2e-6)
    assert np.isclose(batched.elapsed, .03) and np.isclose(batched.effector.dt, .01)
    assert np.isclose(batched.effector.minidt, .001)
    assert isinstance(batched.effector, RK4Arm26)
    import torch
    assert batched.states['joint'].dtype == torch.float64
    # Constant joint acceleration has q(t)=q(0)+v(0)t+a*t^2/2.
    # This exercises the real joint integration; using MotorNet's Euler position
    # update inside otherwise correct RK4 stages fails this analytic check.
    from unittest.mock import patch
    accelerated = experiment.make_task((0,))[0].effector
    initial = accelerated.states['joint'].clone()
    acceleration = torch.tensor([[.2, -.3]], dtype=torch.float64)
    def constant_acceleration(action, states, endpoint_load, joint_load):
        return dict(joint=torch.cat((states['joint'][:, 2:], acceleration), dim=1),
                    muscle=torch.zeros_like(states['muscle'][:, :1]))
    with patch.object(accelerated, 'ode', side_effect=constant_acceleration):
        accelerated.step(np.zeros((1, 6)))
    expected = initial.clone()
    expected[:, :2] += initial[:, 2:]*.01 + .5*acceleration*.01**2
    expected[:, 2:] += acceleration*.01
    np.testing.assert_allclose(experiment.array(accelerated.states['joint']), experiment.array(expected), atol=1e-12, rtol=0)
    muscle = batched.states['muscle']
    force_state = batched.effector.muscle.integrate(0., torch.zeros_like(muscle[:, :1]), muscle, batched.states['geometry'])
    np.testing.assert_allclose(experiment.array(muscle), experiment.array(force_state), atol=1e-12, rtol=0)
    # Replay restores all state and delayed observations, not just hand position.
    policies = experiment.models()
    task, obs = experiment.make_task((0, 3))
    data = experiment.record(task, obs, 'base', policies, (15., 1.5), 4)
    row = {key:(value[:, 0] if key not in ('sequence','time_s','first_hit') else
                value[0] if key == 'first_hit' else value) for key,value in data.items()}
    row['metadata'] = np.asarray(json.dumps({'controller':'base'}))
    restored, obs = experiment.restore(row, 2, .001)
    replay = experiment.record(restored, obs, 'base', policies, (15., 1.5), 1)
    for key in ('phi', 'observation', 'goal', 'state_joint', 'state_muscle'):
        np.testing.assert_allclose(replay[key][1, 0], row[key][3], atol=1e-6)
    assert np.isfinite(experiment.muscle_forces(restored.effector, np.ones((1, 6))*.3).detach().numpy()).all()
    # The neural encoder has its own physical-state ordering, not the 30D policy observation.
    legacy_root=experiment.LEGACY if legacy_root is None else legacy_root
    legacy=experiment.load_episode(legacy_root/'data/neural/episodes/episode_00000.npz')
    encoded,_=experiment.make_task(physical_dt=.01,postures=[legacy['initial_posture_deg']])
    np.testing.assert_allclose(experiment.physical_state(encoded)[0],legacy['state'][0],atol=2e-6)
    np.testing.assert_allclose(experiment.features(encoded)[0],legacy['feature'][0],atol=2e-6)
    allocation = experiment.pd_command(restored, (15., 1.5))
    assert allocation.shape == (1, 6) and np.all((allocation >= .001) & (allocation <= 1))
    probes = experiment.pulse_probe(restored, experiment.array(restored.goal)[0])
    assert [v['horizon_s'] for v in probes] == [.05, .15, .3]
    assert all(np.isfinite(v['response_matrix']).all() for v in probes)
    import copy
    repeated=copy.deepcopy(restored.effector)
    repeated.states={key:value.repeat((2,)+(1,)*(value.ndim-1)) for key,value in repeated.states.items()}
    batched_probes=experiment.pulse_batch(repeated,np.repeat(experiment.array(restored.goal),2,axis=0))
    for probe_row in batched_probes:
        for a,b in zip(probes,probe_row):np.testing.assert_allclose(a['response_matrix'],b['response_matrix'],atol=2e-5)
    solo=experiment.branch(row,2,.001,policies,(15.,1.5),measure_probes=False)
    paired=experiment.branch_many(row,2,.001,policies,(15.,1.5),[('base',None,False),('duplicate',None,False)],measure_probes=False)
    for result in paired:
        assert result['acquired']==solo['acquired'] and result['intervention_steps']==0
        np.testing.assert_allclose(result['progress_500ms_m'],solo['progress_500ms_m'],atol=2e-6)
    return {'rk4_analytic_accuracy':True, 'rk4_joint_acceleration':True, 'synchronized_force_geometry':True,
            'bellman_and_terminal':True, 'discount_tail_mass':True, 'event_index':True,
            'batch_equivalence':True, 'command_clock':True, 'snapshot_replay':True,
            'public_force_api':True, 'feedback_allocation':True, 'physical_pulses':True,
            'legacy_neural_state_encoding':True, 'batched_probes':True, 'paired_branch_equivalence':True}


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2))
