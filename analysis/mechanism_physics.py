"""Classical RK4 for the existing rigid-tendon MotorNet plant.

Only joint state and activation are dynamical variables. Geometry and force
must be evaluated at the same stage state, rather than one stage behind it.
Historical 10 ms Euler rollouts continue to use the unmodified MotorNet class.
"""
import motornet as mn
import torch


class RK4Arm26(mn.effector.RigidTendonArm26):
    def __init__(self, **kwargs):
        super().__init__(integration_method='rk4', **kwargs)
        self.double()

    def synchronize(self):
        """Rebuild algebraic quantities after restoring joint state/activation."""
        joint = self.states['joint'].double()
        geometry = self.get_geometry(joint)
        activation = self.states['muscle'][:, :1].double()
        muscle = self.muscle.integrate(0., torch.zeros_like(activation), activation, geometry)
        self._set_state(dict(joint=joint, muscle=muscle, geometry=geometry))

    def integration_step(self, dt, state_derivative, states):
        position, velocity = (states['joint'] + dt*state_derivative['joint']).chunk(2, dim=1)
        velocity = self.skeleton.clip_velocity(position, velocity)
        joint = torch.cat((self.skeleton.clip_position(position), velocity), dim=1)
        geometry = self.get_geometry(joint)
        muscle = self.muscle.integrate(dt, state_derivative['muscle'], states['muscle'], geometry)
        return dict(joint=joint, muscle=muscle, geometry=geometry)

    def ode(self, action, states, endpoint_load, joint_load):
        derivative = super().ode(action, states, endpoint_load, joint_load)
        # MotorNet supplies acceleration only; RK4 also needs dq/dt = velocity.
        derivative['joint'] = torch.cat((states['joint'][:, self.skeleton.dof:], derivative['joint']), dim=1)
        return derivative

    def _rungekutta4(self, action, endpoint_load, joint_load):
        initial = self.states
        k1 = self.ode(action, initial, endpoint_load, joint_load)
        stage = self.integration_step(self.half_minidt, k1, initial)
        k2 = self.ode(action, stage, endpoint_load, joint_load)
        stage = self.integration_step(self.half_minidt, k2, initial)
        k3 = self.ode(action, stage, endpoint_load, joint_load)
        stage = self.integration_step(self.minidt, k3, initial)
        k4 = self.ode(action, stage, endpoint_load, joint_load)
        derivative = {key: (k1[key] + 2*k2[key] + 2*k3[key] + k4[key])/6 for key in k1}
        self._set_state(self.integration_step(self.minidt, derivative, initial))
