import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

class Planar(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 s                 = -1,
                 delta             = 1.,
                 umax              = 1.,
                 control_type      ="ball",
                 control_mode      ="max",
                 disturbance_mode  ="min",
                 control_space     = None,
                 disturbance_space = None):

        self.s     = s
        self.delta = delta

        if   (control_space is None) and (control_type == "ball"):
            print("ball constraints with umax", umax)
            control_space = sets.Ball(jnp.array([0, 0]), umax)

        elif (control_space is None) and (control_type == "box"):
            print("box constraints with umax", umax)
            control_space = sets.Box(jnp.array([-umax, -umax]), jnp.array([umax, umax]))

        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0, 0]), jnp.array([0, 0]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return self.s * state

    def control_jacobian(self, state, time):
        return jnp.diag(jnp.array([state[0]**2 + self.delta, state[1]**2 + self.delta]))

    def disturbance_jacobian(self, state, time):
        return jnp.diag(jnp.array([0, 0]))

    def Df(self, state):
        return jnp.diag(jnp.array([self.s, self.s]))
