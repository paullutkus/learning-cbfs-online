import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

class Bicycle(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 V                 = 0.25,
                 umax              = 0.66,
                 control_mode      ="max",
                 disturbance_mode  ="min",
                 control_space     = None,
                 disturbance_space = None):

        self.V = V

        if control_space     is None:
            control_space     = sets.Box(jnp.array([-umax]), jnp.array([umax]))

        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([ 0   ]), jnp.array([0   ]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.array([self.V * jnp.cos(state[2]), self.V * jnp.sin(state[2]), 0])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0],
            [0],
            [1]
        ])
        '''
        return jnp.array([
            [-state[1]],
            [ state[0]],
            [ 1       ]
        ])
        '''

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0],
            [0],
            [0]
        ])

    def Df(self, state):
        return jnp.array([
            [0, 0, -state[1]],
            [0, 0,  state[0]],
            [0, 0,  0       ],
        ])
