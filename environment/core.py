from environment.util import IntegrationOrder, integrate

from typing import (
    Callable,
    Tuple,
    Optional,
    Union,
    Sequence,
)

from jax import numpy as jnp
import gym

class System(object):
    """Base Class for Continuous Control Systems.
    """
    def f(self,
          x: jnp.array,
          u: Union[float, jnp.ndarray],
          ) -> jnp.ndarray:
        """Dynamics of the System
        .. math::
            \dot{x} = f(x, u)
        """
        raise NotImplementedError

    def g(self,
          x: jnp.array,
          u: Union[float, jnp.ndarray],
          ) -> jnp.ndarray:
        """Continuous Cost of the System
        """
        raise NotImplementedError


class Env(gym.Env):
    """Base Class for Continuous Control Envs (makes sys stateful).
    """
    def __init__(self,
                 sys: System,
                 h: float = 0.001,  # timestep length
                 integration_order: IntegrationOrder = IntegrationOrder.LINEAR,
                 ):
        self.sys = sys
        self.h = h
        self.integration_order = integration_order

    def step1(self,
             x: jnp.array,
             u: Union[float, jnp.ndarray],
             ) -> jnp.ndarray:
        """Take 1 step
        """
        return self.stepn(x, [u] , 1)[0]

    def stepn(self,
             x: jnp.array,
             us: Sequence[Union[float, jnp.ndarray]],
             n: int,
             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Take n steps
            returns (final, intermediate) states
        """
        return integrate(self.sys.f, x, jnp.asarray(us), self.h, n, None, self.integration_order)

    def step_cost(self,
             x: jnp.array,
             u: Union[float, jnp.ndarray],
             ):
        """Integrate cost (g), to get true accumulative cost
        """
        return self.h * self.sys.g(x, u)

    def step(self,
             u: Union[float, jnp.ndarray],
             ):
        """Replace gym's step func
        """
        self.state = self.step1(self.state, u)
        cost = self.step_cost(self.state, u)
        return self.state, -cost, False, {} # TODO(mahan): assuming no termination/info
