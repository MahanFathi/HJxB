from environment.util import IntegrationOrder, integrate

from typing import (
    Callable,
    Tuple,
    Optional,
    Union,
    Sequence,
)

import jax
from jax import jit, vmap, numpy as jnp
import gym

class System(object):
    """Base Class for Continuous Control Systems.
    """
    def __init__(self, ):
        """Make auxiliary diff functions
        These fns are vmap'd, i.e. they take in batch.

        f_fn(x: (N, obs_dim), u: (N, act_dim)) -> (N, obs_dim)
        g_fn(x: (N, obs_dim), u: (N, act_dim)) -> (N, )
        jac_f_fn(x: (N, obs_dim), u: (N, act_dim)) -> (N, obs_dim, obs_dim), (N, obs_dim, act_dim)
        grad_g_fn(x: (N, obs_dim), u: (N, act_dim)) -> (N, obs_dim), (N, act_dim)
        hess_g_u_fn(x: (N, obs_dim), u: (N, act_dim)) -> (N, act_dim, act_dim)
        """
        self.f_fn = jit(vmap(self.f))
        self.g_fn = jit(vmap(self.g))
        self.jac_f_fn = self._make_jac_f_fn()
        self.grad_g_fn = self._make_grad_g_fn()
        self.hess_g_u_fn = self._make_hess_g_u_fun()

    def f(self,
          x: jnp.array, # could be a batch of states and actions
          u: jnp.ndarray,
          ) -> jnp.ndarray:
        """Dynamics of the System
        .. math::
            \dot{x} = f(x, u)
        .input:
            x: (state_dim, )
            u: (act_dim, )
        .output:
            x_dot: (state_dim, )
        """
        raise NotImplementedError

    def g(self,
          x: jnp.array,
          u: jnp.ndarray,
          ) -> jnp.ndarray:
        """Continuous Cost of the System
        .input:
            x: (state_dim, )
            u: (act_dim, )
        .output:
            cost: ()
        """
        raise NotImplementedError

    def _make_jac_f_fn(self, ):
        single_jac_f_fn = jax.jacfwd(self.f, argnums=[0, 1])
        batch_jac_f_fn = jit(vmap(single_jac_f_fn))
        return batch_jac_f_fn

    def _make_grad_g_fn(self, ):
        single_grad_g_fn = jax.grad(self.g, argnums=[0, 1])
        batch_grad_g_fn = jit(vmap(single_grad_g_fn))
        return batch_grad_g_fn

    def _make_hess_g_u_fun(self, ):
        single_grad_g_u_fn = jax.grad(self.g, argnums=1)
        single_hess_g_u_fn = jax.jacfwd(single_grad_g_u_fn, argnums=1)
        batch_hess_g_u_fn = jit(vmap(single_hess_g_u_fn))
        return batch_hess_g_u_fn


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
        self.T = None # (float: the horizon)
        self.integration_order = integration_order
        self.seed()
        self.step1_batch_fn = self._make_step1_batch_fn()

    @property
    def timesteps(self, ):
        return int(self.T // self.h)

    def step1(self,
             x: jnp.array,
             u: Union[float, jnp.ndarray],
             ) -> jnp.ndarray:
        """Take 1 step
        """
        return self.stepn(x, [u] , 1)[0]

    def _make_step1_batch_fn(self, ):
        return jit(vmap(self.step1))

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

    def seed(self, seed=None):
        self.PRNGkey = jax.random.PRNGKey(0)

    def sample_state(self,
                     size: int, # sample_size
                     ):
        self.PRNGkey, subkey = jax.random.split(self.PRNGkey)
        state_dim = self.observation_space.shape[0]
        sample_state = jax.random.uniform(
            subkey,
            (size, state_dim),
            minval=-self.observation_space.high,
            maxval= self.observation_space.high,
        )
        return sample_state

    # feel free to override this
    def reset(self):
        self.state = jnp.squeeze(self.sample_state(1))
        return self.state
