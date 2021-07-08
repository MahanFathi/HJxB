from .usolver import USolver

from jax import numpy as jnp
from jax import jit, vmap
from flax.core.frozen_dict import FrozenDict


class QuadraticSolver(USolver):
    """ Solves for cost function quadratic in `u`
    """

    def solve(self, x_batch: jnp.ndarray, params: FrozenDict):
        """
        x_batch: (N, state_dim)
        f2: (N, state_dim, act_dim)
        pjpx: (N, 1, state_dim)
        R: (N, act_dim, act_dim)
        """

        batch_size = x_batch.shape[0]
        dummy_u = jnp.zeros((batch_size, self.act_shape[0]))
        f2 = self.sys.jac_f_fn(x_batch, dummy_u)[1]
        pjpx = self.value_net.pjpx_fn(x_batch, params)
        R = 0.5 * self.sys.hess_g_u_fn(x_batch, dummy_u)

        @jit
        @vmap
        def _u_star_solver_fn(R, f2, pjpx):
            R_inv = jnp.linalg.inv(R)
            u_star = - 0.5 * R_inv @ f2.T @ pjpx.T
            return jnp.squeeze(u_star, axis=-1)

        return _u_star_solver_fn(R, f2, pjpx)
