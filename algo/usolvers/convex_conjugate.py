import algo
from .usolver import USolver

from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax import jit, vmap


class ConvConjSolver(USolver):
    """ Generally solves for g(x, u) = l_1(x) + l_2(u)
    """

    def solve(self, x_batch: jnp.ndarray):
        """
        x_batch: (N, state_dim)
        f2: (N, state_dim, act_dim)
        pjpx: (N, 1, state_dim)
        """

        batch_size = x_batch.shape[0]
        dummy_u = jnp.zeros((batch_size, self.act_shape[0]))
        f2 = self.sys.jac_f_fn(x_batch, dummy_u)[1] # assuming affine
        pjpx = self.value_net.pjpx_fn(x_batch, self.algo.vparams)

        @jit
        @vmap
        def minimizer(f2, pjpx):
            """
               f2: (state_dim, act_dim)
               pjpx: (1, state_dim)
               """
            min_f = lambda u: jnp.mean(jnp.square(
                self.sys.grad_l2_fn(jnp.zeros(self.obs_shape), u) +
                jnp.squeeze(pjpx @ f2, axis=0)
            ))
            results = minimize(
                min_f,
                jnp.zeros(self.act_shape),
                method="BFGS",
            )
            return results.x

        return minimizer(f2, pjpx)
