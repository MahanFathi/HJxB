from .usolver import USolver
from environment import Env

from yacs.config import CfgNode

from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax import jit, vmap


class ConvConjSolver(USolver):
    """ Generally solves for g(x, u) = l_1(x) + l_2(u)
    """

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

    def _make_solver(self, ):
        self._grad_l2 = self.env.sys.grad_l2_fn

        def u_star_solver_fn(f2, pjpx):
            """
            f2: (N, state_dim, act_dim)
            pjpx: (N, 1, state_dim)
            """
            @jit
            @vmap
            def minimizer(f2, pjpx):
                """
                f2: (state_dim, act_dim)
                pjpx: (1, state_dim)
                """
                min_f = lambda u: jnp.mean(jnp.square(
                    self._grad_l2(jnp.zeros(self.obs_shape), u) +
                    jnp.squeeze(pjpx @ f2, axis=0)
                ))
                results = minimize(
                    min_f,
                    jnp.zeros(self.act_shape),
                    method="BFGS",
                )
                return results.x

            return minimizer(f2, pjpx)

        return u_star_solver_fn
