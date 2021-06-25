from .usolver import USolver
from environment import Env

from yacs.config import CfgNode

from jax import numpy as jnp
from jax import jit, vmap


class QuadraticSolver(USolver):
    """ Solves for cost function quadratic in `u`
    """

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

    def _make_solver(self, ):

        def u_star_solver_fn(f2, pjpx):
            """
            f2: (N, state_dim, act_dim)
            pjpx: (N, 1, state_dim)
            """
            batch_size = f2.shape[0]
            dummy_x = jnp.zeros([batch_size, self.obs_shape[0]])
            dummy_u = jnp.zeros([batch_size, self.act_shape[0]])
            R = 0.5 * self.env.sys.hess_g_u_fn(dummy_x, dummy_u)

            @jit
            @vmap
            def _u_star_solver_fn(R, f2, pjpx):
                R_inv = jnp.linalg.inv(R)
                u_star = - 0.5 * R_inv @ f2.T @ pjpx.T
                return jnp.squeeze(u_star, axis=-1)

            return _u_star_solver_fn(R, f2, pjpx)
        return u_star_solver_fn

