from .usolver import USolver

from jax import numpy as jnp
from jax import jit, vmap
from flax.core.frozen_dict import FrozenDict


class DiscreteSolver(USolver):
    """ Solves for optimal u, via discrete bellman eq.
    """

    def __init__(self, algo):
        super().__init__(algo)
        self._prepare_actions()

    def _prepare_actions(self, ):
        grid_points = self.cfg.USOLVER.DISCRETE.GRID_POINTS

        if type(grid_points) is list:
            assert len(grid_points) == self.act_shape[0], "incorrect len of grid_points"
        elif type(grid_points) is int:
            grid_points = [grid_points] * self.act_shape[0]
        else:
            raise TypeError("grid_points type is not valid")

        ranges = []
        for i, gp in enumerate(grid_points):
            ranges.append(
                jnp.linspace(
                    self.env.action_space.low[i],
                    self.env.action_space.high[i],
                    int(gp) // 2 * 2 + 1,
                )
            )
        self.grid = jnp.array(jnp.meshgrid(*ranges, indexing='ij', sparse=False))
        self.grid = jnp.moveaxis(self.grid, 0, -1)
        self.actions = self.grid.reshape(-1, self.act_shape[0])

    def solve(self, x_batch: jnp.ndarray, params: FrozenDict):
        """
        x_batch: (N, state_dim)
        f2: (N, state_dim, act_dim)
        pjpx: (N, 1, state_dim)
        R: (N, act_dim, act_dim)
        """

        gamma = self.cfg.TRAIN.GAMMA
        N = self.actions.shape[0]

        @jit
        @vmap
        def _u_star_solver_fn(x: jnp.array):
            """
               .input:
               x: (obs_dim, )
               .output:
               u_star: (act_dim, )
               """

            x_repeated = jnp.repeat(x[None, :], N, axis=0)
            x_next = self.sys.f_fn(x_repeated, self.actions)
            j_next = self.value_net.apply(self.algo.vparams, x_next)
            g = self.sys.g_fn(x_repeated, self.actions) * self.env.h
            j = g[:, None] + gamma * j_next
            min_index = jnp.argmin(j, axis=0)
            return self.actions[min_index[0]]

        return _u_star_solver_fn(x_batch)
