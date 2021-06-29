from environment import Env
from yacs.config import CfgNode

from jax import numpy as jnp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from algo import BaseAlgo


class USolver(object):

    def __init__(self, algo: 'BaseAlgo'):
        self.algo = algo
        self.cfg = self.algo.cfg
        self.env = self.algo.env
        self.sys = self.algo.sys
        self.value_net = self.algo.value_net
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

    def solve(self, x_batch: jnp.ndarray):
        """Returns the optimal action wrt J* at hand
        .input:
        x: (N, state_dim)
        .output:
        u_star: (N, act_dim)
        """
        raise NotImplementedError
