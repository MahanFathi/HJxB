from .base import BaseContinuous
from environment import Env

import jax
from jax import jit, numpy as jnp
from yacs.config import CfgNode


class DDcFV(BaseContinuous):
    """Data-driven Continuous Fitted Value"""

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

    def _get_x_train(self, N=None):
        N = N or self.dataset_size
        x_dataset = self.env.sample_state(self.dataset_size)
        return x_dataset
