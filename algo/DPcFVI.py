from .base import BaseAlgo
from environment import Env

import jax
from jax import jit, numpy as jnp
from yacs.config import CfgNode


class DPcFVI(BaseAlgo):
    """Continuous Fitted Value Iteration"""

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

    def get_x_train(self, N=None):
        N = N or self.dataset_size
        x_dataset = self.env.sample_state(self.dataset_size)
        return x_dataset
