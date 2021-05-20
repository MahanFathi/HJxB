from .base import BaseAlgo
from environment import Env
from network import ValueNet

import jax
from jax import jit, numpy as jnp
from yacs.config import CfgNode

from functools import partial


class VanillaCFVI(BaseAlgo):
    """Continuous Fitted Value Iteration"""

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 value_net: ValueNet,
                 ):
        super().__init__(cfg, env, value_net)
        self.dataset_size = cfg.TRAIN.DATASET_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def gather_dataset(self, ):
        return (jnp.ones((self.dataset_size, self.env.observation_space.shape[0])), jnp.ones((self.dataset_size, 1)))

