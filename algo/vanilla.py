from .base import BaseAlgo
from environment import Env
from network import ValueNet

import jax
from jax import jit, numpy as jnp
from yacs.config import CfgNode


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
        self.gamma = cfg.TRAIN.GAMMA

    def gather_dataset(self, ):
        x_dataset = self.env.sample_state(self.dataset_size)
        u_star_dataset = self.get_optimal_u(x_dataset)
        x_next_dataset = self.env.step1_batch_fn(x_dataset, u_star_dataset)
        j_next_dataset = self.value_net.nn.apply(self.optimizer.target, x_next_dataset)
        g_dataset = self.sys.g(x_dataset, u_star_dataset) * self.env.h # TODO(mahan) a bit ugly
        j_dataset = g_dataset + self.gamma * j_next_dataset
        return x_dataset, j_dataset
