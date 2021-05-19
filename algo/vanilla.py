"""Vanilla HJB Update"""
from .base import BaseAlgo

import jax

def VanillaCFVI(BaseAlgo):
    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 value_net: ValueNet,
                 ):
        super().__init__(cfg, env, value_net)
        self.dataset_size = cfg.TRAIN.DATASET_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def train(self, epochs: int):
        """Train the thing
        epochs: visitation times for the dataset
        """
        for epoch in range(epochs):
            x_dataset, j_dataset = self.gather_dataset()
            rng, input_rng = jax.random.split(rng)
            self.train_epoch(x_dataset, j_dataset, input_rng)

    def gather_dataset(self, ):

    def train_epoch(self,
                    x_dataset: jnp.ndarray,
                    j_dataset: jnp.ndarray,
                    rng,
                    ):

    def train_step(self, ):
