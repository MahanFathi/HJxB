from algo.base import Base
from environment import Env

from yacs.config import CfgNode

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from functools import partial


class BaseContinuous(Base):
    def __init__(self, cfg: CfgNode, env: Env):
        super().__init__(cfg, env)

    def _get_residual(self, params, x_batch):
        """ Returns the HJB residual
            Internally finds min_u and differentiates wrt min ops.
        .input:
            x: (N, state_dim)
        .output:
            residual: (N, 1)
        """
        u_star_batch = self.get_optimal_u(x_batch)
        x_next_batch = self.env.step1_batch_fn(x_batch, u_star_batch)
        j_next_batch = self.value_net.apply(self.optimizer.target, x_next_batch)
        g_batch = self.sys.g_fn(x_batch, u_star_batch) * self.env.h # TODO(mahan) a bit ugly
        j_batch = g_batch + self.gamma * j_next_batch
        return j_batch

    def get_x_train(self, N=None):
        """The datapoints for which the dataset is built"""
        raise NotImplementedError

    def train(self, epochs: int):
        """ Train the thing
            epochs: visitation times for the dataset
        """
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset = self.get_x_train()
            j_dataset = self.get_target_j(x_dataset)
            rng, input_rng = jax.random.split(rng)
            self._train_epoch(epoch, x_dataset, j_dataset, input_rng)

    def _train_epoch(self,
                    epoch: int,
                    x_dataset: jnp.ndarray,
                    j_dataset: jnp.ndarray,
                    rng,
                    ):
        """Trains an epoch
        """
        steps_per_epoch = self.dataset_size // self.batch_size
        perms = jax.random.permutation(rng, self.dataset_size)
        perms = perms[:steps_per_epoch * self.batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, self.batch_size))
        loss_log = []
        for perm in perms:  # represents a batch
            x_batch = x_dataset[perm]
            j_batch = j_dataset[perm]
            loss, grad = self._loss_and_grad(x_batch, j_batch, self.optimizer.target)
            self.optimizer = self.optimizer.apply_gradient(grad)
            loss_log.append(loss)

        # report to tb
        if epoch % self.cfg.LOG.LOG_EVERY_N_EPOCHS == 0:
            self.summary_writer.scalar(
                "train_loss",
                np.mean(loss_log),
                epoch,
            )

        # evaluate policy
        if epoch % self.cfg.LOG.EVAL_EVERY_N_EPOCHS == 0:
            mean_cost = self.eval_policy(10)
            print("Env eval cost: {}".format(mean_cost))
            self.summary_writer.scalar(
                "eval_cost",
                mean_cost,
                epoch,
            )

        print("Epoch value train loss: {}".format(np.mean(loss_log)))

    # @partial(jit, static_argnums=(0,)) # TODO(mahan): self is static?
    def _loss_and_grad(self, x_batch, params):

        def loss_fn(params):
            residuals = self._get_residual(params, x_batch)
            loss = jnp.mean(jnp.square(residuals))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad
