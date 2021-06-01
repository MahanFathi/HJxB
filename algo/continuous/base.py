from algo.base import Base
from environment import Env

from yacs.config import CfgNode
from flax.core.frozen_dict import FrozenDict

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from functools import partial


class BaseContinuous(Base):

    def _get_x_train(self,
                     N: int = None,
                     ):
        """The datapoints for which the dataset is built"""
        raise NotImplementedError

    def __init__(self, cfg: CfgNode, env: Env):
        super().__init__(cfg, env)

    def _get_residual(self,
                      x_batch: jnp.ndarray,
                      params: FrozenDict,
                      ):
        """ Returns the HJB residual
            Internally finds min_u and differentiates wrt min ops.
        .input:
            x: (N, state_dim)
            params: FrozenDict
        .output:
            residual: (N, 1)
        """
        u_star_batch = self.get_optimal_u(x_batch, params)
        pjpx_batch = self.value_net.pjpx_fn(x_batch, params) # TODO(mahan): computed twice
        g_batch = self.sys.g_fn(x_batch, u_star_batch)
        f_batch = self.sys.f_fn(x_batch, u_star_batch)
        residuals = g_batch + pjpx_batch @ jnp.expand_dims(f_batch, -1)
        return residuals

    def train(self, epochs: int):
        """ Train the thing
            epochs: visitation times for the dataset
        """
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset = self._get_x_train()
            rng, input_rng = jax.random.split(rng)
            self._train_epoch(epoch, x_dataset, input_rng)

    def _train_epoch(self,
                    epoch: int,
                    x_dataset: jnp.ndarray,
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
            loss, grad = self._loss_and_grad(x_batch, self.optimizer.target)
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
            mean_cost = self.eval_policy(self.cfg.LOG.EVAL_ACROSS_N_RUNS)
            print("Env eval cost: {}".format(mean_cost))
            self.summary_writer.scalar(
                "eval_cost",
                mean_cost,
                epoch,
            )

        print("Epoch value train loss: {}".format(np.mean(loss_log)))

    # @partial(jit, static_argnums=(0,)) # TODO(mahan): self is static?
    def _loss_and_grad(self,
                       x_batch: jnp.ndarray,
                       params: FrozenDict
                       ):

        def loss_fn(params):
            residuals = self._get_residual(x_batch, params)
            loss = jnp.mean(jnp.square(residuals)) # TODO(mahan): add regularization?
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad
