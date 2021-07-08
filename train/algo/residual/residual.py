from train.algo.base_hjb import BaseHJB

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from flax.core.frozen_dict import FrozenDict

from functools import partial


class Residual(BaseHJB):

    def get_x_train(self, N=None):
        N = N or self.dataset_size
        x_dataset = self.env.sample_state(N)
        return x_dataset

    def train(self, epochs: int):
        """ Train the thing
            epochs: visitation times for the dataset
        """
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset = self.get_x_train()
            rng, input_rng = jax.random.split(rng)
            self.train_epoch(epoch, x_dataset, input_rng)

    def train_epoch(self,
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
            loss, grad = self.loss_and_grad(x_batch, self.optimizer.target)
            self.optimizer = self.optimizer.apply_gradient(grad)
            loss_log.append(loss)

        # update target network
        if epoch % self.cfg.TRAIN.UPDATE_TARGET_NET_EVERY_N_EPOCHS == 0:
            self.vparams = self.optimizer.target # update target net params

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

    @partial(jit, static_argnums=(0,)) # TODO(mahan): self is static?
    def loss_and_grad(self,
                      x_batch: jnp.ndarray,
                      params: FrozenDict,
                      ):

        def loss_fn(params: FrozenDict):
            residuals = self.get_residual(x_batch, params)
            loss = jnp.mean(jnp.square(residuals)) # TODO(mahan): add regularization?
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad

