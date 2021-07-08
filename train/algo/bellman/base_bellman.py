from train.algo.base_hjb import BaseHJB

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from functools import partial


class BaseBellman(BaseHJB):

    def get_x_train(self, N=None):
        """The datapoints for which the dataset is built"""
        raise NotImplementedError

    def train(self, epochs: int):
        """Train the thing
        epochs: visitation times for the dataset
        """
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset = self.get_x_train()
            j_dataset = self.get_target_j(x_dataset, self.vparams)
            rng, input_rng = jax.random.split(rng)
            self.train_epoch(epoch, x_dataset, j_dataset, input_rng)

    def train_epoch(self,
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
            loss, grad = self.loss_and_grad(x_batch, j_batch, self.optimizer.target)
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
            if self.cfg.LOG.ANIMATE:
                self.animate(epoch)

        # save weights
        if epoch % self.cfg.LOG.SAVE_WEIGHTS_EVERY_N_EPOCHS == 0:
            self.save_params(str(epoch))

        print("Epoch {}, J train loss: {}".format(epoch, np.mean(loss_log)))

    @partial(jit, static_argnums=(0,))
    def loss_and_grad(self, x_batch, j_batch, params):
        """Trains a batch
        """

        def loss_fn(params):
            predictions = self.value_net.apply(params, x_batch)
            loss = jnp.mean(jnp.square(predictions - j_batch))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad
