from .base import BaseAlgo
from environment import Env

import jax
from jax import jit, vmap
from jax import jit, numpy as jnp
import numpy as np
from yacs.config import CfgNode

from functools import partial


class ResMin(BaseAlgo):
    """HJB Residual Minimization"""

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

    def get_x_train(self, N=None):
        N = N or self.dataset_size
        x_dataset = self.env.sample_state(self.dataset_size)
        return x_dataset

    def train(self, epochs: int):
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset = self.get_x_train()
            rng, input_rng = jax.random.split(rng)
            self.train_epoch(epoch, x_dataset, input_rng)

    def train_epoch(self, epoch: int, x_dataset: jnp.ndarray, rng):

        steps_per_epoch = self.dataset_size // self.batch_size
        perms = jax.random.permutation(rng, self.dataset_size)
        perms = perms[:steps_per_epoch * self.batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, self.batch_size))
        loss_log = []
        for perm in perms:  # represents a batch
            x_batch = x_dataset[perm]
            loss, grad = self.loss_and_grad(self.optimizer.target, x_batch)
            self.optimizer = self.optimizer.apply_gradient(grad)
            loss_log.append(loss)

        self.vparams = self.optimizer.target

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
    def loss_and_grad(self, params, x_batch):
        """Trains a batch
        """

        def loss_fn(params):
            pjpx = self.value_net.pjpx_fn(x_batch, params)
            u_star_batch = self.solve_u_star(x_batch, pjpx)
            @vmap
            def lazy_coding(pjpx, f):
                return pjpx @ f
            res = self.env.sys.g_fn(x_batch, u_star_batch) + lazy_coding(pjpx.squeeze(), self.env.sys.f_fn(x_batch, u_star_batch))
            return (res ** 2).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad


    def solve_u_star(self, x_batch: jnp.ndarray, pjpx: jnp.ndarray):
        """
        x_batch: (N, state_dim)
        f2: (N, state_dim, act_dim)
        pjpx: (N, 1, state_dim)
        R: (N, act_dim, act_dim)
        """

        batch_size = x_batch.shape[0]
        dummy_u = jnp.zeros((batch_size, self.env.action_space.shape[0]))
        f2 = self.sys.jac_f_fn(x_batch, dummy_u)[1]
        R = 0.5 * self.sys.hess_g_u_fn(x_batch, dummy_u)

        @jit
        @vmap
        def _u_star_solver_fn(R, f2, pjpx):
            R_inv = jnp.linalg.inv(R)
            u_star = - 0.5 * R_inv @ f2.T @ pjpx.T
            return jnp.squeeze(u_star, axis=-1)

        return _u_star_solver_fn(R, f2, pjpx)
