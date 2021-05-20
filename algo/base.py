from environment import Env
from network import ValueNet
from utils import logger

from flax import linen as nn, optim
from yacs.config import CfgNode

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from functools import partial


class BaseAlgo(object):
    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 value_net: ValueNet,
                 ):
        self.cfg = cfg
        self.env = env
        self.sys = env.sys
        self.value_net = value_net
        self._init_value_net_and_optimizer()
        self.u_star_solver_fn = self._make_u_star_solver()
        self.summary_writer = logger.get_summary_writer(cfg)

    def _init_value_net_and_optimizer(self, ):
        sample_x = self.env.sample_state(1)
        self.vparams = self.value_net.nn.init(self.env.PRNGkey, sample_x)
        self.optimizer = optim.Adam(learning_rate=self.cfg.VALUE_NET.LR).create(self.vparams)

    def _make_u_star_solver(self, ):
        @jit
        @vmap
        def u_star_solver_fn(R, f2, pjpx):
            R_inv = jnp.linalg.inv(R)
            u_star = - 0.5 * R_inv @ f2.T @ pjpx.T
            return jnp.squeeze(u_star, axis=-1)

        return u_star_solver_fn

    def get_optimal_u(self,
                      x_batch: jnp.ndarray,  # (N, obs_dim)
                      ):
        """Returns the optimal action wrt J* at hand
        .input:
            x: (N, state_dim)
        .output:
            u_star: (N, act_dim)
        """
        # TODO(mahan): get rid of the need for dummy_u
        dummy_u = jnp.zeros((x_batch.shape[0], self.env.action_space.shape[0]))
        R = 0.5 * self.sys.hess_g_u(x_batch, dummy_u)
        f2 = self.sys.jac_f(x_batch, dummy_u)[1]
        pjpx = self.value_net.pjpx_fn(x_batch, self.vparams)
        u_star = self.u_star_solver_fn(R, f2, pjpx)
        return u_star

    def train(self, epochs: int):
        """Train the thing
        epochs: visitation times for the dataset
        """
        rng = jax.random.PRNGKey(0)
        for epoch in range(epochs):
            x_dataset, j_dataset = self.gather_dataset()
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
        self.summary_writer.scalar("train_loss", np.mean(loss_log), epoch)
        print(loss)

    @partial(jit, static_argnums=(0,))
    def loss_and_grad(self, x_batch, j_batch, params):
        """Trains a batch
        """

        def loss_fn(params):
            predictions = self.value_net.nn.apply(params, x_batch)
            loss = jnp.mean(jnp.square(predictions - j_batch))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(params)
        return loss, grad

