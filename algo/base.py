from environment import Env
from network import ValueNet
from utils import logger, gif
from .usolvers import make_usolver

from flax import linen as nn, optim, serialization
from yacs.config import CfgNode

import jax
from jax import numpy as jnp, jit, vmap
import numpy as np

from functools import partial
import experiment_buddy


class BaseAlgo(object):
    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        self.cfg = cfg
        self.env = env
        self.sys = env.sys
        self._dump_train_config()
        self._init_value_net_and_optimizer()
        self.summary_writer = experiment_buddy.deploy(host=cfg.HOST)
        # logger.get_summary_writer(cfg)

        self.dataset_size = cfg.TRAIN.DATASET_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.gamma = cfg.TRAIN.GAMMA

        self.usolver = make_usolver(cfg, self)

    def _dump_train_config(self, ):
        logdir = logger.get_logdir_path(self.cfg)
        with open(logdir.joinpath("config.yaml"), "w") as f:
            f.write(self.cfg.dump())
        f.close()

    def _init_value_net_and_optimizer(self, ):
        self.value_net = ValueNet(self.cfg, self.env.obs2feat)
        dummy_x = self.env.sample_state(1)
        dummy_features = self.env.obs2feat(dummy_x)
        self.vparams = self.value_net.nn.init(jax.random.PRNGKey(666), dummy_features)
        self.optimizer = optim.GradientDescent(learning_rate=self.cfg.VALUE_NET.LR).create(self.vparams)

    def get_target_j(self,
                     x_batch: jnp.ndarray,
                     ):
        """Returns the target J*
        .input:
            x: (N, state_dim)
        .output:
            j: (N, 1)
        """
        u_star_batch = self.usolver.solve(x_batch)
        x_next_batch = self.env.step1_batch_fn(x_batch, u_star_batch)
        j_next_batch = self.value_net.apply(self.vparams, x_next_batch)
        g_batch = self.sys.g_fn(x_batch, u_star_batch) * self.env.h # TODO(mahan) a bit ugly
        j_batch = g_batch[:, None] + self.gamma * j_next_batch
        return j_batch

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
            j_dataset = self.get_target_j(x_dataset)
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
            self.summary_writer.add_scalar(
                "train_loss",
                np.mean(loss_log),
                epoch,
            )

        # evaluate policy
        if epoch % self.cfg.LOG.EVAL_EVERY_N_EPOCHS == 0:
            mean_cost = self.eval_policy(self.cfg.LOG.EVAL_ACROSS_N_RUNS)
            print("Env eval cost: {}".format(mean_cost))
            self.summary_writer.add_scalar(
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

    def eval_policy(self, N: int):
        """ Evaluate policy inferred from J*
            amont N full rollouts.

        .input:
            N: if 1 call env.reset for deterministic eval
        """
        cost_batch = jnp.zeros((N, ))
        if (N == 1):
            x_batch = jnp.expand_dims(self.env.reset(), 0)
        else:
            x_batch = self.env.sample_state(N)

        for t in range(self.env.timesteps):
            u_star_batch = self.usolver.solve(x_batch)
            g_batch = self.sys.g_fn(x_batch, u_star_batch) * self.env.h
            cost_batch += g_batch
            x_batch = self.env.step1_batch_fn(x_batch, u_star_batch)

        mean_cost = jnp.mean(cost_batch)
        return mean_cost

    def save_params(self, name: str):
        """ Save value net params
        """
        logdir = logger.get_logdir_path(self.cfg)
        params_dir = logdir.joinpath("params")
        params_dir.mkdir(exist_ok=True)
        params_file = params_dir.joinpath("{}.flax".format(name))

        param_bytes = serialization.to_bytes(self.optimizer.target)

        with open(params_file, "wb") as f:
            f.write(param_bytes)

    def load_params(self, path: str):
        """ Load value net params
        """
        with open(path, "rb") as f:
            byte_params = f.read()
        serialization.from_bytes(self.vparams, byte_params)
        serialization.from_bytes(self.optimizer.target, byte_params)

    def animate(self, name: str):
        """ Animate
        """
        # get frames
        frames = []
        self.env.reset()
        for _ in range(self.env.timesteps):
            frames.append(self.env.render(mode="rgb_array"))
            u = self.usolver.solve(jnp.expand_dims(self.env.state, 0))
            self.env.step(jnp.squeeze(u, 0)) # take a random action
        # self.env.close()

        # store
        gifs_dir = logger.get_logdir_path(self.cfg).joinpath("gifs")
        gifs_dir.mkdir(exist_ok=True)
        gif.save_frames(
            frames,
            gifs_dir.joinpath("{}.gif".format(name)),
            self.env.h,
        )
