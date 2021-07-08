from environment import Env
from network import ValueNet
from utils import logger, gif
from train.usolvers import make_usolver

from flax import linen as nn, optim, serialization
from flax.core.frozen_dict import FrozenDict
from yacs.config import CfgNode

import jax
from jax import numpy as jnp


class BaseHJB(object):
    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        self.cfg = cfg
        self.env = env
        self.sys = env.sys
        self._dump_train_config()
        self._init_value_net_and_optimizer()
        self.summary_writer = logger.get_summary_writer(cfg)

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
                     params: FrozenDict,
                     ):
        """Returns the target J*
        .input:
            x: (N, state_dim)
        .output:
            j: (N, 1)
        """
        u_star_batch = self.usolver.solve(x_batch, params)
        x_next_batch = self.env.step1_batch_fn(x_batch, u_star_batch)
        j_next_batch = self.value_net.apply(params, x_next_batch)
        g_batch = self.sys.g_fn(x_batch, u_star_batch) * self.env.h # TODO(mahan) a bit ugly
        j_batch = g_batch[:, None] + self.gamma * j_next_batch
        return j_batch

    def get_residual(self,
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
        u_star_batch = self.usolver.solve(x_batch, self.vparams)
        pjpx_batch = self.value_net.pjpx_fn(x_batch, params) # TODO(mahan): computed twice
        g_batch = self.sys.g_fn(x_batch, u_star_batch)
        f_batch = self.sys.f_fn(x_batch, u_star_batch)
        residuals = g_batch[:, None] + pjpx_batch @ jnp.expand_dims(f_batch, -1)
        return residuals

    def train(self, epochs: int):
        """Train the thing
        epochs: visitation times for the dataset
        """
        raise NotImplementedError

    def eval_policy(self, N: int):
        """ Evaluate policy inferred from J*
            amont N full rollouts.

        .input:
            N: if 1 call env.reset for deterministic eval
        """
        cost_batch = jnp.zeros((N,))
        if (N == 1):
            x_batch = jnp.expand_dims(self.env.reset(), 0)
        else:
            x_batch = self.env.sample_state(N)

        for t in range(self.env.timesteps):
            u_star_batch = self.usolver.solve(x_batch, self.vparams)
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
            self.env.step(jnp.squeeze(u, 0))  # take a random action
        # self.env.close()

        # store
        gifs_dir = logger.get_logdir_path(self.cfg).joinpath("gifs")
        gifs_dir.mkdir(exist_ok=True)
        gif.save_frames(
            frames,
            gifs_dir.joinpath("{}.gif".format(name)),
            self.env.h,
        )
