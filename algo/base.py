from environment import Env
from network import ValueNet
from utils import logger

from flax import linen as nn, optim
from flax.core.frozen_dict import FrozenDict
from yacs.config import CfgNode

from jax import numpy as jnp, jit, vmap


class Base(object):
    """ Base class,
        mutual parent to both
        discrete (bellman backup)
        and continuous methods.
    """
    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        self.cfg = cfg
        self.env = env
        self.sys = env.sys
        self._init_value_net_and_optimizer()
        self.u_star_solver_fn = self._make_u_star_solver()
        self.summary_writer = logger.get_summary_writer(cfg)

        self.dataset_size = cfg.TRAIN.DATASET_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.gamma = cfg.TRAIN.GAMMA

    def _init_value_net_and_optimizer(self, ):
        self.value_net = ValueNet(self.cfg, self.env.obs2feat)
        dummy_x = self.env.sample_state(1)
        dummy_features = self.env.obs2feat(dummy_x)
        self.vparams = self.value_net.nn.init(self.env.PRNGkey, dummy_features)
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
                      params: FrozenDict = None,
                      ):
        """Returns the optimal action wrt J* at hand
        .input:
            x: (N, state_dim)
        .output:
            u_star: (N, act_dim)
        """
        if params is None:
            params = self.vparams

        # TODO(mahan): get rid of the need for dummy_u
        dummy_u = jnp.zeros((x_batch.shape[0], self.env.action_space.shape[0]))
        R = 0.5 * self.sys.hess_g_u_fn(x_batch, dummy_u)
        f2 = self.sys.jac_f_fn(x_batch, dummy_u)[1]
        pjpx = self.value_net.pjpx_fn(x_batch, params)
        u_star = self.u_star_solver_fn(R, f2, pjpx)
        return u_star

    def train(self,
              epochs: int
              ):
        """ Train the thing
            epochs: visitation times for the dataset
        """
        raise NotImplementedError

    def eval_policy(self,
                    N: int
                    ):
        """ Evaluate policy inferred from J*
            amont N full rollouts.
        """

        cost_batch = jnp.zeros((N, ))
        x_batch = self.env.sample_state(N)

        for t in range(self.env.timesteps):
            u_star_batch = self.get_optimal_u(x_batch)
            g_batch = self.sys.g_fn(x_batch, u_star_batch) * self.env.h
            cost_batch += g_batch
            x_batch = self.env.step1_batch_fn(x_batch, u_star_batch)

        mean_cost = jnp.mean(cost_batch)
        return mean_cost
