from environment import Env
from network import ValueNet

from flax import linen as nn, optim
from yacs.config import CfgNode

from jax import numpy as jnp, jit, vmap
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
                      x_batch: jnp.ndarray, # (N, obs_dim)
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

# def loss(self, params):
