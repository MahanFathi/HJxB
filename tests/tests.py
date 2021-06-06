import unittest

import jax
from jax import numpy as jnp, test_util

import numpy as np
import scipy as sp

from config.defaults import get_cfg_defaults
from environment import make_env
from algo import make_algo

class TestDerivatives(unittest.TestCase):

    def setUp(self, ):
        self.rng = jax.random.PRNGKey(2021)
        self.cfg = get_cfg_defaults()

    def test_jacobians_dummy(self, ):
        """ Also checks if jacobians
            are implemented correctly.
        """
        self.cfg.merge_from_list([
            "ENV.ENV_NAME", "DummyEnv",
        ])
        env = make_env(self.cfg)
        sys = env.sys
        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.shape[0]

        N = 4
        self.rng, subkey = jax.random.split(self.rng)
        x = jax.random.normal(subkey, (N, obs_size))
        u = jax.random.normal(subkey, (N, act_size))

        jac_f_x, jac_f_u = sys.jac_f_fn(x, u)
        self.assertTrue(jnp.all(jnp.equal(jac_f_x, sys.A)), "f_x jac failed.")
        self.assertTrue(jnp.all(jnp.equal(jac_f_u, sys.B)), "f_u jac failed.")

        hess_g_u = sys.hess_g_u_fn(x, u)
        self.assertTrue(jnp.all(jnp.equal(hess_g_u, sys.R + sys.R.T)), "g_u hess failed.")

        test_util.check_grads(sys.f_fn, (x, u), order=1)
        test_util.check_grads(sys.g_fn, (x, u), order=2)

    def test_jacobians_pendulum(self, ):
        self.cfg.merge_from_list([
            "ENV.ENV_NAME", "PendulumEnv",
        ])
        env = make_env(self.cfg)
        sys = env.sys
        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.shape[0]

        N = 4
        self.rng, subkey = jax.random.split(self.rng)
        x = jax.random.normal(subkey, (N, obs_size))
        u = jax.random.normal(subkey, (N, act_size))

        test_util.check_grads(sys.f_fn, (x, u), order=1)
        test_util.check_grads(sys.g_fn, (x, u), order=2)

        # TODO(mahan): check if derivatives are close numerically
        # def f_x_jvp(x, v):
        #     x, v = tuple(x)[0], tuple(v)[0]
        #     f = sys.f_fn(x, u)
        #     jvp = sys.jac_f_fn(x, u)[0] @ v
        #     return f, jvp

        # NOTE(mahan): batch'd jacs should be handled
        # print(test_util.check_jvp(
        #     lambda x: sys.f_fn(x),
        #     f_x_jvp, x
        # ))

    def test_perturbation(self, ):

        self.cfg.merge_from_list([
            "TRAIN.ALGO_NAME", "DPcFVI",
            "ENV.ENV_NAME", "PendulumEnv",
            "TRAIN.DATASET_SIZE", 4,
            "TRAIN.BATCH_SIZE", 1,
        ])
        self.cfg.freeze()

        env = make_env(self.cfg)
        algo = make_algo(self.cfg, env)
        x_dataset = algo.get_x_train()
        j_dataset = algo.get_target_j(x_dataset)
        loss_fn = lambda params: algo.loss_and_grad(x_dataset, j_dataset, params)[0]
        loss_grad_fn = lambda params: algo.loss_and_grad(x_dataset, j_dataset, params)[1]
        params = algo.optimizer.target
        test_util.check_grads(loss_fn, (params, ), order=2)


if __name__ == '__main__':
    unittest.main()
