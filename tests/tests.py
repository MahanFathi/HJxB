import unittest

import jax
from jax import numpy as jnp

from environment.envs.dummy import DummySys, DummyEnv

class TestDerivatives(unittest.TestCase):

    def setUp(self, ):
        self.rng = jax.random.PRNGKey(2021)
        self.env = DummyEnv()
        self.sys = self.env.sys

    def test_derivatives(self, ):
        N = 4
        self.rng, subkey = jax.random.split(self.rng)
        x = jax.random.normal(subkey, (N, self.sys.obs_size))
        u = jax.random.normal(subkey, (N, self.sys.act_size))

        jac_f_x, jac_f_u = self.sys.jac_f_fn(x, u)
        self.assertTrue(jnp.all(jnp.equal(jac_f_x, self.sys.A)), "f_x jac failed.")
        self.assertTrue(jnp.all(jnp.equal(jac_f_u, self.sys.B)), "f_u jac failed.")

        hess_g_u = self.sys.hess_g_u_fn(x, u)
        self.assertTrue(jnp.all(jnp.equal(hess_g_u, self.sys.R + self.sys.R.T)), "g_u hess failed.")


if __name__ == '__main__':
    unittest.main()
