from environment import System, Env

import gym

import jax
from jax import numpy as jnp

import os


class DummySys(System):
    """This env is used in unit tests"""

    def __init__(self):
        super().__init__()
        self.obs_size = 3
        self.act_size = 2

        self.rng = jax.random.PRNGKey(0)
        self.rng, subkey = jax.random.split(self.rng)

        self.A = jax.random.uniform(subkey, (self.obs_size, self.obs_size))
        self.B = jax.random.uniform(subkey, (self.obs_size, self.act_size))
        self.R = jax.random.uniform(subkey, (self.act_size, self.act_size))

    def f(self, x, u):
        x = jnp.expand_dims(x, -1)
        u = jnp.expand_dims(u, -1)
        return jnp.squeeze(self.A @ x + self.B @ u, -1)

    def g(self, x, u):
        u = jnp.expand_dims(u, -1)
        return jnp.squeeze(u.T @ self.R @ u)


class DummyEnv(Env):
    def __init__(self, h=0.05):
        super().__init__(
            DummySys(), h=h,
        )
        self.T = 3.
        self.action_space = gym.spaces.Box(
            low=jnp.array([-10.] * self.sys.act_size),
            high=jnp.array([10.] * self.sys.act_size),
            shape=(self.sys.act_size, ),
            dtype=jnp.float32
        )
        self.observation_space = gym.spaces.Box(
            low=jnp.array([-10.] * self.sys.obs_size),
            high=jnp.array([10.] * self.sys.obs_size),
            shape=(self.sys.obs_size, ),
            dtype=jnp.float32,
        )
