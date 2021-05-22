from environment import System, Env

import gym

import jax
from jax import numpy as jnp
from jax import jit, vmap

from functools import partial

import os


class DoubleIntegratorSys(System):

    def __init__(self):
        super().__init__()
        self.mass = 5.
        self.max_force = 2.
        self.max_dist = 4.20
        self.max_speed = 4.20

    def f(self, x, u):
        x, xdot = jnp.split(x, 2, -1)
        u = jnp.clip(u, -self.max_force, self.max_force)
        xdotdot = u / self.mass
        return jnp.hstack([xdot, xdotdot])

    def g(self, x, u):
        Q = jnp.array([[1., 0.], [0., .1]])
        R = jnp.array([[.005]])
        x = jnp.expand_dims(x, -1)
        u = jnp.expand_dims(u, -1)
        cost = x.T @ Q @ x + u.T @ R @ u
        return jnp.squeeze(cost)


class DoubleIntegratorEnv(Env):
    def __init__(self, h=0.05):
        super().__init__(
            DoubleIntegratorSys(), h=h,
        )
        self.action_space = gym.spaces.Box(
            low=-self.sys.max_force,
            high=self.sys.max_force,
            shape=(1, ),
            dtype=jnp.float32
        )
        self.observation_space = gym.spaces.Box(
            low=jnp.array([-self.sys.max_dist, -self.sys.max_speed]),
            high=jnp.array([self.sys.max_dist, self.sys.max_speed]),
            shape=(2, ),
            dtype=jnp.float32,
        )
