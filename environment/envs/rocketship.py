from environment import System, Env

import gym

import jax
from jax import numpy as jnp
from jax import jit, vmap

from functools import partial

import os


class RocketShipSys(System):

    def __init__(self):
        super().__init__()
        self.mass = 1.
        self.max_force = 2.
        self.max_th_acc = 2.
        self.max_dist = 4.20
        self.max_speed = 4.20
        self.max_th_speed = 3.

    def f(self, x, u):
        q1, q1d, q2, q2d, th, thd = jnp.split(x, 6, -1)
        u1, u2 = jnp.split(u, 2, -1)
        q1dd = u1 * jnp.cos(th) / self.mass
        q2dd = u1 * jnp.sin(th) / self.mass
        thdd = u2
        return jnp.hstack([
            q1d,
            q1dd,
            q2d,
            q2dd,
            thd,
            thdd,
        ])

    def g(self, x, u):
        R = jnp.array([[.05, .0], [.0, 0.01]])
        Q = jnp.array([
            [1.,  0.,  0.,  0.,  0.,  0.],
            [0.,  .5,  0.,  0.,  0.,  0.],
            [0.,  0.,  1.,  0.,  0.,  0.],
            [0.,  0.,  0.,  .5,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.],
            [0.,  0.,  0.,  0.,  0.,  0.],
        ])
        x = jnp.expand_dims(x, -1)
        u = jnp.expand_dims(u, -1)
        cost = x.T @ Q @ x + u.T @ R @ u
        return jnp.squeeze(cost)


class RocketShipEnv(Env):
    def __init__(self, h=0.05):
        super().__init__(
            RocketShipSys(), h=h,
        )
        self.T = 5.
        self.action_space = gym.spaces.Box(
            low=jnp.array([-self.sys.max_force, -self.sys.max_th_acc]),
            high=jnp.array([self.sys.max_force, self.sys.max_th_acc]),
            shape=(2, ),
            dtype=jnp.float32,
        )
        obs_high = jnp.array([
            self.sys.max_dist, self.sys.max_speed,
            self.sys.max_dist, self.sys.max_speed,
            jnp.pi, self.sys.max_th_speed,
        ])
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            shape=(6, ),
            dtype=jnp.float32,
        )
        self.viewer = None

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-5., 5., -5., 5.)
            dot = rendering.make_circle(.1)
            body = rendering.make_capsule(1., .1)
            body.set_color(.8, .3, .3)
            dot.set_color(.0, .0, .0)
            self.transform = rendering.Transform()
            body.add_attr(self.transform)
            self.viewer.add_geom(dot)
            self.viewer.add_geom(body)

        self.transform.set_translation(self.state[0], self.state[2])
        self.transform.set_rotation(self.state[4])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def obs2feat(self, x):
        q, th, thd = jnp.split(x, [4, 5], -1)
        return jnp.hstack([
            q,
            jnp.cos(th), jnp.sin(th),
            thd,
        ])

    def reset(self):
        # deterministic eval
        self.state = jnp.array([
            -self.sys.max_dist / 2., 0.,
            self.sys.max_dist / 2., 0.,
            0.0, 0.0,
        ])
        return self.state
