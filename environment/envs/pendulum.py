from environment import System, Env

import gym

import jax
from jax import numpy as jnp
from jax import jit

from functools import partial

import os


class PendulumSys(System):

   def __init__(self, gravity=9.81):
      self.max_speed = 8
      self.max_torque = 2.
      self.gravity = gravity
      self.mass = 1.
      self.lenght = 1.

   @partial(jit, static_argnums=(0,))
   def f(self, x, u):
      th, thdot = x

      gravity = self.gravity
      m = self.mass
      l = self.lenght

      u = jnp.clip(u, -self.max_torque, self.max_torque)[0]
      thdotdot = (-3 * gravity / (2 * l) * jnp.sin(th + jnp.pi) + 3. / (m * l ** 2) * u)
      clipped_thdot = jnp.clip(thdot, -self.max_speed, self.max_speed) # any good in doing this?
      return jnp.array([clipped_thdot, thdotdot])

   @partial(jit, static_argnums=(0,))
   def g(self, x, u):
      th, thdot = x
      cost = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
      return cost

def angle_normalize(x):
   return (jnp.mod((x + jnp.pi), (2 * jnp.pi)) - jnp.pi)


class PendulumEnv(Env):
   def __init__(self, ):
      super(PendulumEnv, self).__init__(
         sys=PendulumSys(), h=0.05,
      )
      self.action_space = gym.spaces.Box(
         low=-self.sys.max_torque,
         high=self.sys.max_torque,
         shape=(1, ),
         dtype=jnp.float32
      )
      self.observation_space = gym.spaces.Box(
         low=jnp.array([-jnp.pi, -self.sys.max_speed]),
         high=jnp.array([jnp.pi, self.sys.max_speed]),
         shape=(2, ),
         dtype=jnp.float32,
      )
      self.viewer = None
      self.seed()

   # render methods are shameless copied from openai gym
   def render(self, mode='human'):
      if self.viewer is None:
         from gym.envs.classic_control import rendering
         self.viewer = rendering.Viewer(500, 500)
         self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
         rod = rendering.make_capsule(1, .2)
         rod.set_color(.8, .3, .3)
         self.pole_transform = rendering.Transform()
         rod.add_attr(self.pole_transform)
         self.viewer.add_geom(rod)
         axle = rendering.make_circle(.05)
         axle.set_color(0, 0, 0)
         self.viewer.add_geom(axle)
      self.pole_transform.set_rotation(self.state[0] + jnp.pi / 2)
      return self.viewer.render(return_rgb_array=mode == 'rgb_array')

   def close(self):
      if self.viewer:
         self.viewer.close()
         self.viewer = None
