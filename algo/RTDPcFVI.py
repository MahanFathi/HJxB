from .base import BaseAlgo
from environment import Env
from utils.buffer import ReplayBuffer

import jax
from jax import jit, numpy as jnp
from yacs.config import CfgNode
import random


class RTDPcFVI(BaseAlgo):
    """Continuous Fitted Value Iteration"""

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        super().__init__(cfg, env)

        self.parallel_rollouts = cfg.TRAIN.N_PARALLEL_ROLLOUTS

        self.replay_buffer = ReplayBuffer(
            size=cfg.TRAIN.BUFFER_SIZE,
            batch_size=self.parallel_rollouts,
            observation_space=env.observation_space,
        )

        self.epoch_counter = 0
        self.x_prev = None
        self.greedy_epsilon = cfg.TRAIN.GREEDY_EPSILON

        self.rng = jax.random.PRNGKey(999) # rip juice

    def get_x_train(self, N: int = None):
        """
        This method can also be thought of as
        a sub-routine that gets called before
        each training epoch.
        """

        if self.epoch_counter % self.env.timesteps == 0:
            self.x_prev = self.env.sample_state(self.parallel_rollouts)
            self.replay_buffer.add(self.x_prev)

        x = self._collect_1step(self.x_prev)
        self.x_prev = x

        N = N or self.dataset_size
        assert self.replay_buffer.capacity >= N
        while N > self.replay_buffer.size:
            x = self._collect_1step(self.x_prev)
            self.x_prev = x

        self.epoch_counter += 1

        x_train = self.replay_buffer.sample(N)
        return x_train


    def _collect_1step(self, x_batch: jnp.ndarray, epsilon_greedy: bool = True):
        """
        Takes an optimal step wrt the J* and
        stores transitions, which are basically
        the next states, in the replay buffer.
        """
        # act epsilon-greedy
        if random.random() < self.greedy_epsilon:
            u_star_batch = self.get_optimal_u(x_batch)
        else:
            self.rng, input_rng = jax.random.split(self.rng)
            u_star_batch = jax.random.uniform(
                input_rng,
                (x_batch.shape[0], ) + self.env.action_space.shape,
                minval=-self.env.action_space.high,
                maxval= self.env.action_space.high,
            )

        x_next_batch = self.env.step1_batch_fn(x_batch, u_star_batch)
        self.replay_buffer.add(x_next_batch)

        return x_next_batch
