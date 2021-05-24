import numpy as np
from gym import spaces
from jax import numpy as jnp


class ReplayBuffer:

    def __init__(self,
                 size: int,
                 batch_size: int,
                 observation_space: spaces.Space,
                 # action_space: spaces.Space,
                 ):

        assert size % batch_size == 0, "Buffer size should be divisible by batch size."

        self.pos = 0
        self.capacity = size
        self.batch_size = batch_size
        self.n_input_sofar = 0

        self.observation_space = observation_space
        # self.action_space = action_space

        self.x_storage = np.empty((self.capacity, ) + observation_space.shape, dtype=observation_space.dtype)

    @property
    def size(self, ):
        return min(self.capacity, self.n_input_sofar)

    def add(self,
            x: jnp.ndarray, # size (N, obs_dim)
            ):
        assert x.shape[0] == self.batch_size

        if self.pos == self.capacity:
            self.pos = 0

        self.x_storage[self.pos:(self.pos + self.batch_size), :] = x
        self.pos += self.batch_size
        self.n_input_sofar += self.batch_size

    def sample(self,
               N: int,
               ):
        idx = np.random.randint(self.size, size=N)
        return jnp.asarray(self.x_storage[idx, :])
