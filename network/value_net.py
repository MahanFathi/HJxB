import jax
from jax import numpy as jnp, jit
import flax
from flax import linen as nn
from yacs.config import CfgNode

from typing import (
    Sequence
)


class ValueNet(object):
    def __init__(self,
                 cfg: CfgNode,
                ):
        self.nn = DenseNet(cfg.VALUE_NET.FEATURES + [1])

    # TODO(mahan): is this the fastest way to do this?
    def pjpx_fn(self, x, params):
        """Calculate the partial derivative of J wrt x for a batch observations
        .. math::
            \pdv{J^*}{x}
        .input:
            x: (N, state_dim)
        .output:
            pjpx: (N, state_dim)
        """
        single_pjpx_fn = jax.grad(lambda x: jnp.squeeze(self.nn.apply(params, x)))
        batch_pjpx_fn = jax.vmap(single_pjpx_fn)
        return batch_pjpx_fn(x)


# TODO(mahan): this is more of a util
class DenseNet(nn.Module):
    features: Sequence[int]

    def setup(self, ):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
          x = lyr(x)
          if i != len(self.layers) - 1:
            x = nn.relu(x)
        return x


