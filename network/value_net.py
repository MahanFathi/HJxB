import jax
from jax import numpy as jnp, jit
import flax
from flax import linen as nn
from yacs.config import CfgNode

from typing import (
    Sequence,
    Callable,
)


class ValueNet(object):
    def __init__(self,
                 cfg: CfgNode,
                 obs2feat_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
                ):
        """
        obs2feat_fn: feature extractor function
        """
        self.obs2feat_fn = obs2feat_fn
        if self.obs2feat_fn is None:
            self.obs2feat_fn = lambda x: x
        self.nn = DenseNet(cfg.VALUE_NET.FEATURES + [1])

    def apply(self, params, x):
        """Wrapper around nn.apply"""
        features = self.obs2feat_fn(x)
        return self.nn.apply(params, features)

    # TODO(mahan): is this the best way to do this?
    def pjpx_fn(self, x, params):
        """Calculate the partial derivative of J wrt x for a batch observations
        .. math::
            \pdv{J^*}{x}
        .input:
            x: (N, state_dim)
        .output:
            pjpx: (N, 1, state_dim)
        """
        single_pjpx_fn = jax.jacfwd(lambda x: self.apply(params, x))
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


