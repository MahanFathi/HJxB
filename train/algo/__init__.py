from environment import Env
from yacs.config import CfgNode

from .base_hjb import BaseHJB

from .bellman import DPcFVI
from .bellman import RTDPcFVI

from .residual import Residual

__all__ = []

__all__ += ["BaseHJB"]

__all__ += ["DPcFVI", "RTDPcFVI"]

__all__ += ["Residual"]


def make_algo(cfg: CfgNode, env: Env):
    return globals()[cfg.TRAIN.ALGO_NAME](
        cfg=cfg, env=env,
    )
