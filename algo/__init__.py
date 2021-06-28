from environment import Env
from yacs.config import CfgNode

from .base import BaseAlgo
from .DPcFVI import DPcFVI
from .RTDPcFVI import RTDPcFVI

__all__ = ["BaseAlgo", "DPcFVI", "RTDPcFVI"]

def make_algo(cfg: CfgNode, env: Env):
    return globals()[cfg.TRAIN.ALGO_NAME](
        cfg=cfg, env=env,
    )
