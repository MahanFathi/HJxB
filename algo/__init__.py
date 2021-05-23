from environment import Env
from yacs.config import CfgNode

from .DPcFVI import DPcFVI
from .RTDPcFVI import RTDPcFVI

__all__ = ["DPcFVI", "RTDPcFVI"]

def make_algo(cfg: CfgNode, env: Env):
    return globals()[cfg.TRAIN.ALGO_NAME](
        cfg=cfg, env=env,
    )
