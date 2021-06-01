from environment import Env
from yacs.config import CfgNode

from .bellman.DPcFVI import DPcFVI
from .bellman.RTDPcFVI import RTDPcFVI

from .continuous.DDcFV import DDcFV

__all__ = ["DPcFVI", "RTDPcFVI"]
__all__ += ["DDcFV"]

def make_algo(cfg: CfgNode, env: Env):
    return globals()[cfg.TRAIN.ALGO_NAME](
        cfg=cfg, env=env,
    )
