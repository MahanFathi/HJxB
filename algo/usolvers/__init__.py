from environment import Env
from yacs.config import CfgNode

from .convex_conjugate import ConvConjSolver
from .quadratic import QuadraticSolver

__all__ = ["QuadraticSolver", "ConvConjSolver"]


def make_usolver(cfg: CfgNode, env: Env):
    return globals()[cfg.TRAIN.USOLVER_NAME](
        cfg=cfg, env=env,
    )
