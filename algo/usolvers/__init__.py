import algo
from environment import Env
from yacs.config import CfgNode

from .convex_conjugate import ConvConjSolver
from .quadratic import QuadraticSolver

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from algo import BaseAlgo


__all__ = ["QuadraticSolver", "ConvConjSolver"]


def make_usolver(cfg: CfgNode, algo: 'BaseAlgo'):
    return globals()[cfg.TRAIN.USOLVER_NAME](algo)
