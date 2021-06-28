import algo
from environment import Env
from yacs.config import CfgNode

from .convex_conjugate import ConvConjSolver
from .quadratic import QuadraticSolver
from .discrete import DiscreteSolver

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from algo import BaseAlgo


__all__ = ["QuadraticSolver", "ConvConjSolver", "DiscreteSolver"]


def make_usolver(cfg: CfgNode, algo: 'BaseAlgo'):
    return globals()[cfg.USOLVER.USOLVER_NAME](algo)
