from .core import System, Env
from .envs.pendulum import PendulumSys, PendulumEnv
from .envs.double_integrator import DoubleIntegratorSys, DoubleIntegratorEnv
from yacs.config import CfgNode

__all__ = ["System", "Env"]

__all__ += ["PendulumSys", "PendulumEnv"]

__all__ += ["DoubleIntegratorSys", "DoubleIntegratorEnv"]

def make_env(cfg: CfgNode):
    return globals()[cfg.ENV.ENV_NAME](h=cfg.ENV.TIMESTEP)
