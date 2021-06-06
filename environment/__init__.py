from .core import System, Env
from .envs.dummy import DummySys, DummyEnv
from .envs.double_integrator import DoubleIntegratorSys, DoubleIntegratorEnv
from .envs.pendulum import PendulumSys, PendulumEnv
from yacs.config import CfgNode

__all__ = ["System", "Env"]

__all__ += ["DummySys", "DummyEnv"]
__all__ += ["DoubleIntegratorSys", "DoubleIntegratorEnv"]
__all__ += ["PendulumSys", "PendulumEnv"]

def make_env(cfg: CfgNode):
    return globals()[cfg.ENV.ENV_NAME](h=cfg.ENV.TIMESTEP)
