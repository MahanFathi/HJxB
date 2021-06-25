from environment import Env
from yacs.config import CfgNode


class USolver(object):

    def __init__(self,
                 cfg: CfgNode,
                 env: Env,
                 ):
        self.cfg = cfg
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape
        self.u_star_solver_fn = self._make_solver()

    def _make_solver(self, ):
        raise NotImplementedError
