from config.defaults import get_cfg_defaults
from environment import make_env
from algo import make_algo

import experiment_buddy


def main():
    # build the config
    cfg = get_cfg_defaults()
    experiment_buddy.register(cfg)
    # cfg.freeze()

    env = make_env(cfg)
    algo = make_algo(cfg, env)
    algo.train(cfg.TRAIN.ITERATIONS)


if __name__ == '__main__':
    main()
