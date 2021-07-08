from config.defaults import get_cfg_defaults
from environment import make_env
from train.algo import make_algo

import argparse


def main():
    parser = argparse.ArgumentParser(description="HJxB")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to yaml config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    env = make_env(cfg)
    algo = make_algo(cfg, env)
    algo.train(cfg.TRAIN.ITERATIONS)


if __name__ == '__main__':
    main()
