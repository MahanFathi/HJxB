from config.defaults import get_cfg_defaults
from algo.vanilla import VanillaCFVI
from network import ValueNet
from environment import make_env

import jax

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
    value_net = ValueNet(cfg)
    algo = VanillaCFVI(cfg, env, value_net)
    algo.train(cfg.TRAIN.EPOCHS)


if __name__ == '__main__':
    main()
