import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# VALUE NET
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.ENV_NAME = "PendulumEnv"
_C.ENV.TIMESTEP = 0.05

# ---------------------------------------------------------------------------- #
# VALUE NET
# ---------------------------------------------------------------------------- #
_C.VALUE_NET = CN()
_C.VALUE_NET.FEATURES = [32, 16, 8]
_C.VALUE_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.ALGO_NAME = "DPcFVI"
_C.TRAIN.ITERATIONS = 256
_C.TRAIN.DATASET_SIZE = 1024
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.GAMMA = 0.9
_C.TRAIN.BUFFER_SIZE = 8192
_C.TRAIN.N_PARALLEL_ROLLOUTS = 32
_C.TRAIN.GREEDY_EPSILON = 0.9 # act greedy 90% of the times
_C.TRAIN.UPDATE_TARGET_NET_EVERY_N_EPOCHS = 1

# ---------------------------------------------------------------------------- #
# LOGGING
# ---------------------------------------------------------------------------- #
_C.LOG = CN()
_C.LOG.LOG_DIR = "./logs"
_C.LOG.LOG_EVERY_N_EPOCHS = 100
_C.LOG.EVAL_EVERY_N_EPOCHS = 100
_C.LOG.EVAL_ACROSS_N_RUNS = 10


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
