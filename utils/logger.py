from flax.metrics import tensorboard
from yacs.config import CfgNode

from datetime import datetime
from pathlib import Path

LOG_PATH = None

def get_logdir_path(cfg: CfgNode) -> Path:
    global LOG_PATH
    if LOG_PATH:
        return LOG_PATH
    logdir_name = "{}_{}".format(
        cfg.ENV.ENV_NAME,
        datetime.now().strftime("%Y.%m.%d_%H:%M:%S"),
    )
    log_path = Path(cfg.LOG.LOG_DIR).joinpath(logdir_name)
    print(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    LOG_PATH = log_path
    return log_path

def get_summary_writer(cfg: CfgNode) -> tensorboard.SummaryWriter:
    log_path = get_logdir_path(cfg)
    return tensorboard.SummaryWriter(str(log_path))
