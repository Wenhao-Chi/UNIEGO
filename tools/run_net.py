# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args

from tools.test_net import test
from tools.train_proxy import train as train_proxy
from tools.train_proxy_gen2 import train as train_proxy_gen2
# from tools.train_net_gen2 import evaluate_teachers
# from tools.train_net_online import train
# from tools.test_net_online import test


def get_func(cfg):
    # train_func = train
    if cfg.UNIEGO.GENERATION == 'proxy_gen2':
        train_func = train_proxy_gen2
    elif cfg.UNIEGO.GENERATION == 'proxy':
        train_func = train_proxy
    else:
        raise ValueError(
            "tools/run_net.py supports GENERATION 'proxy' or 'proxy_gen2', "
            f"but got '{cfg.UNIEGO.GENERATION}'"
        )
    test_func = test
    return train_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    train, test = get_func(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        # launch_job(cfg=cfg, init_method=args.init_method, func=evaluate_teachers)
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
