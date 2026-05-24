# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse

import timesformer.utils.checkpoint as cu
from timesformer.config.defaults import get_cfg


def _resolve_runtime_placeholders(cfg):
    max_epoch = cfg.SOLVER.MAX_EPOCH
    replacements = {
        "{SOLVER.MAX_EPOCH:05d}": "{:05d}".format(max_epoch),
        "{SOLVER.MAX_EPOCH}": str(max_epoch),
    }

    for node in (cfg.TRAIN, cfg.TEST):
        checkpoint_path = node.CHECKPOINT_FILE_PATH
        if not isinstance(checkpoint_path, str):
            continue
        for placeholder, value in replacements.items():
            checkpoint_path = checkpoint_path.replace(placeholder, value)
        node.CHECKPOINT_FILE_PATH = checkpoint_path


def parse_args():
    """
    Parse the command line arguments for UNIEGO training and evaluation.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Run a UNIEGO video training, inference, or evaluation config."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See timesformer/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    _resolve_runtime_placeholders(cfg)

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
