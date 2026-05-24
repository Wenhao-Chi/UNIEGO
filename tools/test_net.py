# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix
import cv2
from einops import rearrange, reduce, repeat
import scipy.io

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter

logger = logging.get_logger(__name__)


def _save_preds_as_npy(test_loader, all_preds, cfg):
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
    os.makedirs(save_dir, exist_ok=True)

    num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    preds_np = all_preds.detach().cpu().numpy()
    for video_id, pred in enumerate(preds_np):
        video_item = test_loader.dataset._path_to_videos[video_id * num_clips]
        if isinstance(video_item, (list, tuple)):
            video_item = video_item[0]
        video_name = os.path.splitext(os.path.basename(video_item))[0]
        np.save(os.path.join(save_dir, "{}.npy".format(video_name)), pred)

    logger.info(
        "Successfully saved {} prediction npy files to {}".format(
            len(preds_np), save_dir
        )
    )


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            # for key, val in meta.items():
            #     if isinstance(val, (list,)):
            #         for i in range(len(val)):
            #             val[i] = val[i].cuda(non_blocking=True)
            #     else:
            #         meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds, tokens, cls_tokens, _ = model(inputs)

            if cfg.SAVE_TOKENS:
                token_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TOKEN_SAVE_DIR)
                os.makedirs(token_save_dir, exist_ok=True)
                tokens_to_save = cls_tokens if cfg.GENERATION == "proxy" else tokens
                if tokens_to_save is not None:
                    for name, token in zip(meta['filename'], tokens_to_save.detach().cpu().numpy()):
                        save_path = os.path.join(token_save_dir, f"{name}.npy")
                        np.save(save_path, token)
                        print(f"[Saved] {save_path}")

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
                _save_preds_as_npy(test_loader, all_preds, cfg)
            du.synchronize()

        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            try:
                preds_cls = torch.argmax(all_preds, dim=1).numpy()
                labels_cls = all_labels.numpy()
                cm = confusion_matrix(labels_cls, preds_cls)
                cm_path = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.csv")
                np.savetxt(cm_path, cm, delimiter=",", fmt='%d')
                logger.info(f"Confusion Matrix (CSV) successfully saved to {cm_path}")
            except Exception as e:
                logger.info(f"Failed to save confusion matrix: {e}")

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
