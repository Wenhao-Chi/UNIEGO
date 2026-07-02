# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import os
import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import timesformer.models.losses as losses
import timesformer.models.optimizer as optim
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.datasets import utils as data_utils
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter
from timesformer.utils.multigrid import MultigridSchedule

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

logger = logging.get_logger(__name__)


def get_distill_modalities(exo_modality):
    """
    Normalize cfg.UNIEGO.EXO_MODALITY and keep one or more modalities for proxy distillation.
    """
    if isinstance(exo_modality, str):
        modalities = [exo_modality] if exo_modality else []
    else:
        modalities = [modality for modality in exo_modality if modality]

    if not modalities:
        raise ValueError("train_proxy requires cfg.UNIEGO.EXO_MODALITY to contain at least one modality.")

    return list(dict.fromkeys(modalities))


def build_modality_feat_dict(meta, use_gpu):
    """
    Keep all modality-specific teacher features in one dictionary so it is easy
    to extend with new modalities later.
    """
    modality_feats = {
        "exo_rgb": meta["exo_rgb"],
        "exo_skl": meta["exo_skl"],
        "exo_siglip": meta["exo_siglip"],
        "ego_siglip": meta["ego_siglip"],
        "exo_skego": meta["exo_skego"],
        "ego_depth": meta["ego_depth"],
        "exo_depth": meta["exo_depth"],
        "ego_dino": meta["ego_dino"],
        "exo_dino": meta["exo_dino"],
    }

    if use_gpu:
        for modality_name, feat_value in modality_feats.items():
            modality_feats[modality_name] = feat_value.cuda(non_blocking=True)

    return modality_feats


def build_student_token_dict(tokens, aux_tokens, selected_modalities):
    if len(selected_modalities) == 1:
        return {selected_modalities[0]: tokens}

    if not isinstance(aux_tokens, dict):
        raise ValueError(
            "Multi-modality proxy distillation expects the model to return projected tokens per modality."
        )

    missing_modalities = [modality for modality in selected_modalities if modality not in aux_tokens]
    if missing_modalities:
        raise KeyError(
            f"Missing projected tokens for modalities {missing_modalities}. "
            f"Available tokens: {list(aux_tokens.keys())}"
        )

    return {modality: aux_tokens[modality] for modality in selected_modalities}


def get_batch_size(inputs):
    if isinstance(inputs, (list,)):
        return inputs[0].size(0)
    return inputs.size(0)


def calculate_dist_loss(student_token, teacher_tokens, loss_type, loss_weight=1.0):
    """
    Args:
        student_token (Tensor): [B, C]
        teacher_tokens (Tensor): [B, C]
        loss_type (String): 'cosine' or 'mse'
        loss_weight (float): loss weight for dist loss

    Returns:
        loss (Tensor)
    """
    valid_mask = (teacher_tokens.abs().amax(dim=1) > 1e-6).to(student_token.dtype)  # [B]
    if loss_type == 'cosine':
        per_sample = 1.0 - torch.nn.functional.cosine_similarity(student_token, teacher_tokens, dim=-1)  # (B,)
        loss = loss_weight * (per_sample * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    else:
        mse_per_sample = (student_token - teacher_tokens).pow(2).mean(dim=-1)
        loss = loss_weight * (mse_per_sample * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    return loss


def train_epoch(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    selected_modalities = get_distill_modalities(cfg.UNIEGO.EXO_MODALITY)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        if not cfg.MIXUP.ENABLED:
           loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        else:
           mixup_fn = Mixup(
               mixup_alpha=cfg.MIXUP.ALPHA, cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX, prob=cfg.MIXUP.PROB, switch_prob=cfg.MIXUP.SWITCH_PROB, mode=cfg.MIXUP.MODE,
               label_smoothing=0.1, num_classes=cfg.MODEL.NUM_CLASSES)
           hard_labels = labels
           inputs, labels = mixup_fn(inputs, labels)
           loss_fun = SoftTargetCrossEntropy()

        modality_feat_dict = build_modality_feat_dict(meta, cfg.NUM_GPUS > 0)
        for selected_modality in selected_modalities:
            if selected_modality not in modality_feat_dict:
                raise KeyError(
                    f"Unsupported EXO_MODALITY '{selected_modality}'. "
                    f"Available modalities: {list(modality_feat_dict.keys())}"
                )
        # breakpoint()
        preds, tokens, aux_tokens, _ = model(inputs, exo=None)
        extra_stats = {}
        loss = loss_fun(preds, labels)
        loss_cls = loss
        extra_stats["loss_cls"] = loss.item()
        if cfg.UNIEGO.TRAINING_MODE == 'basic':
            pass
        elif cfg.UNIEGO.TRAINING_MODE == 'dist':
            student_token_dict = build_student_token_dict(tokens, aux_tokens, selected_modalities)
            loss_feat_values = []

            for selected_modality in selected_modalities:
                selected_feats = modality_feat_dict[selected_modality]
                loss_feat = calculate_dist_loss(
                    student_token=student_token_dict[selected_modality],
                    teacher_tokens=selected_feats,
                    loss_type=cfg.UNIEGO.LOSS_TYPE,
                    loss_weight=cfg.UNIEGO.LOSS_WEIGHT,
                )
                # breakpoint()
                extra_stats[f"loss_dist_{selected_modality}_feat"] = loss_feat.item()
                loss_feat_values.append(loss_feat)

            loss_feat_total = torch.stack(loss_feat_values).mean()
            extra_stats["loss_dist_feat"] = loss_feat_total.item()
            loss = loss + loss_feat_total
        else:
            raise ValueError(
                f"train_proxy only supports TRAINING_MODE 'basic' or 'dist', "
                f"but got '{cfg.UNIEGO.TRAINING_MODE}'"
            )

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

        # check Nan Loss.
        misc.check_nan_losses(loss)

        batch_size = get_batch_size(inputs)


        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        top1_err, top5_err = None, None
        if cfg.DATA.MULTI_LABEL:
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce(
                    [loss, top1_err, top5_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            stats=extra_stats
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        val_meter.data_toc()
        batch_size = get_batch_size(inputs)
        tokens = None
        preds, tokens, aux_tokens, _ = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

        val_meter.update_predictions(preds, labels)
        
        if cfg.UNIEGO.SAVE_TOKENS:
            token_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.UNIEGO.TOKEN_SAVE_DIR)
            os.makedirs(token_save_dir, exist_ok=True)
            if isinstance(aux_tokens, dict):
                for modality_name, modality_tokens in aux_tokens.items():
                    modality_save_dir = os.path.join(token_save_dir, modality_name)
                    os.makedirs(modality_save_dir, exist_ok=True)
                    for name, token in zip(meta['filename'], modality_tokens.detach().cpu().numpy()):
                        save_path = os.path.join(modality_save_dir, f"{name}.npy")
                        np.save(save_path, token)
                        print(f"[Saved] {save_path}")
            elif tokens is not None:
                for name, token in zip(meta['filename'], tokens.detach().cpu().numpy()):
                    save_path = os.path.join(token_save_dir, f"{name}.npy")
                    np.save(save_path, token)
                    print(f"[Saved] {save_path}")

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [
            label.clone().detach() for label in val_meter.all_labels
        ]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(
            preds=all_preds, labels=all_labels, global_step=cur_epoch
        )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    if cfg.DETECTION.ENABLE:
        raise ValueError("train_proxy does not support detection mode.")
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.LOG_FILE)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if not cfg.TRAIN.FINETUNE:
      start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
      start_epoch = 0
      cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader,
            model,
            optimizer,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()

    if cfg.DATA.REPORT_DISTILL_COVERAGE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        data_utils.log_distill_coverage(train_loader.dataset)
