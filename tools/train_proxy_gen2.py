# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a proxy student with proxy candidates as teachers."""

import json
import os
import numpy as np
import pprint
import torch
import torch.nn.functional as F
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
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter
from timesformer.utils.multigrid import MultigridSchedule

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

logger = logging.get_logger(__name__)


def get_proxy_candidate_names(proxy_candidates):
    if isinstance(proxy_candidates, str):
        candidates = [proxy_candidates] if proxy_candidates else []
    else:
        candidates = [candidate for candidate in proxy_candidates if candidate]

    if not candidates:
        raise ValueError(
            "train_proxy_gen2 requires cfg.DATA.PROXY_CANDIDATES to contain at least one candidate model."
        )

    return candidates


def get_proxy_branch_flags(exo_modality):
    if isinstance(exo_modality, str):
        modalities = [exo_modality] if exo_modality else []
    else:
        modalities = [modality for modality in exo_modality if modality]

    use_random = "random" in modalities
    use_auto = "auto" in modalities
    use_feat_branch = "feats" in modalities or use_auto or "logits" not in modalities
    use_logits_branch = "logits" in modalities or use_auto

    return use_feat_branch, use_logits_branch, use_random


def get_batch_size(inputs):
    if isinstance(inputs, list):
        return inputs[0].size(0)
    return inputs.size(0)


def get_candidate_stat_name(candidate_name):
    return candidate_name.replace(os.sep, "_").replace(" ", "_")


def move_candidate_meta_to_gpu(meta, candidate_names):
    for candidate_name in candidate_names:
        for suffix in ("feats", "logits"):
            meta_key = f"{candidate_name}_{suffix}"
            if meta_key in meta:
                meta[meta_key] = meta[meta_key].cuda(non_blocking=True)


def build_candidate_tensors(meta, candidate_names, require_logits=True):
    feats = []
    logits = []

    for candidate_name in candidate_names:
        feat_key = f"{candidate_name}_feats"
        logits_key = f"{candidate_name}_logits"

        if feat_key not in meta:
            raise KeyError(f"Missing proxy candidate feature '{feat_key}' in dataset meta.")
        if require_logits and logits_key not in meta:
            raise KeyError(f"Missing proxy candidate logits '{logits_key}' in dataset meta.")

        feats.append(meta[feat_key])
        if logits_key in meta:
            logits.append(meta[logits_key])

    return feats, logits


def calculate_dist_loss(
    student_token,
    teacher_tokens,
    loss_type,
    loss_weight=1.0,
    element_weight=None,
    reduction="mean",
):
    if student_token.dim() == 3:
        valid_mask = (
            teacher_tokens.abs().reshape(teacher_tokens.size(0), -1).amax(dim=1) > 1e-6
        ).to(student_token.dtype)

        if loss_type == "cosine":
            if element_weight is not None:
                student_token = student_token * element_weight
                teacher_tokens = teacher_tokens * element_weight
            per_sample = 1.0 - F.cosine_similarity(student_token, teacher_tokens, dim=-1)
            per_sample = per_sample.mean(dim=1)
            loss_per_sample = loss_weight * (per_sample * valid_mask)
        else:
            mse = (student_token - teacher_tokens).pow(2)
            if element_weight is not None:
                mse = mse * element_weight
            mse_per_sample = mse.mean(dim=-1).mean(dim=-1)
            loss_per_sample = loss_weight * (mse_per_sample * valid_mask)
    else:
        valid_mask = (teacher_tokens.abs().amax(dim=1) > 1e-6).to(student_token.dtype)

        if loss_type == "cosine":
            if element_weight is not None:
                student_token = student_token * element_weight
                teacher_tokens = teacher_tokens * element_weight
            per_sample = 1.0 - F.cosine_similarity(student_token, teacher_tokens, dim=-1)
            loss_per_sample = loss_weight * (per_sample * valid_mask)
        else:
            mse = (student_token - teacher_tokens).pow(2)
            if element_weight is not None:
                mse = mse * element_weight
            mse_per_sample = mse.mean(dim=-1)
            loss_per_sample = loss_weight * (mse_per_sample * valid_mask)

    if reduction == "none":
        return loss_per_sample
    return loss_per_sample.sum() / valid_mask.sum().clamp(min=1.0)


def calculate_logits_dist_loss(
    student_logits,
    teacher_logits,
    loss_weight=1.0,
    reduction="mean",
):
    valid_mask = (teacher_logits.abs().amax(dim=1) > 1e-6).to(student_logits.dtype)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    kl_per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    loss_per_sample = loss_weight * (kl_per_sample * valid_mask)

    if reduction == "none":
        return loss_per_sample
    return loss_per_sample.sum() / valid_mask.sum().clamp(min=1.0)


def update_candidate_count_stats(extra_stats, candidate_names, selected_indices, valid_mask):
    for idx, candidate_name in enumerate(candidate_names):
        stat_name = get_candidate_stat_name(candidate_name)
        extra_stats[f"dist_{stat_name}_count"] = ((selected_indices == idx) & valid_mask).sum().item()


def init_candidate_selection_counts(candidate_names):
    return {candidate_name: 0 for candidate_name in candidate_names}


def accumulate_candidate_selection_counts(selection_counts, candidate_names, selected_indices, valid_mask):
    if not candidate_names:
        return

    if selected_indices.dim() == 1:
        selected_indices = selected_indices.unsqueeze(1)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(1)

    valid_mask = valid_mask.bool()
    for idx, candidate_name in enumerate(candidate_names):
        selection_counts[candidate_name] += int(((selected_indices == idx) & valid_mask).sum().item())


def reduce_candidate_selection_counts(selection_counts, candidate_names, use_gpu):
    if not candidate_names:
        return {}

    device = torch.device("cuda", torch.cuda.current_device()) if use_gpu else torch.device("cpu")
    count_tensor = torch.tensor(
        [selection_counts[candidate_name] for candidate_name in candidate_names],
        dtype=torch.float32,
        device=device,
    )
    if du.get_world_size() > 1:
        du.all_reduce([count_tensor], average=False)

    return {
        candidate_name: int(count_tensor[idx].item())
        for idx, candidate_name in enumerate(candidate_names)
    }


def build_candidate_group_indices(candidate_names):
    ego_indices = []
    exo_indices = []
    for idx, candidate_name in enumerate(candidate_names):
        base_name = os.path.basename(candidate_name).lower()
        if base_name.startswith("ego"):
            ego_indices.append(idx)
        elif base_name.startswith("exo"):
            exo_indices.append(idx)
        else:
            return None

    if not ego_indices or not exo_indices:
        return None

    return [exo_indices, ego_indices]


def select_topk_candidates(scores, top_k, candidate_group_indices=None, largest=False):
    top_k = min(top_k, scores.size(1))
    if top_k <= 0:
        empty_idx = torch.empty(scores.size(0), 0, dtype=torch.long, device=scores.device)
        empty_scores = scores[:, :0]
        return empty_scores, empty_idx

    if candidate_group_indices is None:
        return scores.topk(k=top_k, dim=1, largest=largest)

    selected_scores = []
    selected_idx = []
    for group_indices in candidate_group_indices:
        group_top_k = min(top_k, len(group_indices))
        group_index_tensor = torch.tensor(group_indices, dtype=torch.long, device=scores.device)
        group_scores = scores.index_select(1, group_index_tensor)
        group_topk_scores, group_topk_rel_idx = group_scores.topk(k=group_top_k, dim=1, largest=largest)
        group_topk_idx = group_index_tensor[group_topk_rel_idx]
        selected_scores.append(group_topk_scores)
        selected_idx.append(group_topk_idx)

    return torch.cat(selected_scores, dim=1), torch.cat(selected_idx, dim=1)


def save_candidate_selection_summary(selection_counts, output_dir):
    summary = {
        "total_valid_selections": int(sum(selection_counts.values())),
        "candidate_selection_counts": selection_counts,
    }
    save_path = os.path.join(output_dir, "proxy_candidate_selection_counts.json")
    with open(save_path, "w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)

    logger.info("Final proxy candidate selection counts: %s", summary)
    logger.info("Saved proxy candidate selection summary to %s", save_path)


def train_epoch(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    candidate_group_indices=None,
):
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    candidate_names = (
        get_proxy_candidate_names(cfg.DATA.PROXY_CANDIDATES)
        if cfg.TRAINING_MODE == "dist"
        else []
    )
    top_k = min(cfg.TOP_K, len(candidate_names)) if candidate_names else 0
    use_feat_branch, use_logits_branch, use_random_candidates = get_proxy_branch_flags(cfg.EXO_MODALITY)
    epoch_selection_counts = init_candidate_selection_counts(candidate_names)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, list):
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if candidate_names:
                move_candidate_meta_to_gpu(meta, candidate_names)

        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        train_meter.data_toc()

        if not cfg.MIXUP.ENABLED:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            target_labels = labels
        else:
            mixup_fn = Mixup(
                mixup_alpha=cfg.MIXUP.ALPHA,
                cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
                cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX,
                prob=cfg.MIXUP.PROB,
                switch_prob=cfg.MIXUP.SWITCH_PROB,
                mode=cfg.MIXUP.MODE,
                label_smoothing=0.1,
                num_classes=cfg.MODEL.NUM_CLASSES,
            )
            hard_labels = labels
            inputs, labels = mixup_fn(inputs, labels)
            loss_fun = SoftTargetCrossEntropy()
            target_labels = hard_labels

        preds, tokens, cls_tokens, _ = model(inputs, exo=None)
        extra_stats = {}
        loss = loss_fun(preds, labels)
        loss_cls = loss
        extra_stats["loss_cls"] = loss.item()

        if cfg.TRAINING_MODE == "basic":
            pass
        elif cfg.TRAINING_MODE == "dist":
            all_feats, all_logits = build_candidate_tensors(meta, candidate_names)
            feats_stack = torch.stack(all_feats, dim=1)
            logits_stack = torch.stack(all_logits, dim=1)
            batch_idx = torch.arange(target_labels.size(0), device=target_labels.device)

            ce_losses = [
                F.cross_entropy(candidate_logits, target_labels, reduction="none")
                for candidate_logits in all_logits
            ]
            ce_stack = torch.stack(ce_losses, dim=1)

            with torch.no_grad():
                student_ce = F.cross_entropy(preds, target_labels, reduction="none")

            teacher_preds = logits_stack.argmax(dim=-1)

            correct_mask = teacher_preds == target_labels.unsqueeze(1)
            if cfg.DIST_REQUIRE_TEACHER_CORRECT:
                valid_mask = correct_mask
            else:
                valid_mask = torch.ones_like(correct_mask, dtype=torch.bool)
            valid_mask = valid_mask & (ce_stack < student_ce.unsqueeze(1)) & (ce_stack <= cfg.DIST_THRESHOLD)
            masked_ce = torch.where(valid_mask, ce_stack, torch.full_like(ce_stack, float("inf")))
            topk_ce, topk_idx = select_topk_candidates(masked_ce, top_k, candidate_group_indices, largest=False)
            selected_teacher_count = topk_idx.size(1)
            if use_random_candidates:
                topk_idx = torch.randint(
                    0, len(candidate_names), (target_labels.size(0), selected_teacher_count), device=target_labels.device
                )
                topk_ce = masked_ce[batch_idx.unsqueeze(1), topk_idx]

            selected_valid_mask = torch.isfinite(topk_ce)
            has_valid_teacher = selected_valid_mask.any(dim=1).float()
            valid_teacher_counts = selected_valid_mask.float().sum(dim=1).clamp(min=1.0)
            loss_dist_feat_per_sample = torch.zeros(target_labels.size(0), device=target_labels.device)
            loss_dist_logits_per_sample = torch.zeros(target_labels.size(0), device=target_labels.device)


            for idx in range(selected_teacher_count):
                curr_teacher_idx = topk_idx[:, idx]
                curr_quality_mask = selected_valid_mask[:, idx].float()
                curr_feats = feats_stack[batch_idx, curr_teacher_idx]
                curr_logits = logits_stack[batch_idx, curr_teacher_idx]
                curr_weight = curr_quality_mask / valid_teacher_counts
                # breakpoint()
                if use_feat_branch:
                    step_loss_feat = calculate_dist_loss(
                        student_token=tokens,
                        teacher_tokens=curr_feats,
                        loss_type=cfg.LOSS_TYPE,
                        loss_weight=cfg.LOSS_WEIGHT_FEATS,
                        reduction="none",
                    )
                    loss_dist_feat_per_sample += curr_weight * step_loss_feat

                if use_logits_branch:
                    step_loss_logits = calculate_logits_dist_loss(
                        student_logits=preds,
                        teacher_logits=curr_logits,
                        loss_weight=cfg.LOSS_WEIGHT_LOGITS,
                        reduction="none",
                    )
                    loss_dist_logits_per_sample += curr_weight * step_loss_logits

            if use_feat_branch:
                loss_dist_feat = loss_dist_feat_per_sample.sum() / has_valid_teacher.sum().clamp(min=1.0)
                extra_stats["loss_dist_feat"] = loss_dist_feat.item()
                loss = loss + loss_dist_feat
            if use_logits_branch:
                loss_dist_logits = loss_dist_logits_per_sample.sum() / has_valid_teacher.sum().clamp(min=1.0)
                extra_stats["loss_dist_logits"] = loss_dist_logits.item()
                loss = loss + loss_dist_logits

            extra_stats["dist_valid_count"] = has_valid_teacher.sum().item()
            accumulate_candidate_selection_counts(
                epoch_selection_counts,
                candidate_names,
                topk_idx,
                selected_valid_mask,
            )
            update_candidate_count_stats(
                extra_stats,
                candidate_names,
                topk_idx,
                selected_valid_mask,
            )
        else:
            raise ValueError(
                "train_proxy_gen2 only supports TRAINING_MODE 'basic' or 'dist', "
                f"but got '{cfg.TRAINING_MODE}'"
            )

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

        misc.check_nan_losses(loss)
        batch_size = get_batch_size(inputs)

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward()
            if (cur_iter + 1) % num_iters == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        top1_err, top5_err = None, None
        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                [loss] = du.all_reduce([loss])
            loss = loss.item()
        else:
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()

        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            batch_size * max(cfg.NUM_GPUS, 1),
            stats=extra_stats,
        )
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

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return epoch_selection_counts


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, list):
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()

        val_meter.data_toc()
        batch_size = get_batch_size(inputs)
        preds, tokens, _, _ = model(inputs)

        if cfg.DATA.MULTI_LABEL:
            if cfg.NUM_GPUS > 1:
                preds, labels = du.all_gather([preds, labels])
        else:
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(top1_err, top5_err, batch_size * max(cfg.NUM_GPUS, 1))

            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

        val_meter.update_predictions(preds, labels)

        if cfg.SAVE_TOKENS:
            token_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TOKEN_SAVE_DIR)
            os.makedirs(token_save_dir, exist_ok=True)
            for name, token in zip(meta["filename"], tokens.detach().cpu().numpy()):
                save_path = os.path.join(token_save_dir, f"{name}.npy")
                np.save(save_path, token)
                print(f"[Saved] {save_path}")

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [label.clone().detach() for label in val_meter.all_labels]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, list):
                    for idx in range(len(inputs)):
                        inputs[idx] = inputs[idx].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    optimizer = optim.construct_optimizer(model, cfg)
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(cfg, "train", is_precise_bn=True)
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter


def train(cfg):
    du.init_distributed_training(cfg)
    if cfg.DETECTION.ENABLE:
        raise ValueError("train_proxy_gen2 does not support detection mode.")

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR)

    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)

    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))
    candidate_names = []
    total_selection_counts = {}
    candidate_group_indices = None
    if cfg.TRAINING_MODE == "dist":
        candidate_names = get_proxy_candidate_names(cfg.DATA.PROXY_CANDIDATES)
        if cfg.GROUP_TOPK_BY_VIEW:
            candidate_group_indices = build_candidate_group_indices(candidate_names)
        total_selection_counts = init_candidate_selection_counts(candidate_names)
        logger.info("Proxy candidate teachers: %s", candidate_names)
        if cfg.GROUP_TOPK_BY_VIEW and candidate_group_indices is not None:
            logger.info("Enabled grouped top-k selection by ego/exo view.")

    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    optimizer = optim.construct_optimizer(model, cfg)

    if not cfg.TRAIN.FINETUNE:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    else:
        start_epoch = 0
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model)

    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True) if cfg.BN.USE_PRECISE_STATS else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    logger.info("Start epoch: %s", start_epoch + 1)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                model, optimizer, train_loader, val_loader, precise_bn_loader, train_meter, val_meter = build_trainer(
                    cfg
                )

                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from %s", last_checkpoint)
                cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer)

        loader.shuffle_dataset(train_loader, cur_epoch)
        epoch_selection_counts = train_epoch(
            train_loader,
            model,
            optimizer,
            train_meter,
            cur_epoch,
            cfg,
            writer,
            candidate_group_indices,
        )
        for candidate_name in candidate_names:
            total_selection_counts[candidate_name] += epoch_selection_counts.get(candidate_name, 0)

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )

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

        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if candidate_names:
        total_selection_counts = reduce_candidate_selection_counts(
            total_selection_counts,
            candidate_names,
            cfg.NUM_GPUS > 0,
        )
        if du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            save_candidate_selection_summary(total_selection_counts, cfg.OUTPUT_DIR)

    if writer is not None:
        writer.close()
