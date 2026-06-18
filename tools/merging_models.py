import os
import sys
import copy
import argparse
from collections import OrderedDict
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call

# Import core Timesformer/SlowFast dependencies
from timesformer.models import build_model
from timesformer.datasets import loader
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
import timesformer.utils.logging as logging


PROXY_MODEL_FALLBACKS = {
    "uniego_vit": "vit_base_patch16_224",
}

PROXY_DATASET_FALLBACKS = {
    "Egoexo_fitness_proxy": "EgoExo_Fitness",
    "Egoexo_4d_proxy": "EgoExo_4D",
}

MERGE_LEVEL_ALIASES = {
    "model": "model",
    "model_level": "model",
    "layer": "layer",
    "layer_level": "layer",
    "parameter": "parameter",
    "parameter_level": "parameter",
    "param": "parameter",
    "average": "average",
    "avg": "average",
    "mean": "average",
    "uniform": "average",
}

# =====================================================================
# Helper functions
# =====================================================================
def load_ckpt(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        raise KeyError(f"'model_state' not found in checkpoint: {path}")
    return ckpt


def strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def build_merge_runtime_cfg(cfg):
    merge_cfg = cfg.clone()
    merge_cfg.defrost()
    overrides = []

    model_name = merge_cfg.MODEL.MODEL_NAME
    if model_name in PROXY_MODEL_FALLBACKS:
        fallback_model = PROXY_MODEL_FALLBACKS[model_name]
        merge_cfg.MODEL.MODEL_NAME = fallback_model
        overrides.append(
            f"MODEL.MODEL_NAME: {model_name} -> {fallback_model}"
        )

    train_dataset = merge_cfg.TRAIN.DATASET
    if train_dataset in PROXY_DATASET_FALLBACKS:
        fallback_train_dataset = PROXY_DATASET_FALLBACKS[train_dataset]
        merge_cfg.TRAIN.DATASET = fallback_train_dataset
        overrides.append(
            f"TRAIN.DATASET: {train_dataset} -> {fallback_train_dataset}"
        )

    test_dataset = merge_cfg.TEST.DATASET
    if test_dataset in PROXY_DATASET_FALLBACKS:
        fallback_test_dataset = PROXY_DATASET_FALLBACKS[test_dataset]
        merge_cfg.TEST.DATASET = fallback_test_dataset
        overrides.append(
            f"TEST.DATASET: {test_dataset} -> {fallback_test_dataset}"
        )

    merge_cfg.freeze()
    return merge_cfg, overrides


def resolve_checkpoint_path(base_dir, model_name, cfg):
    model_dir = os.path.join(base_dir, model_name)
    target_checkpoint = cu.get_path_to_checkpoint(model_dir, cfg.SOLVER.MAX_EPOCH)
    if os.path.exists(target_checkpoint):
        return target_checkpoint

    if cu.has_checkpoint(model_dir):
        return cu.get_last_checkpoint(model_dir)

    raise FileNotFoundError(
        f"Model checkpoint not found under: {model_dir}. "
        f"Expected final checkpoint: {target_checkpoint}"
    )


def normalize_merge_level(merge_level):
    normalized = MERGE_LEVEL_ALIASES.get(str(merge_level).lower())
    if normalized is None:
        supported = ", ".join(["model", "layer", "parameter", "average"])
        raise ValueError(
            f"Unsupported merge level '{merge_level}'. Supported values: {supported}"
        )
    return normalized


def normalize_model_list(models):
    if models is None:
        return []
    if isinstance(models, str):
        return [model.strip() for model in models.split(",") if model.strip()]
    return [str(model).strip() for model in models if str(model).strip()]


def get_layer_group_name(key):
    parts = key.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "blocks" and idx + 1 < len(parts):
            return ".".join(parts[:idx + 2])

    leaf_names = {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    if len(parts) > 1 and parts[-1] in leaf_names:
        return ".".join(parts[:-1])
    return key


# =====================================================================
# Core class: learnable model merging (Learnable Soup)
# =====================================================================
class LearnableModelSoup(nn.Module):
    def __init__(
        self,
        base_model,
        state_dicts,
        merge_level="model",
        skip_prefixes=("model.proj_layer", "proj_layer"),
    ):
        super().__init__()
        self.base_model = base_model
        self.k = len(state_dicts)
        self.merge_level = normalize_merge_level(merge_level)
        self.skip_prefixes = skip_prefixes

        self.base_state = base_model.state_dict()
        device = next(base_model.parameters()).device
        self.model_states = [
            {
                key: value.to(device=device, non_blocking=True).detach()
                for key, value in state_dict.items()
            }
            for state_dict in state_dicts
        ]

        self.merge_keys, self.static_keys, self.dropped_keys = self._collect_keys()
        self.group_names, self.key_to_group = self._build_groups()
        if self.merge_level == "model" and self.merge_keys:
            self.w = nn.Parameter(torch.zeros(self.k))
        elif self.merge_level in {"layer", "parameter"} and self.merge_keys:
            self.w = nn.Parameter(torch.zeros(len(self.group_names), self.k))
        else:
            self.w = None
        self.register_buffer(
            "average_weights",
            torch.full((self.k,), 1.0 / self.k, device=device),
        )

    def _collect_keys(self):
        merge_keys, static_keys = [], []
        dropped = sum(1 for key in self.model_states[0] if key not in self.base_state)

        for key, base_tensor in self.base_state.items():
            if key.startswith(self.skip_prefixes):
                dropped += 1
                continue

            values = [state.get(key) for state in self.model_states]
            if any(
                value is None or tuple(value.shape) != tuple(base_tensor.shape)
                for value in values
            ):
                dropped += 1
                continue

            if values[0].is_floating_point():
                merge_keys.append(key)
            else:
                static_keys.append(key)

        return merge_keys, static_keys, dropped

    def _build_groups(self):
        if self.merge_level == "model":
            return (["model"] if self.merge_keys else []), {}
        if self.merge_level == "average":
            return [], {}
        if self.merge_level == "parameter":
            return list(self.merge_keys), {
                key: idx for idx, key in enumerate(self.merge_keys)
            }

        groups = OrderedDict()
        for key in self.merge_keys:
            group_name = get_layer_group_name(key)
            groups.setdefault(group_name, len(groups))
        return list(groups.keys()), {
            key: groups[get_layer_group_name(key)] for key in self.merge_keys
        }

    def _weights(self):
        if self.w is None or self.merge_level == "average":
            return self.average_weights
        if self.merge_level == "model":
            return F.softmax(self.w, dim=0)
        return F.softmax(self.w, dim=1)

    def _blend_key(self, key, weights):
        key_weights = (
            weights if weights.dim() == 1 else weights[self.key_to_group[key]]
        )
        return sum(
            key_weights[i] * self.model_states[i][key]
            for i in range(self.k)
        )

    def format_weight_summary(self, max_groups=4):
        if self.merge_level == "average":
            return "uniform average"
        if self.w is None:
            return "no learnable merge keys"
        if self.merge_level == "model":
            weights = self._weights().detach().cpu().numpy()
            return "[" + ", ".join([f"{w:.3f}" for w in weights]) + "]"

        weights = self._weights().detach().cpu().numpy()
        shown = []
        for idx, group_name in enumerate(self.group_names[:max_groups]):
            weight_str = ", ".join([f"{w:.3f}" for w in weights[idx]])
            shown.append(f"{group_name}=[{weight_str}]")
        if len(self.group_names) > max_groups:
            shown.append(f"... +{len(self.group_names) - max_groups} groups")
        return "; ".join(shown)

    def _print_final_weights(self):
        if not du.is_master_proc():
            return

        print("\n" + "=" * 50)
        if self.merge_level == "average":
            print("Final merge weight distribution: uniform average")
            print("=" * 50 + "\n")
            return

        if self.w is None:
            print("Final merge weight distribution: no learnable merge keys")
            print("=" * 50 + "\n")
            return

        if self.merge_level == "model":
            weights = self._weights().detach().cpu()
            print("Final model merge weight distribution")
            print("=" * 50)
            for i, w_val in enumerate(weights):
                print(f"Model {i + 1} weight: {w_val.item():.4f}")
            print("=" * 50 + "\n")
            return

        weights = self._weights().detach().cpu()
        print(f"Final {self.merge_level}-level merge weight distribution")
        print("=" * 50)
        max_groups_to_print = 30
        for group_idx, group_name in enumerate(self.group_names[:max_groups_to_print]):
            weight_str = ", ".join(
                [
                    f"model {model_idx + 1}: {weight.item():.4f}"
                    for model_idx, weight in enumerate(weights[group_idx])
                ]
            )
            print(f"{group_name}: {weight_str}")
        if len(self.group_names) > max_groups_to_print:
            omitted = len(self.group_names) - max_groups_to_print
            print(f"... {omitted} more groups omitted")
        print("=" * 50 + "\n")

    def forward(self, *args, **kwargs):
        weights = self._weights()
        blended_state = {
            key: self._blend_key(key, weights)
            for key in self.merge_keys
        }

        blended_state.update(
            {key: self.model_states[0][key] for key in self.static_keys}
        )

        return functional_call(self.base_model, blended_state, args, kwargs)

    @torch.no_grad()
    def export_final_state_dict(self):
        weights = self._weights()
        final_state = OrderedDict()

        self._print_final_weights()

        for key in self.merge_keys:
            final_state[key] = self._blend_key(key, weights).cpu()

        for key in self.static_keys:
            final_state[key] = self.model_states[0][key].cpu()

        return final_state


# Global variable used to store intercepted custom arguments
CUSTOM_ARGS = None


# =====================================================================
# Multi-GPU train/test worker function
# =====================================================================
def train_soup_worker(cfg):
    # 1. Read environment variables
    base_dir = os.environ["SOUP_BASE_DIR"]
    ranked_models = os.environ["SOUP_RANKED_MODELS"].split(",")
    top_k = int(os.environ["SOUP_TOP_K"])
    out_path = os.environ["SOUP_OUT"]
    merge_level = normalize_merge_level(os.environ.get("SOUP_MERGE_LEVEL", "model"))

    # 2. Force cfg.OUTPUT_DIR to use the requested output directory
    cfg.OUTPUT_DIR = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize distributed runtime and logging
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.LOG_FILE)
    logger = logging.get_logger(__name__)

    merge_cfg, cfg_overrides = build_merge_runtime_cfg(cfg)
    if du.is_master_proc() and cfg_overrides:
        logger.info("==> merging_models uses the original train/test pipeline with the following config overrides:")
        for override in cfg_overrides:
            logger.info(f"    - {override}")

    # 3. Resolve checkpoint paths
    selected_models = ranked_models[:top_k]
    ckpts = []
    for model_name in selected_models:
        ckpt_path = resolve_checkpoint_path(base_dir, model_name, merge_cfg)
        ckpts.append(ckpt_path)

    if du.is_master_proc():
        logger.info(
            f"==> Preparing to merge {len(ckpts)} models with merge_level='{merge_level}'..."
        )
        for path in ckpts:
            logger.info(f"    - Loading: {path}")

    # 4. Load parameters from all selected models
    state_dicts = []
    for path in ckpts:
        ckpt = load_ckpt(path)
        state_dicts.append(strip_module_prefix(ckpt["model_state"]))

    # 5. Build the base model
    base_model = build_model(merge_cfg)
    if hasattr(base_model, "module"):
        base_model = base_model.module
    base_model.eval()

    # 6. Build the Soup wrapper
    soup_model = LearnableModelSoup(
        base_model,
        state_dicts,
        merge_level=merge_level,
    )
    if merge_cfg.NUM_GPUS:
        soup_model = soup_model.cuda()

    if du.is_master_proc():
        logger.info(
            "==> Merge summary: "
            f"level={soup_model.merge_level}, "
            f"models={soup_model.k}, "
            f"merge_keys={len(soup_model.merge_keys)}, "
            f"static_keys={len(soup_model.static_keys)}, "
            f"weight_groups={len(soup_model.group_names)}, "
            f"dropped_keys={soup_model.dropped_keys}"
        )

    # ==========================
    # Stage A: optimize merge weights w
    # ==========================
    if soup_model.w is not None:
        train_loader = loader.construct_loader(merge_cfg, "train")
        optimizer = torch.optim.Adam([soup_model.w], lr=0.01)
        # optimizer = torch.optim.Adam([soup_model.w], lr=0.02, weight_decay=0.01)
        epochs = 2
        total_iters = len(train_loader)
        for epoch in range(epochs):
            soup_model.train()
            soup_model.base_model.eval()
            for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
                if merge_cfg.NUM_GPUS:
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                optimizer.zero_grad()

                preds, tokens, tokens_sec, loss_dit = soup_model(inputs, exo=None)
                loss = F.cross_entropy(preds, labels)
                loss.backward()

                if merge_cfg.NUM_GPUS > 1:
                    torch.distributed.all_reduce(
                        soup_model.w.grad, op=torch.distributed.ReduceOp.SUM
                    )
                    soup_model.w.grad /= merge_cfg.NUM_GPUS

                optimizer.step()

                if cur_iter % 10 == 0 and du.is_master_proc():
                    logger.info(
                        f"Epoch [{epoch + 1}/{epochs}] | "
                        f"Iter {cur_iter}/{total_iters} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Weights: {soup_model.format_weight_summary()}"
                    )
    elif du.is_master_proc():
        logger.info(
            "==> No learnable merge-weight optimization is needed; "
            "exporting with fixed merge weights."
        )

    # ==========================
    # Stage B: export and save the merged model
    # ==========================
    # Every process builds the merged state dict; results are identical because w is already synchronized
    final_state_dict = soup_model.export_final_state_dict()

    if du.is_master_proc():
        logger.info("\n==> Exporting and saving the final merged model...")
        template_ckpt = copy.deepcopy(load_ckpt(ckpts[0]))

        has_module_prefix = any(k.startswith("module.") for k in template_ckpt["model_state"].keys())
        if has_module_prefix:
            restored_dict = OrderedDict()
            for k, v in final_state_dict.items():
                restored_dict[f"module.{k}"] = v
            template_ckpt["model_state"] = restored_dict
        else:
            template_ckpt["model_state"] = final_state_dict

        torch.save(template_ckpt, out_path)
        logger.info(f"==> Success! The merged model has been saved to: {out_path}")

    # ==========================
    # Stage C: automatically run evaluation
    # ==========================
    du.synchronize()  # Ensure the main process finishes saving before all processes continue

    if du.is_master_proc():
        logger.info("\n" + "=" * 50)
        logger.info("==> Evaluating the merged model on the test set...")
        logger.info("=" * 50)

    # Load with strict=False for safer compatibility
    base_model.load_state_dict(final_state_dict, strict=False)

    # Build the test DataLoader and TestMeter
    test_loader = loader.construct_loader(merge_cfg, "test")
    from timesformer.utils.meters import TestMeter
    test_meter = TestMeter(
        len(test_loader.dataset) // (merge_cfg.TEST.NUM_ENSEMBLE_VIEWS * merge_cfg.TEST.NUM_SPATIAL_CROPS),
        merge_cfg.TEST.NUM_ENSEMBLE_VIEWS * merge_cfg.TEST.NUM_SPATIAL_CROPS,
        merge_cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        merge_cfg.DATA.MULTI_LABEL,
        merge_cfg.DATA.ENSEMBLE_METHOD,
    )

    # Reuse the core testing routine from test_net.py
    from tools.test_net import perform_test

    # Run the native testing flow (logging, prediction export, confusion matrix export)
    perform_test(test_loader, base_model, test_meter, merge_cfg, writer=None)

    if du.is_master_proc():
        logger.info(f"\n==> Merging and evaluation are complete. Logs and outputs are stored in: {cfg.OUTPUT_DIR}")


def main():
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument("--base_dir", type=str, default=None)
    custom_parser.add_argument("--ranked_models", type=str, nargs="+", default=None)
    custom_parser.add_argument("--top_k", type=int, default=None)
    custom_parser.add_argument("--out", type=str, default=None)
    custom_parser.add_argument("--merge_level", type=str, default=None)

    CUSTOM_ARGS, remaining_argv = custom_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_argv

    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    if CUSTOM_ARGS.merge_level is None:
        CUSTOM_ARGS.merge_level = cfg.UNIEGO.MERGE_LEVEL
    try:
        CUSTOM_ARGS.merge_level = normalize_merge_level(CUSTOM_ARGS.merge_level)
    except ValueError as err:
        custom_parser.error(str(err))

    if CUSTOM_ARGS.base_dir is None:
        CUSTOM_ARGS.base_dir = cfg.DATA.PROXY_CANDIDATE_ROOT
    if not CUSTOM_ARGS.base_dir:
        custom_parser.error(
            "--base_dir is required when cfg.DATA.PROXY_CANDIDATE_ROOT is empty"
        )

    if CUSTOM_ARGS.ranked_models is None:
        CUSTOM_ARGS.ranked_models = normalize_model_list(cfg.DATA.PROXY_CANDIDATES)
    else:
        CUSTOM_ARGS.ranked_models = normalize_model_list(CUSTOM_ARGS.ranked_models)
    if not CUSTOM_ARGS.ranked_models:
        custom_parser.error(
            "--ranked_models is required when cfg.DATA.PROXY_CANDIDATES is empty"
        )

    if CUSTOM_ARGS.top_k is None:
        CUSTOM_ARGS.top_k = cfg.UNIEGO.MERGE_TOP_K if cfg.UNIEGO.MERGE_TOP_K > 0 else cfg.UNIEGO.TOP_K
    if CUSTOM_ARGS.top_k > len(CUSTOM_ARGS.ranked_models):
        print(f"Warning: requested top_k exceeds the number of ranked models. Auto-adjusting to {len(CUSTOM_ARGS.ranked_models)}")
        CUSTOM_ARGS.top_k = len(CUSTOM_ARGS.ranked_models)

    if CUSTOM_ARGS.out is None:
        CUSTOM_ARGS.out = cfg.UNIEGO.MERGE_OUTPUT_PATH
    if not CUSTOM_ARGS.out:
        out_filename = f"merged_top_{CUSTOM_ARGS.top_k}.pyth"
        if CUSTOM_ARGS.merge_level != "model":
            out_filename = f"merged_top_{CUSTOM_ARGS.top_k}_{CUSTOM_ARGS.merge_level}.pyth"
        CUSTOM_ARGS.out = os.path.join(
            cfg.OUTPUT_DIR,
            out_filename,
        )

    os.environ["SOUP_BASE_DIR"] = CUSTOM_ARGS.base_dir
    os.environ["SOUP_RANKED_MODELS"] = ",".join(CUSTOM_ARGS.ranked_models)
    os.environ["SOUP_TOP_K"] = str(CUSTOM_ARGS.top_k)
    os.environ["SOUP_OUT"] = CUSTOM_ARGS.out
    os.environ["SOUP_MERGE_LEVEL"] = CUSTOM_ARGS.merge_level

    launch_job(cfg=cfg, init_method=args.init_method, func=train_soup_worker)


if __name__ == "__main__":
    main()
