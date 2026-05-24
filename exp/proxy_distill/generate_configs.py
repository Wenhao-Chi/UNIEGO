#!/usr/bin/env python3
"""Generate proxy-distillation configs for the UNIEGO pipelines."""

import argparse
from pathlib import Path


MODALITIES = [
    "exo_rgb",
    "exo_skl",
    "exo_siglip",
    "ego_siglip",
    "exo_skego",
    "ego_depth",
    "exo_depth",
    "exo_dino",
    "ego_dino",
]

DEFAULT_MAX_EPOCH = 15
TRAIN_EVAL_PERIOD = 5
TRAIN_CHECKPOINT_PERIOD = 5
GEN1_INFER_SPLIT = "train"

DATASETS = {
    "assembly101": {
        "stage1_dataset": "Assembly101_proxy",
        "stage2_dataset": "Assembly101_proxy_gen2",
        "path_to_data_dir": "./data/Assembly101/fine-grained-segments/Ego",
        "path_prefix": "./data/Assembly101/fine-grained-segments",
        "distill_root": "./data/Assembly101/distillation",
        "num_classes": 24,
        "stage1_root": "./models/Assembly101_pi_ego/stage1",
        "stage2_output": "./models/Assembly101_pi_ego_gen2/dist_top1_from_merged",
        "merged_output": "./models/Assembly101_pi_ego/merged_checkpoints/merged_top_9_model.pyth",
        "loss_weight": 5.0,
        "loss_weight_feats": 5.0,
        "loss_weight_logits": 5.0,
    },
    "egoexo_fitness": {
        "stage1_dataset": "Egoexo_fitness_proxy",
        "stage2_dataset": "Egoexo_fitness_proxy_gen2",
        "path_to_data_dir": "./data/EgoExo-Fitness/vclip",
        "path_prefix": "./data/EgoExo-Fitness/videos_subaction",
        "distill_root": "./data/EgoExo-Fitness/distillation",
        "num_classes": 12,
        "stage1_root": "./models/EgoExo_Fitness_pi_ego/stage1",
        "stage2_output": "./models/EgoExo_Fitness_pi_ego_gen2/dist_top1_from_merged",
        "merged_output": "./models/EgoExo_Fitness_pi_ego/merged_checkpoints/merged_top_9_model.pyth",
        "stage2_exo_modality": ["feats"],
        "loss_weight": 5.0,
        "loss_weight_feats": 5.0,
        "stage1_loss_weight_logits": 10.0,
        "stage2_loss_weight_logits": 1.0,
    },
    "egoexo4d": {
        "stage1_dataset": "Egoexo4d_proxy",
        "stage2_dataset": "Egoexo4d_proxy_gen2",
        "path_to_data_dir": "./data/EgoExo4D/annotations/csvs/keystep_segments/ego",
        "path_prefix": "./data/EgoExo4D/keystep_segments_paired",
        "distill_root": "./data/EgoExo4D/distillation",
        "num_classes": 665,
        "stage1_root": "./models/EgoExo_4D_pi_ego/stage1",
        "stage2_output": "./models/EgoExo_4D_pi_ego_gen2/dist_top1_from_merged",
        "merged_output": "./models/EgoExo_4D_pi_ego/merged_checkpoints/merged_top_9_model.pyth",
        "stage2_exo_modality": ["feats", "logits"],
        "stage2_num_ensemble_views": 10,
        "top_k": 9,
        "loss_weight": 5.0,
        "loss_weight_feats": 5.0,
        "stage1_loss_weight_logits": 10.0,
        "stage2_loss_weight_logits": 5.0,
    },
}


def yaml_list(values):
    return "[" + ", ".join("'{}'".format(value) for value in values) + "]"


def checkpoint_path(output_dir):
    return f"{output_dir}/checkpoints/checkpoint_epoch_{{SOLVER.MAX_EPOCH:05d}}.pyth"


def dataset_value(dataset, key, stage=None, default=None):
    if stage is not None:
        stage_key = f"{stage}_{key}"
        if stage_key in dataset:
            return dataset[stage_key]
    if key in dataset:
        return dataset[key]
    return default


def loss_weight_block(dataset, stage):
    return f"""LOSS_WEIGHT: {dataset_value(dataset, "loss_weight", stage, 1.0):.1f}
LOSS_WEIGHT_FEATS: {dataset_value(dataset, "loss_weight_feats", stage, 5.0):.1f}
LOSS_WEIGHT_LOGITS: {dataset_value(dataset, "loss_weight_logits", stage, 5.0):.1f}"""


def common_data_block(dataset, stage):
    root = dataset["distill_root"]
    return f"""DATA:
  PATH_TO_DATA_DIR: '{dataset_value(dataset, "path_to_data_dir", stage)}'
  PATH_PREFIX: '{dataset_value(dataset, "path_prefix", stage)}'
  SKL_MODEL_BY_FEATS: '{root}/exo_skl'
  RGB_MODEL_BY_FEATS: '{root}/exo_rgb'
  EXO_SIGLIP_BY_FEATS: '{root}/exo_siglip'
  EGO_SIGLIP_BY_FEATS: '{root}/ego_siglip'
  EXO_SKEGO_BY_FEATS: '{root}/exo_skego'
  EGO_DEPTH_BY_FEATS: '{root}/ego_depth'
  EXO_DEPTH_BY_FEATS: '{root}/exo_depth'
  EGO_DINO_BY_FEATS: '{root}/ego_dino'
  EXO_DINO_BY_FEATS: '{root}/exo_dino'
  FEATS_DIR: 'features'
  PROXY_CANDIDATE_ROOT: '{dataset["stage1_root"]}'
  PROXY_CANDIDATES: {yaml_list(MODALITIES)}
  PROXY_LOGITS_DIR: 'tokens_training/scores_pred'
  PROXY_FEATS_DIR: 'tokens_training/tokens'
  NUM_FRAMES: 8
  SAMPLING_RATE: 32
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 224
  INPUT_CHANNEL_NUM: [3]
"""


def model_block(num_classes):
    return f"""TIMESFORMER:
  ATTENTION_TYPE: 'divided_space_time'
MODEL:
  MODEL_NAME: pi_proxy_vit
  NUM_CLASSES: {num_classes}
  ARCH: vit
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
"""


def runtime_block(max_epoch):
    return f"""SOLVER:
  BASE_LR: 0.005
  LR_POLICY: steps_with_relative_lrs
  STEPS: [0, 11, 14]
  LRS: [1, 0.1, 0.01]
  MAX_EPOCH: {max_epoch}
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  OPTIMIZING_METHOD: sgd
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
TENSORBOARD:
  ENABLE: False
NUM_GPUS: 4
NUM_SHARDS: 1
RNG_SEED: 0
"""


def base_stage1_config(dataset, max_epoch):
    return f"""# Common Stage 1 config for one UNIEGO proxy candidate.
TRAIN:
  ENABLE: True
  DATASET: {dataset["stage1_dataset"]}
  BATCH_SIZE: 8
  EVAL_PERIOD: {TRAIN_EVAL_PERIOD}
  CHECKPOINT_PERIOD: {TRAIN_CHECKPOINT_PERIOD}
  AUTO_RESUME: True
TEST:
  ENABLE: True
  DATASET: {dataset["stage1_dataset"]}
  BATCH_SIZE: 8
  NUM_ENSEMBLE_VIEWS: 1
  NUM_SPATIAL_CROPS: 3
  SAVE_RESULTS_PATH: 'scores_pred'
{common_data_block(dataset, "stage1")}
{model_block(dataset["num_classes"])}
{runtime_block(max_epoch)}
GENERATION: 'proxy'
TRAINING_MODE: 'dist'
LOSS_TYPE: 'cosine'
{loss_weight_block(dataset, "stage1")}
DIST_THRESHOLD: 4.0
DIST_REQUIRE_TEACHER_CORRECT: True
TOP_K: 1
GROUP_TOPK_BY_VIEW: False
SAVE_TOKENS: False
TOKEN_SAVE_DIR: 'tokens_training/tokens'
MERGE_LEVEL: 'model'
"""


def stage1_config(dataset, modality):
    return f"""# Stage 1 teacher delta: {modality}.
_BASE_: base_stage1.yaml

OUTPUT_DIR: '{dataset["stage1_root"]}/{modality}'
EXO_MODALITY: ['{modality}']
"""


def base_gen1_infer_config(dataset, max_epoch):
    return f"""# Common config for stage1 proxy inference artifact extraction.
# DATA.TEST_SPLIT is fixed to train because gen2 trains from train-split artifacts.
TRAIN:
  ENABLE: False
  DATASET: {dataset["stage1_dataset"]}
  BATCH_SIZE: 8
  AUTO_RESUME: False
TEST:
  ENABLE: True
  DATASET: {dataset["stage1_dataset"]}
  BATCH_SIZE: 8
  NUM_ENSEMBLE_VIEWS: 1
  NUM_SPATIAL_CROPS: 1
  SAVE_RESULTS_PATH: 'tokens_training/scores_pred'
{common_data_block(dataset, "stage1")}
{model_block(dataset["num_classes"])}
{runtime_block(max_epoch)}
GENERATION: 'proxy'
TRAINING_MODE: 'dist'
LOSS_TYPE: 'cosine'
{loss_weight_block(dataset, "stage1")}
DIST_THRESHOLD: 4.0
DIST_REQUIRE_TEACHER_CORRECT: True
TOP_K: 1
GROUP_TOPK_BY_VIEW: False
SAVE_TOKENS: True
TOKEN_SAVE_DIR: 'tokens_training/tokens'
MERGE_LEVEL: 'model'
"""


def gen1_infer_config(dataset, modality):
    output_dir = f"{dataset['stage1_root']}/{modality}"
    checkpoint = checkpoint_path(output_dir)
    return f"""# Run gen1 proxy inference for teacher delta: {modality}.
_BASE_: base_infer_gen1.yaml

DATA:
  TEST_SPLIT: '{GEN1_INFER_SPLIT}'
TEST:
  CHECKPOINT_FILE_PATH: '{checkpoint}'
OUTPUT_DIR: '{output_dir}'
EXO_MODALITY: ['{modality}']
"""


def base_stage2_config(dataset, max_epoch):
    stage2_exo_modality = dataset_value(
        dataset, "exo_modality", "stage2", ["feats", "logits"]
    )
    return f"""# Common Stage 2 / merge config for proxy-gen2.
TRAIN:
  ENABLE: False
  DATASET: {dataset["stage2_dataset"]}
  BATCH_SIZE: 8
  EVAL_PERIOD: {TRAIN_EVAL_PERIOD}
  CHECKPOINT_PERIOD: {TRAIN_CHECKPOINT_PERIOD}
  AUTO_RESUME: False
TEST:
  ENABLE: False
  DATASET: {dataset["stage2_dataset"]}
  BATCH_SIZE: 8
  NUM_ENSEMBLE_VIEWS: {dataset_value(dataset, "num_ensemble_views", "stage2", 1)}
  NUM_SPATIAL_CROPS: 3
  SAVE_RESULTS_PATH: 'scores_pred'
{common_data_block(dataset, "stage2")}
{model_block(dataset["num_classes"])}
{runtime_block(max_epoch)}
OUTPUT_DIR: '{dataset["stage2_output"]}'
GENERATION: 'proxy_gen2'
TRAINING_MODE: 'dist'
EXO_MODALITY: {yaml_list(stage2_exo_modality)}
LOSS_TYPE: 'cosine'
{loss_weight_block(dataset, "stage2")}
DIST_THRESHOLD: 4.0
DIST_REQUIRE_TEACHER_CORRECT: True
TOP_K: {dataset_value(dataset, "top_k", "stage2", 1)}
MERGE_TOP_K: {len(MODALITIES)}
MERGE_OUTPUT_PATH: '{dataset["merged_output"]}'
GROUP_TOPK_BY_VIEW: False
SAVE_TOKENS: False
TOKEN_SAVE_DIR: 'tokens_training/tokens'
MERGE_LEVEL: 'model'
"""


def merge_config():
    return """# Merge stage1 proxy checkpoints into one initialization for gen2.
_BASE_: base_stage2.yaml
"""


def stage2_config(dataset):
    return f"""# Stage 2: train proxy-gen2 from the merged stage1 checkpoint.
_BASE_: base_stage2.yaml

TRAIN:
  ENABLE: True
  CHECKPOINT_FILE_PATH: '{dataset["merged_output"]}'
  CHECKPOINT_EPOCH_RESET: True
TEST:
  ENABLE: True
TRAINING_MODE: 'dist'
"""


def test_stage2_config(dataset):
    checkpoint = checkpoint_path(dataset["stage2_output"])
    return f"""# Test the final proxy-gen2 checkpoint.
_BASE_: base_stage2.yaml

TRAIN:
  ENABLE: False
  AUTO_RESUME: False
TEST:
  ENABLE: True
  CHECKPOINT_FILE_PATH: '{checkpoint}'
TRAINING_MODE: 'basic'
"""


def write_dataset_configs(root, dataset_key, dataset, max_epoch):
    dataset_dir = root / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    expected_gen1_infer_configs = {
        f"infer_gen1_{modality}.yaml" for modality in MODALITIES
    }

    (dataset_dir / "base_stage1.yaml").write_text(
        base_stage1_config(dataset, max_epoch), encoding="utf-8"
    )
    stale_base_export = dataset_dir / "base_export.yaml"
    if stale_base_export.exists():
        stale_base_export.unlink()
    (dataset_dir / "base_infer_gen1.yaml").write_text(
        base_gen1_infer_config(dataset, max_epoch), encoding="utf-8"
    )
    (dataset_dir / "base_stage2.yaml").write_text(
        base_stage2_config(dataset, max_epoch), encoding="utf-8"
    )

    for modality in MODALITIES:
        (dataset_dir / f"stage1_{modality}.yaml").write_text(
            stage1_config(dataset, modality), encoding="utf-8"
        )
        stale_export = dataset_dir / f"export_{modality}.yaml"
        if stale_export.exists():
            stale_export.unlink()
        for stale_export in dataset_dir.glob("export_*.yaml"):
            stale_export.unlink()
        for stale_infer in dataset_dir.glob("infer_gen1_*.yaml"):
            if stale_infer.name not in expected_gen1_infer_configs:
                stale_infer.unlink()
        (dataset_dir / f"infer_gen1_{modality}.yaml").write_text(
            gen1_infer_config(dataset, modality), encoding="utf-8"
        )

    (dataset_dir / "merge_stage1.yaml").write_text(
        merge_config(), encoding="utf-8"
    )
    (dataset_dir / "stage2_dist.yaml").write_text(
        stage2_config(dataset), encoding="utf-8"
    )
    (dataset_dir / "test_stage2.yaml").write_text(
        test_stage2_config(dataset), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate UNIEGO proxy distillation configs."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Dataset key to generate. Defaults to all implemented datasets.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=DEFAULT_MAX_EPOCH,
        help="Epoch count used in generated solver blocks and checkpoint paths.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Config root. Defaults to this script's directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_keys = args.dataset or DATASETS.keys()
    for dataset_key in dataset_keys:
        write_dataset_configs(
            args.root, dataset_key, DATASETS[dataset_key], args.max_epoch
        )


if __name__ == "__main__":
    main()
