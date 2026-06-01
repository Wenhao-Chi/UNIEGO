#!/usr/bin/env python3
"""Run the UNIEGO proxy-distillation pipeline with phase-specific overrides."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).resolve().with_name("config_spec.yaml")
PHASES = ("stage1", "gen1_infer", "merge", "stage2", "test_stage2")
BASE_CONFIGS = {
    "stage1": "base_stage1.yaml",
    "gen1_infer": "base_stage1_infer.yaml",
    "merge": "base_merge.yaml",
    "stage2": "base_stage2.yaml",
    "test_stage2": "base_stage2.yaml",
}


def quote_list(values):
    return "[" + ", ".join(f"'{value}'" for value in values) + "]"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "No"}


def config_path(dataset_key, filename):
    return REPO_ROOT / "exp" / "proxy_distill" / dataset_key / filename


def validate_stage1_dataset(name):
    if name.endswith("_gen2"):
        raise ValueError(
            "base_stage1.yaml should use the stage1 proxy dataset, "
            f"not '{name}'."
        )


def validate_stage2_dataset(name):
    if not name.endswith("_gen2"):
        raise ValueError(
            "base_stage2.yaml should use the stage2 proxy dataset ending in "
            f"_gen2, not '{name}'."
        )


def dataset_from_configs(dataset_key):
    cfg_paths = {
        phase: config_path(dataset_key, filename)
        for phase, filename in BASE_CONFIGS.items()
    }
    for path in dict.fromkeys(cfg_paths.values()):
        if not path.exists():
            raise FileNotFoundError(f"Base config not found: {path}")

    stage1_cfg = load_yaml(cfg_paths["stage1"])
    stage1_infer_cfg = load_yaml(cfg_paths["gen1_infer"])
    merge_cfg = load_yaml(cfg_paths["merge"])
    stage2_cfg = load_yaml(cfg_paths["stage2"])

    stage1_dataset = stage1_cfg["TRAIN"]["DATASET"]
    if stage1_cfg["TEST"]["DATASET"] != stage1_dataset:
        raise ValueError(
            "base_stage1.yaml TRAIN.DATASET and TEST.DATASET should match."
        )
    validate_stage1_dataset(stage1_dataset)
    if stage1_infer_cfg["TRAIN"]["DATASET"] != stage1_dataset:
        raise ValueError(
            "base_stage1_infer.yaml TRAIN.DATASET should match base_stage1.yaml."
        )
    if stage1_infer_cfg["TEST"]["DATASET"] != stage1_dataset:
        raise ValueError(
            "base_stage1_infer.yaml TEST.DATASET should match base_stage1.yaml."
        )

    stage2_dataset = stage2_cfg["TRAIN"]["DATASET"]
    if stage2_cfg["TEST"]["DATASET"] != stage2_dataset:
        raise ValueError(
            "base_stage2.yaml TRAIN.DATASET and TEST.DATASET should match."
        )
    validate_stage2_dataset(stage2_dataset)

    merged_output = merge_cfg["UNIEGO"]["MERGE_OUTPUT_PATH"]
    stage2_checkpoint = stage2_cfg["TRAIN"].get("CHECKPOINT_FILE_PATH", "")
    if stage2_checkpoint and stage2_checkpoint != merged_output:
        raise ValueError(
            "base_stage2.yaml TRAIN.CHECKPOINT_FILE_PATH should match "
            "base_merge.yaml MERGE_OUTPUT_PATH."
        )

    return {
        "key": dataset_key,
        "cfg_paths": cfg_paths,
        "stage1_dataset": stage1_dataset,
        "stage2_dataset": stage2_dataset,
        "stage1_root": stage1_cfg["DATA"]["PROXY_CANDIDATE_ROOT"],
        "stage2_output": stage2_cfg["OUTPUT_DIR"],
        "stage2_candidates": list(stage2_cfg["DATA"]["PROXY_CANDIDATES"]),
        "merged_output": merged_output,
        "stage1_max_epoch": int(stage1_cfg["SOLVER"]["MAX_EPOCH"]),
        "stage2_max_epoch": int(stage2_cfg["SOLVER"]["MAX_EPOCH"]),
    }


def checkpoint_path(output_dir, max_epoch):
    return f"{output_dir}/checkpoints/checkpoint_epoch_{max_epoch:05d}.pyth"


def teacher_names(spec):
    return list(spec["teachers"])


def candidate_overrides(candidates, base_candidates=None):
    if base_candidates is not None and list(candidates) == list(base_candidates):
        return []
    return [
        "DATA.PROXY_CANDIDATES",
        quote_list(candidates),
    ]


def remove_overridden_opts(opts, extra_opts):
    extra_keys = {extra_opts[i] for i in range(0, len(extra_opts) - 1, 2)}
    if not extra_keys:
        return opts

    filtered = []
    i = 0
    while i < len(opts):
        if i + 1 < len(opts) and opts[i] in extra_keys:
            i += 2
            continue
        filtered.append(opts[i])
        if i + 1 < len(opts):
            filtered.append(opts[i + 1])
        i += 2
    return filtered


def stage1_overrides(spec, dataset, paths, teacher, candidates):
    output_dir = f"{paths['stage1_root']}/{teacher}"
    return [
        *candidate_overrides([teacher]),
        "OUTPUT_DIR",
        output_dir,
        "UNIEGO.EXO_MODALITY",
        quote_list([teacher]),
    ]


def gen1_infer_overrides(spec, dataset, paths, teacher, candidates):
    output_dir = f"{paths['stage1_root']}/{teacher}"
    return [
        *candidate_overrides([teacher]),
        "TEST.CHECKPOINT_FILE_PATH",
        checkpoint_path(output_dir, paths["stage1_max_epoch"]),
        "OUTPUT_DIR",
        output_dir,
        "UNIEGO.EXO_MODALITY",
        quote_list([teacher]),
    ]


def merge_overrides(spec, dataset, paths, candidates):
    return candidate_overrides(candidates, paths["stage2_candidates"])


def stage2_overrides(spec, dataset, paths, candidates):
    return candidate_overrides(candidates, paths["stage2_candidates"])


def test_stage2_overrides(spec, dataset, paths, candidates):
    return [
        *candidate_overrides(candidates, paths["stage2_candidates"]),
        "TRAIN.ENABLE",
        "False",
        "TEST.CHECKPOINT_FILE_PATH",
        checkpoint_path(paths["stage2_output"], paths["stage2_max_epoch"]),
    ]


def run_command(command, *, dry_run):
    print("\n>>> " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_net_command(python_bin, base_cfg, opts, extra_opts):
    return [
        python_bin,
        "tools/run_net.py",
        "--cfg",
        str(base_cfg.relative_to(REPO_ROOT)),
        *opts,
        *extra_opts,
    ]


def merge_command(python_bin, base_cfg, opts, extra_opts):
    return [
        python_bin,
        "tools/merging_models.py",
        "--cfg",
        str(base_cfg.relative_to(REPO_ROOT)),
        *opts,
        *extra_opts,
    ]


def selected_phases(args):
    if args.phase:
        return {phase: phase in args.phase for phase in PHASES}
    return {
        "stage1": bool_env("RUN_STAGE1", True),
        "gen1_infer": bool_env("RUN_GEN1_INFER", True),
        "merge": bool_env("RUN_MERGE", True),
        "stage2": bool_env("RUN_STAGE2", True),
        "test_stage2": bool_env("RUN_TEST_STAGE2", False),
    }


def run_dataset(dataset_key, spec, args):
    dataset = dataset_from_configs(dataset_key)
    paths = dataset
    cfg_paths = paths["cfg_paths"]

    phases = selected_phases(args)
    teachers = args.teacher or teacher_names(spec)
    extra_opts = list(args.opts)
    if extra_opts and extra_opts[0] == "--":
        extra_opts = extra_opts[1:]

    print(f"\nDataset: {dataset_key}")
    print("Base configs:")
    for phase in ("stage1", "gen1_infer", "merge", "stage2"):
        print(f"  {phase}: {cfg_paths[phase].relative_to(REPO_ROOT)}")

    if phases["stage1"]:
        print("\n[stage1 proxy training]")
        for teacher in teachers:
            opts = remove_overridden_opts(
                stage1_overrides(spec, dataset, paths, teacher, teachers),
                extra_opts,
            )
            run_command(
                run_net_command(
                    args.python_bin,
                    cfg_paths["stage1"],
                    opts,
                    extra_opts,
                ),
                dry_run=args.dry_run,
            )

    if phases["gen1_infer"]:
        print("\n[stage1 proxy inference]")
        for teacher in teachers:
            opts = remove_overridden_opts(
                gen1_infer_overrides(spec, dataset, paths, teacher, teachers),
                extra_opts,
            )
            run_command(
                run_net_command(
                    args.python_bin,
                    cfg_paths["gen1_infer"],
                    opts,
                    extra_opts,
                ),
                dry_run=args.dry_run,
            )

    if phases["merge"]:
        print("\n[model merging]")
        opts = remove_overridden_opts(
            merge_overrides(spec, dataset, paths, teachers),
            extra_opts,
        )
        run_command(
            merge_command(
                args.python_bin,
                cfg_paths["merge"],
                opts,
                extra_opts,
            ),
            dry_run=args.dry_run,
        )

    if phases["stage2"]:
        print("\n[stage2 proxy training]")
        opts = remove_overridden_opts(
            stage2_overrides(spec, dataset, paths, teachers),
            extra_opts,
        )
        run_command(
            run_net_command(
                args.python_bin,
                cfg_paths["stage2"],
                opts,
                extra_opts,
            ),
            dry_run=args.dry_run,
        )

    if phases["test_stage2"]:
        print("\n[stage2 proxy testing]")
        opts = remove_overridden_opts(
            test_stage2_overrides(spec, dataset, paths, teachers),
            extra_opts,
        )
        run_command(
            run_net_command(
                args.python_bin,
                cfg_paths["test_stage2"],
                opts,
                extra_opts,
            ),
            dry_run=args.dry_run,
        )

    print(f"\n{dataset_key} proxy pipeline complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the UNIEGO proxy-distillation pipeline."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset key to run. Can be passed more than once.",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        action="append",
        help="Run only the selected phase. Can be passed more than once.",
    )
    parser.add_argument(
        "--teacher",
        action="append",
        help="Limit stage1/gen1-infer loops and proxy candidates to this teacher.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC_PATH,
        help="Pipeline spec. Defaults to exp/proxy_distill/config_spec.yaml.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used for child training/evaluation commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides appended to every child command.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spec = load_yaml(args.spec)

    if args.teacher:
        unknown_teachers = sorted(set(args.teacher) - set(teacher_names(spec)))
        if unknown_teachers:
            raise ValueError(f"Unknown teacher key(s): {unknown_teachers}")

    for dataset_key in args.dataset:
        run_dataset(dataset_key, spec, args)


if __name__ == "__main__":
    main()
