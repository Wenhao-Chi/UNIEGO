# Proxy Distillation Pipelines

This folder contains the automated proxy-distillation flows for UNIEGO:

1. Train one gen1 proxy candidate per modality.
2. Run gen1 proxy inference on the train split so gen2 can load candidate tokens and logits.
3. Merge the gen1 proxy checkpoints with `tools/merging_models.py`.
4. Train and test the final gen2 proxy from the merged checkpoint.

Implemented config sets:

- `assembly101`
- `egoexo_fitness`
- `egoexo4d`

Every task can be run directly from its config:

```bash
python tools/run_net.py --cfg exp/proxy_distill/assembly101/stage1_exo_rgb.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/infer_gen1_exo_rgb.yaml
python tools/merging_models.py --cfg exp/proxy_distill/assembly101/merge_stage1.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/stage2_dist.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/test_stage2.yaml
```

The bash runner is only a convenience wrapper that calls the same config files
in order. Run a full pipeline from the repository root:

```bash
bash exp/proxy_distill/assembly101/run_full_pipeline.sh
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
bash exp/proxy_distill/egoexo4d/run_full_pipeline.sh
```

Useful overrides:

```bash
PYTHON_BIN=/path/to/env/bin/python \
CUDA_VISIBLE_DEVICES=0,1 \
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
```

Experiment variables live in YAML configs, not in the runner. For example,
`infer_gen1_exo_rgb.yaml` sets `DATA.TEST_SPLIT: train`, so inference reads
`train.csv` directly from `DATA.PATH_TO_DATA_DIR` and writes the artifacts gen2
uses. Merge settings live in `merge_stage1.yaml` through the shared
`base_stage2.yaml`, including `DATA.PROXY_CANDIDATES`, `MERGE_TOP_K`,
`MERGE_OUTPUT_PATH`, and `MERGE_LEVEL`.

The runner still accepts extra YAML config overrides after the script path for
quick debugging, for example:

```bash
bash exp/proxy_distill/assembly101/run_full_pipeline.sh DATA_LOADER.NUM_WORKERS 4
```

Environment flags:

- `RUN_STAGE1=0`, `RUN_GEN1_INFER=0`, `RUN_MERGE=0`, or `RUN_STAGE2=0` skip a phase.
- `RUN_TEST_STAGE2=1` runs the dedicated final test config after stage2 training.
- `CONFIG_DIR=/path/to/configs` runs another config set with the same layout.
- `PYTHON_BIN=/path/to/env/bin/python` selects the Python environment.

Teacher configs use `_BASE_` inheritance to keep only the teacher-specific
delta. For example, `stage1_exo_rgb.yaml` inherits `base_stage1.yaml` and only
sets `OUTPUT_DIR` plus `EXO_MODALITY`.

To regenerate the YAML scaffold after changing defaults in
`generate_configs.py`, run:

```bash
python exp/proxy_distill/generate_configs.py --dataset assembly101 --max-epoch 15
python exp/proxy_distill/generate_configs.py --dataset egoexo_fitness --max-epoch 15
python exp/proxy_distill/generate_configs.py --dataset egoexo4d --max-epoch 15
```
