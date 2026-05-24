# UNIEGO

Official implementation of UNIEGO, a proxy-distillation pipeline for learning
ego-view video models from multiple ego/exo teacher modalities.

This repository currently supports automated experiments on:

- Assembly101
- EgoExo-Fitness
- EgoExo4D

The codebase builds on TimeSformer-style video transformers and adds a two-stage
proxy training workflow: gen1 proxy teachers, train-split feature export,
checkpoint merging, and gen2 proxy distillation.

## News

- Automated proxy-distillation configs and runners are available under
  `exp/proxy_distill`.
- Each experiment can be launched either through its YAML config directly or
  through a small convenience runner.
- Paper, project page, pretrained checkpoints, and dataset release links will be
  added when they are ready.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate uniego
```

The main dependencies are PyTorch, torchvision, fvcore, PyAV, einops, timm,
OpenCV, SciPy, scikit-learn, simplejson, and tensorboard. If your conda solver
does not install PyAV cleanly, install it from `conda-forge` before running the
video dataloaders:

```bash
conda install av -c conda-forge
```

Run commands from the repository root so Python can resolve the local
`timesformer` package. If you launch scripts from another directory, set:

```bash
export PYTHONPATH=/path/to/UNIEGO:$PYTHONPATH
```

## Data Preparation

UNIEGO expects CSV split files and pre-extracted teacher features. Each dataset
configuration sets the relevant paths in YAML:

- `DATA.PATH_TO_DATA_DIR`: directory containing split files such as `train.csv`,
  `val.csv`, and `test.csv`
- `DATA.PATH_PREFIX`: root directory for video clips
- `DATA.*_BY_FEATS`: roots for teacher features from each modality
- `DATA.PROXY_CANDIDATE_ROOT`: output root for generated gen1 proxy artifacts

Each split CSV should follow the format:

```text
path_to_video_1,label_1
path_to_video_2,label_2
...
path_to_video_N,label_N
```

The automated proxy-distillation configs currently use the following teacher
modalities:

```text
exo_rgb
exo_skl
exo_siglip
ego_siglip
exo_skego
ego_depth
exo_depth
exo_dino
ego_dino
```

Update the dataset paths in `exp/proxy_distill/*/base_*.yaml` or regenerate the
configs from `exp/proxy_distill/generate_configs.py` after changing the default
paths.

## Usage

### Run the Full Automated Pipeline

Run one dataset end to end from the repository root:

```bash
bash exp/proxy_distill/assembly101/run_full_pipeline.sh
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
bash exp/proxy_distill/egoexo4d/run_full_pipeline.sh
```

The runner executes the following stages:

1. Train one gen1 proxy candidate per modality.
2. Run gen1 proxy inference on the train split and save proxy artifacts.
3. Merge gen1 proxy checkpoints into a single initialization.
4. Train the final gen2 proxy model from the merged checkpoint.

The final test stage is optional:

```bash
RUN_TEST_STAGE2=1 bash exp/proxy_distill/assembly101/run_full_pipeline.sh
```

### Run Individual Configs

Every task can also be launched directly from its config:

```bash
python tools/run_net.py --cfg exp/proxy_distill/assembly101/stage1_exo_rgb.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/infer_gen1_exo_rgb.yaml
python tools/merging_models.py --cfg exp/proxy_distill/assembly101/merge_stage1.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/stage2_dist.yaml
python tools/run_net.py --cfg exp/proxy_distill/assembly101/test_stage2.yaml
```

The gen1 inference configs set `DATA.TEST_SPLIT: train`, so the export step reads
the original train split directly and writes the features/logits consumed by
gen2.

### Useful Runtime Overrides

The runners are intentionally thin wrappers around the configs. Common runtime
overrides are:

```bash
PYTHON_BIN=/path/to/env/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash exp/proxy_distill/assembly101/run_full_pipeline.sh
```

To skip phases:

```bash
RUN_STAGE1=0 RUN_GEN1_INFER=0 bash exp/proxy_distill/assembly101/run_full_pipeline.sh
```

Additional YAML config overrides can be appended after the script:

```bash
bash exp/proxy_distill/assembly101/run_full_pipeline.sh DATA_LOADER.NUM_WORKERS 4
```

## Config Generation

The generated YAML files use `_BASE_` inheritance so modality-specific configs
only contain their small deltas. Regenerate configs with:

```bash
python exp/proxy_distill/generate_configs.py --dataset assembly101 --max-epoch 15
python exp/proxy_distill/generate_configs.py --dataset egoexo_fitness --max-epoch 15
python exp/proxy_distill/generate_configs.py --dataset egoexo4d --max-epoch 15
```

See `exp/proxy_distill/README.md` for a more detailed description of the
automatic pipeline layout.

## Repository Structure

```text
exp/proxy_distill/        Automated experiment configs and runners
timesformer/datasets/     Dataset loaders for proxy and proxy-gen2 training
timesformer/models/       Backbone and proxy ViT models
timesformer/utils/        Config parsing, checkpointing, distributed utilities
tools/run_net.py          Main train/test entry point
tools/merging_models.py   Gen1 checkpoint merging
tools/train_proxy.py      Gen1 proxy training
tools/train_proxy_gen2.py Gen2 proxy training
tools/test_net.py         Evaluation and artifact export
```

## Checkpoints and Results

Pretrained checkpoints, merged gen1 checkpoints, and final gen2 checkpoints are
not included in this repository yet. Once released, this section should include:

- Download links for gen1 proxy checkpoints
- Download links for merged checkpoints
- Download links for final gen2 checkpoints
- A table of dataset-level metrics

## Citation

If this project is helpful for your research, please cite the paper once it is
available. The BibTeX entry will be added upon release.

## Acknowledgements

This repository builds on the TimeSformer codebase and follows the practical
README style of the VisCoP and pi-ViT repositories. We thank the authors of
TimeSformer, PySlowFast, timm, and the open-source projects behind the teacher
modalities used by UNIEGO.
