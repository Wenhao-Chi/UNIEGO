# UNIEGO

Official implementation of UNIEGO, a hierarchical proxy-distillation framework
for learning a unified egocentric video encoder from nine ego/exo teachers.

This repository supports automated experiments on:

- EgoExo-Fitness
- Assembly101
- EgoExo4D

The codebase builds on TimeSformer-style video transformers and adds a two-stage
proxy workflow: gen1 proxy training, train-split proxy artifact export,
checkpoint merging, and final gen2 proxy distillation.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate uniego
```

The environment file installs PyTorch, torchvision, fvcore, PyAV, einops, timm,
OpenCV, SciPy, scikit-learn, simplejson, and tensorboard. If your conda solver
does not install PyAV cleanly, install it from `conda-forge` before running the
video dataloaders:

```bash
conda install av -c conda-forge
```

Run commands from the repository root so Python can resolve the local
`timesformer` package.

(Optional) If you launch scripts from another directory, set:

```bash
export PYTHONPATH=/path/to/UNIEGO:$PYTHONPATH
```

## Data Preparation

UNIEGO expects video split CSV files and pre-extracted teacher features. The
default paths below are set in each dataset's stage-specific config files under
`exp/proxy_distill/<dataset>/`. If you relocate the data, edit
`DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, and the `DATA.*_BY_FEATS` teacher
feature paths in the corresponding stage configs.

| Dataset | Dataset key | Classes | Default paths |
| --- | --- | --- | --- |
| Assembly101 | `assembly101` | 24 | Split CSVs: `data/Assembly101/fine-grained-segments/Ego`<br>Videos: `data/Assembly101/fine-grained-segments`<br>Teacher features: `data/Assembly101/distillation` |
| EgoExo-Fitness | `egoexo_fitness` | 12 | Split CSVs: `data/EgoExo-Fitness/vclip`<br>Videos: `data/EgoExo-Fitness/videos_subaction`<br>Teacher features: `data/EgoExo-Fitness/distillation` |
| EgoExo4D | `egoexo4d` | 665 | Split CSVs: `data/EgoExo4D/annotations/csvs/keystep_segments/ego`<br>Videos: `data/EgoExo4D/keystep_segments_paired`<br>Teacher features: `data/EgoExo4D/distillation` |

For the paper setting:

- EgoExo-Fitness follows the official evaluation split.
- Assembly101 pairs egocentric videos from the helmet-mounted `ego04` camera
  with frontal exocentric videos from `exo03`.
- EgoExo4D follows the official split; for each ego video, we pair it with the
  dataset-annotated `best_exo` video, i.e., the most informative exocentric
  viewpoint for that sample.

### Split CSV Files

For each dataset, prepare `train.csv`, `val.csv`, and `test.csv` in the split
CSV directory. Files must not include a header. Each row is:

```text
relative_video_path,label
```

`relative_video_path` is resolved under the dataset's `DATA.PATH_PREFIX`, and
`label` is parsed as an integer class id in the range `0..NUM_CLASSES-1`.

Example:

```text
clips/ego_04_take_0001.mp4,7
clips/ego_04_take_0002.mp4,11
```

### Teacher Features

Stage 1 reads one `.npy` teacher feature per video and modality. By default,
the expected directory layout is:

```text
data/<dataset>/distillation/
  exo_rgb/
  exo_skl/
  exo_siglip/
  ego_siglip/
  exo_skego/
  ego_depth/
  exo_depth/
  exo_dino/
  ego_dino/
```

The base configs and pipeline runner map those directories to the teacher pool
used in the paper:

| Teacher key | GitHub / source | Config field | Expected feature dim |
| --- | --- | --- | --- |
| `exo_rgb` | [TimeSformer](https://github.com/facebookresearch/TimeSformer) | `DATA.RGB_MODEL_BY_FEATS` | 768 |
| `exo_skl` | [ST-GCN (PYSKL)](https://github.com/kennymckormick/pyskl) | `DATA.SKL_MODEL_BY_FEATS` | 256 |
| `exo_siglip` | [SigLIP](https://github.com/google-research/big_vision) | `DATA.EXO_SIGLIP_BY_FEATS` | 1152 |
| `ego_siglip` | [SigLIP](https://github.com/google-research/big_vision) | `DATA.EGO_SIGLIP_BY_FEATS` | 1152 |
| `exo_skego` | [SK-EGO](https://github.com/dominickrei/EgoExo4ADL) | `DATA.EXO_SKEGO_BY_FEATS` | 512 |
| `ego_depth` | [Depth Anything](https://github.com/LiheYoung/Depth-Anything) | `DATA.EGO_DEPTH_BY_FEATS` | 1024 |
| `exo_depth` | [Depth Anything](https://github.com/LiheYoung/Depth-Anything) | `DATA.EXO_DEPTH_BY_FEATS` | 1024 |
| `exo_dino` | [DINOv2](https://github.com/facebookresearch/dinov2) | `DATA.EXO_DINO_BY_FEATS` | 1024 |
| `ego_dino` | [DINOv2](https://github.com/facebookresearch/dinov2) | `DATA.EGO_DINO_BY_FEATS` | 1024 |

Feature filenames are shared across Assembly101, EgoExo-Fitness, and EgoExo4D.
For a CSV row, use the video basename without extension as the key. The filename
only encodes the view:

- Ego teacher features use the original video key.
- Exo teacher features replace `ego` with `exo` in the video key.
- The teacher modality or model name belongs to the directory path, not the
  filename.

```python
from pathlib import Path
import numpy as np
import torch


FEATURE_DIMS = {
    "exo_rgb": 768,
    "exo_skl": 256,
    "exo_siglip": 1152,
    "ego_siglip": 1152,
    "exo_skego": 512,
    "ego_depth": 1024,
    "exo_depth": 1024,
    "exo_dino": 1024,
    "ego_dino": 1024,
}

video_path = "clips/ego_04_take_0001.mp4"
modality = "exo_dino"
view = "exo"

with torch.no_grad():
    pred, feature = teacher_model(x)  # feature shape: [D] or [T, D]

feature = feature.detach().cpu().numpy().astype("float32")
assert feature.shape[-1] == FEATURE_DIMS[modality]

video_key = Path(video_path).stem
if view == "exo":
    video_key = video_key.replace("ego", "exo")

save_dir = Path("data/<DATASET>/distillation") / modality
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / f"{video_key}.npy", feature)
```

Teacher features may be saved as a single vector or as temporal features whose
last dimension matches the expected feature dim. Temporal features are averaged
inside the dataloader. Missing or shape-mismatched features are replaced with
zeros, so check feature coverage carefully before reporting final numbers.

### Gen1 Proxy Artifacts

Gen1 proxy inference exports the train-split artifacts consumed by gen2 under
each candidate directory:

```text
models/<dataset>_pi_ego/stage1/<candidate>/
  tokens_training/tokens/<ego_video_key>.npy
  tokens_training/scores_pred/<ego_video_key>.npy
```

These files keep the original ego video key, even when the candidate was trained
from an exo teacher. If you skip gen1 training or inference, place equivalent
artifacts in the same candidate directories listed by
`DATA.PROXY_CANDIDATE_ROOT` and `DATA.PROXY_CANDIDATES`.

## Usage

### Run the Full Automated Pipeline

Run one dataset end to end from the repository root:

```bash
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
bash exp/proxy_distill/assembly101/run_full_pipeline.sh
bash exp/proxy_distill/egoexo4d/run_full_pipeline.sh
```

Each runner executes gen1 training, gen1 train-split inference, checkpoint
merging, and gen2 training. Add the optional test stage with:

```bash
RUN_TEST_STAGE2=1 bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
```

### Run Individual Phases

The modular runner can launch one phase, and `--teacher` can limit the gen1
loop to a single teacher:

```bash
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase stage1 --teacher exo_rgb
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase gen1_infer --teacher exo_rgb
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase merge
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase stage2
```

`base_stage1_infer.yaml` sets `DATA.TEST_SPLIT: train`, so the export step reads
the original train split directly and writes the features/logits consumed by gen2.

### Useful Runtime Controls

The shell runners are thin wrappers around `run_pipeline.py`. Environment
variables control the Python executable and which phases run:

```bash
PYTHON_BIN=/path/to/env/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
```

To skip phases:

```bash
RUN_STAGE1=0 RUN_GEN1_INFER=0 bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh
```

Additional `KEY VALUE` config overrides can be appended after the runner
arguments. They are passed to every child command and are appended last, so they
override the selected base config and the phase defaults:

```bash
bash exp/proxy_distill/egoexo_fitness/run_full_pipeline.sh DATA_LOADER.NUM_WORKERS 4
```

To inspect the commands without running training:

```bash
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --dry-run
```

## Pipeline Configuration

UNIEGO keeps the automated pipeline compact:

- `config_spec.yaml`: teacher loop order.
- `<dataset>/base_stage1.yaml`: Stage 1 single-teacher proxy training.
- `<dataset>/base_stage1_infer.yaml`: Stage 1 train-split artifact export.
- `<dataset>/base_merge.yaml`: Stage 1 checkpoint merging.
- `<dataset>/base_stage2.yaml`: Stage 2 proxy-gen2 training and testing.
- `UNIEGO` config blocks: distillation mode, teacher modalities, loss weights,
  top-k selection, merge output, and token export settings.
- `run_pipeline.py`: expands teacher loops and gen1 inference exports.

The stage configs can also be called directly. Stage 1 train/infer configs
default to `exo_rgb`; use `run_pipeline.py` to loop over all teachers.

```bash
python tools/run_net.py --cfg exp/proxy_distill/egoexo_fitness/base_stage1.yaml
python tools/run_net.py --cfg exp/proxy_distill/egoexo_fitness/base_stage1_infer.yaml
python tools/merging_models.py --cfg exp/proxy_distill/egoexo_fitness/base_merge.yaml
python tools/run_net.py --cfg exp/proxy_distill/egoexo_fitness/base_stage2.yaml
```

## Repository Structure

```text
exp/proxy_distill/        Automated pipeline spec, base configs, and runners
timesformer/datasets/     Dataset loaders for proxy and proxy-gen2 training
timesformer/models/       Backbone and proxy ViT models
timesformer/utils/        Config parsing, checkpointing, distributed utilities
tools/run_net.py          Main train/test entry point
tools/merging_models.py   Gen1 checkpoint merging
tools/train_proxy.py      Gen1 proxy training
tools/train_proxy_gen2.py Gen2 proxy training
tools/test_net.py         Evaluation and artifact export
```

Local datasets, generated proxy artifacts, checkpoints, logs, and NumPy feature
files are intentionally ignored by Git through `.gitignore`.

## Checkpoints and Results

Final gen2 checkpoints for reproducing the paper results should be downloaded
from the links below and placed under each dataset's gen2 output directory.

| Dataset | Paper top-1 | Final gen2 checkpoint                                                                        | Expected local path |
| --- | --- |----------------------------------------------------------------------------------------------| --- |
| EgoExo-Fitness | 84.7 | [Checkpoint](https://huggingface.co/ColinChi/UNIEGO/blob/main/EgoExo_Fitness_UNIEGO_gen2.pyth) | `models/EgoExo_Fitness_pi_ego_gen2/dist_top1_from_merged/checkpoints/checkpoint_epoch_00015.pyth` |
| Assembly101 | 50.7 | [Checkpoint](https://huggingface.co/ColinChi/UNIEGO/blob/main/Assembly_101_UNIEGO_gen2.pyth) | `models/Assembly101_pi_ego_gen2/dist_top1_from_merged/checkpoints/checkpoint_epoch_00015.pyth` |
| EgoExo4D | 41.1 | [Checkpoint](https://huggingface.co/ColinChi/UNIEGO/resolve/main/EgoExo_4D_UNIEGO_gen2.pyth) | `models/EgoExo_4D_pi_ego_gen2/dist_top2_from_merged/checkpoints/checkpoint_epoch_00015.pyth` |

### Direct Inference with a Gen2 Checkpoint

After placing a checkpoint at the expected gen2 path, run the dataset's test
phase:

```bash
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase test_stage2
```

If the checkpoint or output directory is different, override the paths at
runtime:

```bash
python exp/proxy_distill/run_pipeline.py --dataset egoexo_fitness --phase test_stage2 \
  TEST.CHECKPOINT_FILE_PATH /path/to/gen2_checkpoint.pyth \
  OUTPUT_DIR /path/to/inference_outputs
```

The split directory should contain `test.csv` in `relative_video_path,label`
format. Per-video logits are saved to:

```text
<OUTPUT_DIR>/scores_pred/<video_key>.npy
```

## Citation

If this project is helpful for your research, please cite the paper once it is
available. The BibTeX entry will be added upon release.

## Acknowledgements

This repository builds on the TimeSformer codebase. We thank the authors of
TimeSformer and the open-source projects behind the teacher modalities used by UNIEGO.
