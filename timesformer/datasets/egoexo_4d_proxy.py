# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
import numpy as np
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


def _normalize_exo_modalities(exo_modality):
    if isinstance(exo_modality, str):
        return [exo_modality] if exo_modality else []
    return [modality for modality in exo_modality if modality]


def _resolve_distill_modalities(exo_modality):
    modalities = _normalize_exo_modalities(exo_modality)
    if not modalities:
        raise ValueError("egoexo_4d_proxy requires at least one distillation modality.")
    return list(dict.fromkeys(modalities))


@DATASET_REGISTRY.register()
class Egoexo4d_proxy(torch.utils.data.Dataset):
    """
    EgoExo-4D proxy dataset aligned with Egoexo_fitness_proxy.
    """

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for EgoExo-4D".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EgoExo-4D proxy {}...".format(mode))
        self._construct_loader()

        self.skl_model_by_feats = cfg.DATA.SKL_MODEL_BY_FEATS
        self.rgb_model_by_feats = cfg.DATA.RGB_MODEL_BY_FEATS
        self.exo_siglip_by_feats = cfg.DATA.EXO_SIGLIP_BY_FEATS
        self.ego_siglip_by_feats = cfg.DATA.EGO_SIGLIP_BY_FEATS
        self.exo_skego_by_feats = cfg.DATA.EXO_SKEGO_BY_FEATS
        self.ego_depth_by_feats = cfg.DATA.EGO_DEPTH_BY_FEATS
        self.exo_depth_by_feats = cfg.DATA.EXO_DEPTH_BY_FEATS
        self.ego_dino_by_feats = cfg.DATA.EGO_DINO_BY_FEATS
        self.exo_dino_by_feats = cfg.DATA.EXO_DINO_BY_FEATS
        self.feats_dir = cfg.DATA.FEATS_DIR
        self.distill_modalities = _resolve_distill_modalities(cfg.EXO_MODALITY)

        self.modality_configs = {
            'exo_rgb': {
                'feat_dir': os.path.join(self.rgb_model_by_feats),
                'feat_dim': 768,
                'meta_key': 'exo_rgb',
            },
            'exo_skl': {
                'feat_dir': os.path.join(self.skl_model_by_feats),
                'feat_dim': 256,
                'meta_key': 'exo_skl',
            },
            'exo_siglip': {
                'feat_dir': os.path.join(self.exo_siglip_by_feats),
                'feat_dim': 1152,
                'meta_key': 'exo_siglip',
            },
            'ego_siglip': {
                'feat_dir': os.path.join(self.ego_siglip_by_feats),
                'feat_dim': 1152,
                'meta_key': 'ego_siglip',
            },
            'exo_skego': {
                'feat_dir': os.path.join(self.exo_skego_by_feats),
                'feat_dim': 512,
                'meta_key': 'exo_skego',
            },
            'ego_depth': {
                'feat_dir': os.path.join(self.ego_depth_by_feats),
                'feat_dim': 1024,
                'meta_key': 'ego_depth',
            },
            'exo_depth': {
                'feat_dir': os.path.join(self.exo_depth_by_feats),
                'feat_dim': 1024,
                'meta_key': 'exo_depth',
            },
            'ego_dino': {
                'feat_dir': os.path.join(self.ego_dino_by_feats),
                'feat_dim': 1024,
                'meta_key': 'ego_dino',
            },
            'exo_dino': {
                'feat_dir': os.path.join(self.exo_dino_by_feats),
                'feat_dim': 1024,
                'meta_key': 'exo_dino',
            }
        }
        unsupported_modalities = [
            modality for modality in self.distill_modalities if modality not in self.modality_configs
        ]
        if unsupported_modalities:
            raise KeyError(
                f"Unsupported distillation modalities {unsupported_modalities}. "
                f"Available modalities: {list(self.modality_configs.keys())}"
            )

    def _load_distill_feature(self, feat_path, feat_dim):
        if not os.path.exists(feat_path):
            return torch.zeros(feat_dim, dtype=torch.float32)

        feature = np.asarray(np.load(feat_path))

        if feature.ndim == 1:
            if feature.shape[0] != feat_dim:
                logger.warning(
                    "Feature dim mismatch for %s: expected %d, got %s. Falling back to zeros.",
                    feat_path,
                    feat_dim,
                    feature.shape,
                )
                return torch.zeros(feat_dim, dtype=torch.float32)
            return torch.tensor(feature, dtype=torch.float32)

        if feature.ndim == 2 and feature.shape[0] == feat_dim and feature.shape[1] != feat_dim:
            feature = feature.transpose(1, 0)

        if feature.shape[-1] != feat_dim:
            if feature.size % feat_dim != 0:
                logger.warning(
                    "Feature shape mismatch for %s: expected last dim %d, got %s. Falling back to zeros.",
                    feat_path,
                    feat_dim,
                    feature.shape,
                )
                return torch.zeros(feat_dim, dtype=torch.float32)
            feature = feature.reshape(-1, feat_dim)
        else:
            feature = feature.reshape(-1, feat_dim)

        if feature.shape[0] == 0:
            return torch.zeros(feat_dim, dtype=torch.float32)

        temporal_indices = np.linspace(
            0,
            feature.shape[0] - 1,
            self.cfg.DATA.NUM_FRAMES,
            dtype=np.int64,
        )
        feature = feature[temporal_indices].mean(axis=0)
        return torch.tensor(feature, dtype=torch.float32)

    def _get_distill_filename(self, filename, modality_name):
        if modality_name.startswith("exo_"):
            return filename.replace("ego", "exo")
        return filename

    def _construct_loader(self):
        split_name = utils.get_split_name(self.cfg, self.mode)
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(split_name)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load EgoExo-4D split {} from {}".format(
            split_name, path_to_file
        )
        logger.info(
            "Constructing EgoExo-4D proxy dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                center_frame=self.cfg.DATA.CENTER_FRAME_CLIP,
            )

            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]

            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            frames = frames.permute(3, 0, 1, 2)
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            if not self.cfg.MODEL.ARCH in ['vit']:
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                frames = torch.index_select(
                     frames,
                     1,
                     torch.linspace(
                         0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )

            meta_data = {}
            meta_data['filename'] = self._path_to_videos[index].split("/")[-1].split(".")[0]
            meta_data['distill_modality'] = self.distill_modalities

            for modality_name, modality_cfg in self.modality_configs.items():
                feat_dim = modality_cfg['feat_dim']
                meta_key = modality_cfg['meta_key']

                if modality_name in self.distill_modalities:
                    filename = self._get_distill_filename(
                        os.path.basename(meta_data['filename']),
                        modality_name,
                    ) + '.npy'
                    feat_path = os.path.join(modality_cfg['feat_dir'], filename)
                    meta_data[meta_key] = self._load_distill_feature(feat_path, feat_dim)
                    # print(f'feat_path: {feat_path}, feat_dim: {feat_dim}')
                else:
                    meta_data[meta_key] = torch.zeros(feat_dim, dtype=torch.float32)
            return frames, label, index, meta_data
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        return len(self._path_to_videos)
