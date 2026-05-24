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


def _normalize_proxy_candidates(proxy_candidates):
    if isinstance(proxy_candidates, str):
        return [proxy_candidates] if proxy_candidates else []
    return [candidate for candidate in proxy_candidates if candidate]


@DATASET_REGISTRY.register()
class Egoexo_fitness_proxy_gen2(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], "Split '{}' not supported for EgoExo-Fitness".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        else:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing EgoExo-Fitness proxy-gen2 %s...", mode)
        self._construct_loader()

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.proxy_feat_dim = 768
        self.proxy_candidate_root = cfg.DATA.PROXY_CANDIDATE_ROOT
        self.proxy_candidates = _normalize_proxy_candidates(cfg.DATA.PROXY_CANDIDATES)
        if not self.proxy_candidates:
            raise ValueError(
                "egoexo_fitness_proxy_gen2 requires cfg.DATA.PROXY_CANDIDATES to contain at least one candidate."
            )

        self.candidate_configs = {}
        for candidate_name in self.proxy_candidates:
            if os.path.isabs(candidate_name):
                candidate_dir = candidate_name
            else:
                candidate_dir = os.path.join(self.proxy_candidate_root, candidate_name)
            candidate_key = candidate_name

            if not os.path.exists(candidate_dir):
                logger.warning("Proxy candidate directory not found: %s", candidate_dir)

            self.candidate_configs[candidate_key] = {
                "feat_dir": os.path.join(candidate_dir, cfg.DATA.PROXY_FEATS_DIR),
                "logits_dir": os.path.join(candidate_dir, cfg.DATA.PROXY_LOGITS_DIR),
            }

    def _load_proxy_feature(self, feat_path):
        if not os.path.exists(feat_path):
            return torch.zeros(self.proxy_feat_dim, dtype=torch.float32)

        feature = np.asarray(np.load(feat_path))
        if feature.ndim == 1:
            if feature.shape[0] != self.proxy_feat_dim:
                logger.warning(
                    "Proxy feature dim mismatch for %s: expected %d, got %s. Falling back to zeros.",
                    feat_path,
                    self.proxy_feat_dim,
                    feature.shape,
                )
                return torch.zeros(self.proxy_feat_dim, dtype=torch.float32)
            return torch.tensor(feature, dtype=torch.float32)

        if feature.ndim == 2 and feature.shape[0] == self.proxy_feat_dim and feature.shape[1] != self.proxy_feat_dim:
            feature = feature.transpose(1, 0)

        if feature.shape[-1] != self.proxy_feat_dim:
            if feature.size % self.proxy_feat_dim != 0:
                logger.warning(
                    "Proxy feature shape mismatch for %s: expected last dim %d, got %s. Falling back to zeros.",
                    feat_path,
                    self.proxy_feat_dim,
                    feature.shape,
                )
                return torch.zeros(self.proxy_feat_dim, dtype=torch.float32)
            feature = feature.reshape(-1, self.proxy_feat_dim)
        else:
            feature = feature.reshape(-1, self.proxy_feat_dim)

        if feature.shape[0] == 0:
            return torch.zeros(self.proxy_feat_dim, dtype=torch.float32)

        feature = feature.mean(axis=0)
        return torch.tensor(feature, dtype=torch.float32)

    def _load_proxy_logits(self, logits_path):
        if not os.path.exists(logits_path):
            return torch.zeros(self.num_classes, dtype=torch.float32)

        logits = np.asarray(np.load(logits_path)).reshape(-1)
        if logits.shape[0] != self.num_classes:
            logger.warning(
                "Proxy logits dim mismatch for %s: expected %d, got %s. Falling back to zeros.",
                logits_path,
                self.num_classes,
                logits.shape,
            )
            return torch.zeros(self.num_classes, dtype=torch.float32)

        return torch.tensor(logits, dtype=torch.float32)

    def _construct_loader(self):
        split_name = utils.get_split_name(self.cfg, self.mode)
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(split_name))
        assert PathManager.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as file_obj:
            for clip_idx, path_label in enumerate(file_obj.read().splitlines()):
                assert len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)) == 2
                path, label = path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, path))
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert len(self._path_to_videos) > 0, "Failed to load EgoExo-Fitness split {} from {}".format(
            split_name, path_to_file
        )
        logger.info(
            "Constructing EgoExo-Fitness proxy-gen2 dataloader (size: %d) from %s",
            len(self._path_to_videos),
            path_to_file,
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
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))
        elif self.mode in ["test"]:
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

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
            except Exception as err:
                logger.info("Failed to load video from %s with error %s", self._path_to_videos[index], err)

            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx %d from %s; trial %d",
                    index,
                    self._path_to_videos[index],
                    i_try,
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
            )

            if frames is None:
                logger.warning(
                    "Failed to decode video idx %d from %s; trial %d",
                    index,
                    self._path_to_videos[index],
                    i_try,
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            label = self._labels[index]
            frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
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

            if self.cfg.MODEL.ARCH not in ["vit"]:
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                frames = torch.index_select(
                    frames,
                    1,
                    torch.linspace(0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES).long(),
                )

            meta_data = {}
            meta_data["filename"] = self._path_to_videos[index].split("/")[-1].split(".")[0]
            filename = os.path.basename(meta_data["filename"]) + ".npy"

            for candidate_name, candidate_cfg in self.candidate_configs.items():
                feat_path = os.path.join(candidate_cfg["feat_dir"], filename)
                logits_path = os.path.join(candidate_cfg["logits_dir"], filename)
                meta_data[f"{candidate_name}_feats"] = self._load_proxy_feature(feat_path)
                meta_data[f"{candidate_name}_logits"] = self._load_proxy_logits(logits_path)
            return frames, label, index, meta_data

        raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))

    def __len__(self):
        return len(self._path_to_videos)
