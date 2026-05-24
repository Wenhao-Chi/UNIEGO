# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from functools import partial

from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import trunc_normal_

from .build import MODEL_REGISTRY
from .vit import VisionTransformer, _conv_filter, default_cfgs


PROXY_DISTILL_DIM_MAP = {
    "exo_rgb": 768,
    "exo_skl": 256,
    "exo_siglip":1152,
    "ego_siglip":1152,
    "exo_skego": 512,
    "exo_depth":1024,
    "ego_depth":1024,
    "exo_dino":1024,
    "ego_dino":1024,
}


def _normalize_exo_modalities(exo_modality):
    if isinstance(exo_modality, str):
        return [exo_modality] if exo_modality else []
    return [modality for modality in exo_modality if modality]


def _resolve_proxy_modalities(exo_modality):
    modalities = _normalize_exo_modalities(exo_modality)
    if not modalities:
        raise ValueError("pi_proxy_vit requires at least one exo modality.")

    for modality in modalities:
        if modality not in PROXY_DISTILL_DIM_MAP:
            raise KeyError(
                f"Unsupported proxy exo modality '{modality}'. "
                f"Available modalities: {list(PROXY_DISTILL_DIM_MAP.keys())}"
            )

    return list(dict.fromkeys(modalities))

    
def _resolve_single_proxy_modality(exo_modality):
    modalities = _resolve_proxy_modalities(exo_modality)
    if len(modalities) != 1:
        raise ValueError(
            "Single-projection proxy path expects exactly one modality, "
            f"but got {modalities}"
        )
    return modalities[0]


class VisionTransformerProxy(VisionTransformer):
    """
    Proxy ViT for feature distillation.
    Reuses the current ViT backbone/head and only adds a modality-specific
    projection from CLS token to teacher feature space.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        num_frames=8,
        attention_type='divided_space_time',
        dropout=0.,
        training_mode='',
        exo_modality='',
        same_dim_proxy_proj=False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            hybrid_backbone=hybrid_backbone,
            norm_layer=norm_layer,
            num_frames=num_frames,
            attention_type=attention_type,
            dropout=dropout,
            training_mode='basic',
            exo_modality=exo_modality,
        )

        self.training_mode = training_mode
        self.exo_modality = exo_modality
        self.proxy_modality = None
        self.proxy_modalities = []
        self.proxy_distill_dim = None
        self.proj_layer = None
        self.proj_layers = None
        self.same_dim_proxy_proj = same_dim_proxy_proj

        if self.training_mode == 'dist':
            if self.same_dim_proxy_proj:
                self.proxy_distill_dim = embed_dim
                self.proxy_modalities = _normalize_exo_modalities(self.exo_modality)
            else:
                self.proxy_modalities = _resolve_proxy_modalities(self.exo_modality)
                if len(self.proxy_modalities) == 1:
                    self.proxy_modality = _resolve_single_proxy_modality(self.exo_modality)
                    self.proxy_distill_dim = PROXY_DISTILL_DIM_MAP[self.proxy_modality]
                else:
                    self.proj_layers = nn.ModuleDict()
                    for modality in self.proxy_modalities:
                        proj_layer = nn.Linear(embed_dim, PROXY_DISTILL_DIM_MAP[modality])
                        trunc_normal_(proj_layer.weight, std=.02)
                        nn.init.constant_(proj_layer.bias, 0)
                        self.proj_layers[modality] = proj_layer

            if self.proj_layers is None:
                self.proj_layer = nn.Linear(embed_dim, self.proxy_distill_dim)
                trunc_normal_(self.proj_layer.weight, std=.02)
                nn.init.constant_(self.proj_layer.bias, 0)

    def forward(self, x, exo=None):
        x_all = self.forward_features(x)
        x_cls = x_all[:, 0]
        x_proj = x_cls
        x_proj_dict = None

        if self.training_mode == 'dist':
            if self.proj_layers is not None:
                x_proj_dict = {
                    modality: proj_layer(x_cls)
                    for modality, proj_layer in self.proj_layers.items()
                }
            else:
                if self.proj_layer is None:
                    raise RuntimeError("Proxy distillation projection layer was not initialized.")
                x_proj = self.proj_layer(x_cls)

        x_logits = self.head(x_cls)
        if x_proj_dict is not None:
            return x_logits, x_cls, x_proj_dict, None
        return x_logits, x_proj, x_cls, None


@MODEL_REGISTRY.register()
class pi_proxy_vit(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(pi_proxy_vit, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.model = VisionTransformerProxy(
            img_size=cfg.DATA.TRAIN_CROP_SIZE,
            num_classes=cfg.MODEL.NUM_CLASSES,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            exo_modality=cfg.EXO_MODALITY,
            training_mode=cfg.TRAINING_MODE,
            same_dim_proxy_proj=(cfg.GENERATION == 'proxy_gen2'),
            **kwargs,
        )

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(
                self.model,
                num_classes=self.model.num_classes,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=_conv_filter,
                img_size=cfg.DATA.TRAIN_CROP_SIZE,
                num_patches=self.num_patches,
                attention_type=self.attention_type,
                pretrained_model=pretrained_model,
            )

    def forward(self, x, exo=None):
        x = self.model(x, exo=exo)
        return x
