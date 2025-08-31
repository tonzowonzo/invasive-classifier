import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


@dataclass
class DinoV3Config:
    num_classes: int
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m"
    dropout: float = 0.2
    temporal: str = "meanmax"   # ["meanmax", "gru"]
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0
    feature_dim_override: Optional[int] = None
    # keep inputs as-is (dataset handles normalization)
    normalize_in_model: bool = False
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


class TemporalMeanMaxHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        mean = feats.mean(dim=1)
        mx = feats.max(dim=1).values
        return self.classifier(torch.cat([mean, mx], dim=1))


class TemporalGRUHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(feats)        # h: [1, B, H]
        return self.classifier(h[-1]) # [B, H] -> [B, C]


class DinoV3TrackClassifier(nn.Module):
    def __init__(self, cfg: DinoV3Config):
        super().__init__()
        self.cfg = cfg
        if not _HAS_TIMM:
            raise ImportError("timm is required (`pip install timm`).")

        # backbone without classifier head
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=True, num_classes=0)
        feat_dim = cfg.feature_dim_override or getattr(self.backbone, "num_features", 768)

        # (un)freeze
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.unfreeze_last_n_blocks > 0:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("Backbone has no 'blocks' to unfreeze.")
            for b in blocks[-cfg.unfreeze_last_n_blocks:]:
                for p in b.parameters():
                    p.requires_grad = True

        # temporal head
        if cfg.temporal.lower() == "gru":
            self.temporal = TemporalGRUHead(feat_dim, cfg.num_classes, hidden=512, dropout=cfg.dropout)
        else:
            self.temporal = TemporalMeanMaxHead(feat_dim, cfg.num_classes, dropout=cfg.dropout)

        # optional in-model normalization (off by default)
        if cfg.normalize_in_model:
            self.register_buffer("mean", torch.tensor(cfg.norm_mean).view(1, 1, 3, 1, 1))
            self.register_buffer("std",  torch.tensor(cfg.norm_std).view(1, 1, 3, 1, 1))
        else:
            self.mean = None
            self.std = None

    @torch.no_grad()
    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone(frames)  # [N, D]

    def _backbone_requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W] already normalized by dataset
        B, T, C, H, W = x.shape
        if (self.mean is not None) and (self.std is not None):
            x = (x - self.mean) / (self.std + 1e-6)

        x = x.view(B * T, C, H, W)
        with torch.set_grad_enabled(self._backbone_requires_grad()):
            feats = self._extract_frame_features(x)  # [B*T, D]
        feats = feats.view(B, T, -1)                 # [B, T, D]
        return self.temporal(feats)                  # [B, num_classes]


# ---- factory ----
def build_dinov3_track_classifier(
    num_classes: int,
    backbone_name: str = "vit_base_patch16_224",
    temporal: str = "meanmax",
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    dropout: float = 0.2,
    local_checkpoint: Optional[str] = None,
    normalize_in_model: bool = False,
) -> DinoV3TrackClassifier:
    cfg = DinoV3Config(
        num_classes=num_classes,
        backbone_name=backbone_name,
        temporal=temporal,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        dropout=dropout,
        normalize_in_model=normalize_in_model,
    )
    model = DinoV3TrackClassifier(cfg)

    # optional local DINOv3 backbone weights
    if local_checkpoint and os.path.isfile(local_checkpoint):
        sd = torch.load(local_checkpoint, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        print(f"[DINOv3] Loaded local checkpoint: {local_checkpoint}")
        print(f"[DINOv3] missing={len(missing)} unexpected={len(unexpected)}")

    return model