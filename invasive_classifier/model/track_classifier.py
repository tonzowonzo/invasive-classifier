# invasive_classifier/model/dinov3_track_classifier.py
# PyTorch model: frozen DINOv3 backbone + temporal head for track classification.
# Expects input shape [B, T, C, H, W] (e.g., crops sampled along a track).

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# We’ll prefer timm’s DINOv3 weights (simple + reliable).
# If you want to load a local DINOv3 repo checkpoint, see notes at the bottom.
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


@dataclass
class DinoV3Config:
    num_classes: int
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m"  # good default
    dropout: float = 0.2
    temporal: str = "meanmax"   # ["meanmax", "gru"]
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0  # set >0 to fine-tune last N transformer blocks
    feature_dim_override: Optional[int] = None  # leave None unless using a custom backbone


class TemporalMeanMaxHead(nn.Module):
    """Pools over time with mean+max and applies a small MLP."""
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        mean = feats.mean(dim=1)
        mx = feats.max(dim=1).values
        x = torch.cat([mean, mx], dim=1)
        return self.classifier(x)


class TemporalGRUHead(nn.Module):
    """A tiny GRU over time, then FC."""
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        out, h = self.gru(feats)        # h: [1, B, H]
        last = h[-1]                    # [B, H]
        return self.classifier(last)


class DinoV3TrackClassifier(nn.Module):
    """
    Wrapper that:
      - extracts per-frame features with a (frozen) DINOv3 ViT
      - pools temporally with mean+max (default) or a tiny GRU
    Forward expects x: [B, T, C, H, W] -> logits: [B, num_classes]
    """
    def __init__(self, cfg: DinoV3Config):
        super().__init__()
        self.cfg = cfg

        if not _HAS_TIMM:
            raise ImportError(
                "timm is required. `pip install timm` (>=0.9 recommended). "
                "Alternatively, adapt this file to load your local dinov3 repo checkpoint."
            )

        # Create DINOv3 backbone (no classifier head)
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=True, num_classes=0)
        feat_dim = cfg.feature_dim_override or getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # Fallback: try common attribute names
            feat_dim = getattr(self.backbone, "feature_info", [{}])[-1].get("num_chs", 768)

        # Freeze or selectively unfreeze
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Optionally unfreeze last N transformer blocks for light FT
        if cfg.unfreeze_last_n_blocks > 0:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                # some timm vit variants call them "stages" or similar
                raise ValueError("Could not access transformer blocks on this backbone to unfreeze.")
            for b in blocks[-cfg.unfreeze_last_n_blocks:]:
                for p in b.parameters():
                    p.requires_grad = True

        # Temporal head
        if cfg.temporal.lower() == "gru":
            self.temporal = TemporalGRUHead(feat_dim, cfg.num_classes, hidden=512, dropout=cfg.dropout)
        else:
            self.temporal = TemporalMeanMaxHead(feat_dim, cfg.num_classes, dropout=cfg.dropout)

        # simple input norm (optional; you can also normalize in your dataset/transforms)
        # Values are intentionally light since thermal has different stats than ImageNet.
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 1, 3, 1, 1))

    @torch.no_grad()
    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B*T, C, H, W]
        returns: [B*T, D]
        """
        # DINOv3 expects 3-channel inputs. Your dataset already replicates grayscale -> 3ch.
        feats = self.backbone(frames)
        # `timm` DINOv3 returns a feature vector (no CLS head) when num_classes=0.
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] in [0,1] float
        """
        B, T, C, H, W = x.shape

        # normalize lightly (optional)
        x = (x - self.mean) / (self.std + 1e-6)

        # Flatten time into batch and extract features
        x = x.view(B * T, C, H, W)
        with torch.set_grad_enabled(self._backbone_requires_grad()):
            feats = self._extract_frame_features(x)  # [B*T, D]
        D = feats.shape[-1]
        feats = feats.view(B, T, D)                 # [B, T, D]

        # Temporal pooling/classification
        logits = self.temporal(feats)               # [B, num_classes]
        return logits

    def _backbone_requires_grad(self) -> bool:
        # True if any parameter in the backbone requires grad (e.g., when unfreezing last N blocks)
        for p in self.backbone.parameters():
            if p.requires_grad:
                return True
        return False


# ---------- convenience factory ----------
def build_dinov3_track_classifier(
    num_classes: int,
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m",
    temporal: str = "meanmax",
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    dropout: float = 0.2,
) -> DinoV3TrackClassifier:
    cfg = DinoV3Config(
        num_classes=num_classes,
        backbone_name=backbone_name,
        temporal=temporal,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        dropout=dropout,
    )
    return DinoV3TrackClassifier(cfg)
