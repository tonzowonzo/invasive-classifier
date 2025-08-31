# invasive_classifier/model/track_classifier.py
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# ---------------- config ----------------
@dataclass
class DinoV3Config:
    num_classes: int
    backbone_name: str = "vit_base_patch16_224"   # plain ViT-B/16
    dropout: float = 0.2
    temporal: str = "meanmax"                     # ["meanmax","gru"]
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0
    feature_dim_override: Optional[int] = None
    local_checkpoint: Optional[str] = None        # <--- NEW


# ---------------- ckpt helpers ----------------
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "teacher", "student", "module"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise ValueError("Unrecognized checkpoint format for local DINOv3 weights.")

def _strip_prefix(sd, prefixes=("module.", "backbone.", "encoder.", "model.")):
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out

def _remap_dinov3_to_timm(sd):
    sd = _strip_prefix(sd)
    drop = ("head", "global_head", "classifier", "proj")
    remapped = {}
    for k, v in sd.items():
        if any(k.startswith(d) for d in drop):
            continue
        k = k.replace("ln_", "norm_").replace("ln.", "norm.")
        remapped[k] = v
    return remapped


# ---------------- temporal heads ----------------
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
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(feats)   # h: [1,B,H]
        return self.head(h[-1])


# ---------------- main model ----------------
class DinoV3TrackClassifier(nn.Module):
    def __init__(self, cfg: DinoV3Config):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required. `pip install timm`")
        self.cfg = cfg

        # 1) Build plain ViT-B/16 without classifier
        self.backbone = timm.create_model(cfg.backbone_name, pretrained=False, num_classes=0)

        # 2) Load local DINOv3 weights if provided; else fall back to timm pretrained
        if cfg.local_checkpoint:
            ckpt = torch.load(cfg.local_checkpoint, map_location="cpu")
            sd = _remap_dinov3_to_timm(_extract_state_dict(ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DINOv3] Loaded local checkpoint: {cfg.local_checkpoint}")
            print(f"[DINOv3] missing={len(missing)} unexpected={len(unexpected)}")
        else:
            self.backbone = timm.create_model(cfg.backbone_name, pretrained=True, num_classes=0)

        feat_dim = cfg.feature_dim_override or getattr(self.backbone, "num_features", 768)

        # 3) Freeze or partially unfreeze
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.unfreeze_last_n_blocks > 0:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("Backbone has no .blocks; cannot unfreeze last N blocks.")
            for b in blocks[-cfg.unfreeze_last_n_blocks:]:
                for p in b.parameters():
                    p.requires_grad = True

        # 4) Temporal head
        if cfg.temporal.lower() == "gru":
            self.temporal = TemporalGRUHead(feat_dim, cfg.num_classes, hidden=512, dropout=cfg.dropout)
        else:
            self.temporal = TemporalMeanMaxHead(feat_dim, cfg.num_classes, dropout=cfg.dropout)

        # light input norm (your dataset already outputs [0,1])
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1,1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.25,0.25,0.25]).view(1,1,3,1,1))

    @torch.no_grad()
    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone(frames)  # [B*T, D]

    def _backbone_requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] in [0,1]
        B, T, C, H, W = x.shape
        x = (x - self.mean) / (self.std + 1e-6)
        x = x.view(B*T, C, H, W)
        with torch.set_grad_enabled(self._backbone_requires_grad()):
            feats = self._extract_frame_features(x)   # [B*T, D]
        feats = feats.view(B, T, -1)                  # [B, T, D]
        return self.temporal(feats)                   # [B, num_classes]


# ---------------- factory ----------------
def build_dinov3_track_classifier(
    num_classes: int,
    backbone_name: str = "vit_base_patch16_224",
    temporal: str = "meanmax",
    freeze_backbone: bool = True,
    unfreeze_last_n_blocks: int = 0,
    dropout: float = 0.2,
    local_checkpoint: Optional[str] = None,  # <--- NEW
) -> DinoV3TrackClassifier:
    cfg = DinoV3Config(
        num_classes=num_classes,
        backbone_name=backbone_name,
        temporal=temporal,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        dropout=dropout,
        local_checkpoint=local_checkpoint,
    )
    return DinoV3TrackClassifier(cfg)
