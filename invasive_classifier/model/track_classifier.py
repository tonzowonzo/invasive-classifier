# invasive_classifier/model/object_detector.py
import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

try:
    import timm

    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


@dataclass
class DetectorConfig:
    """Configuration for the DinoV3 Object Detector."""

    num_classes: int
    # --- CHANGE: Added img_size ---
    img_size: int = 224
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m"
    in_channels: int = 1
    dropout: float = 0.1
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 0
    feature_dim_override: Optional[int] = None


class SimpleDetectionHead(nn.Module):
    """
    A simple detection head that takes backbone features and predicts
    class logits and bounding boxes for each feature token.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.class_head = nn.Linear(in_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 4),
        )

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_logits = self.class_head(feats)
        pred_boxes = self.box_head(feats).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class DinoV3ObjectDetector(nn.Module):
    """
    An object detector using a DINOv3 Vision Transformer backbone, adapted for
    flexible input sizes.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        if not _HAS_TIMM:
            raise ImportError("timm is required (`pip install timm`).")

        # --- CHANGE: Pass img_size to timm.create_model ---
        # timm will automatically handle positional embedding interpolation.
        self.backbone = timm.create_model(
            cfg.backbone_name, pretrained=True, num_classes=0, img_size=cfg.img_size
        )
        print(f"✅ Model configured for input size: {cfg.img_size}x{cfg.img_size}")

        feat_dim = cfg.feature_dim_override or getattr(
            self.backbone, "num_features", 768
        )
        self._adapt_input_channels(cfg.in_channels)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.unfreeze_last_n_blocks > 0:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("Backbone has no 'blocks' to unfreeze.")
            for b in blocks[-cfg.unfreeze_last_n_blocks :]:
                for p in b.parameters():
                    p.requires_grad = True

        self.detection_head = SimpleDetectionHead(feat_dim, cfg.num_classes)

    def _adapt_input_channels(self, in_channels: int):
        if in_channels == 3:
            return

        # This logic remains the same
        patch_embed = self.backbone.patch_embed
        original_conv = patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(
                dim=1, keepdim=True
            ).repeat(1, in_channels, 1, 1)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
        patch_embed.proj = new_conv
        print(f"✅ Backbone input layer adapted for {in_channels} channels.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone.forward_features(x)
        patch_tokens = feats[:, 1:, :]
        return self.detection_head(patch_tokens)


def build_dinov3_detector(
    num_classes: int,
    # --- CHANGE: Added img_size parameter ---
    img_size: int = 224,
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m",
    in_channels: int = 1,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
    local_checkpoint: Optional[str] = None,
) -> DinoV3ObjectDetector:
    """
    Helper factory function to build the DinoV3ObjectDetector.
    """
    cfg = DetectorConfig(
        num_classes=num_classes,
        img_size=img_size,  # Pass it to the config
        backbone_name=backbone_name,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    model = DinoV3ObjectDetector(cfg)

    if local_checkpoint and os.path.isfile(local_checkpoint):
        sd = torch.load(local_checkpoint, map_location="cpu")
        # Pop positional embeddings if they cause a mismatch, timm handles it
        if (
            "pos_embed" in sd
            and sd["pos_embed"].shape != model.backbone.pos_embed.shape
        ):
            print("Removing mismatched positional embedding from checkpoint.")
            sd.pop("pos_embed")

        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        print(f"[DINOv3] Loaded local checkpoint: {local_checkpoint}")
        print(f"[DINOv3] missing={len(missing)} unexpected={len(unexpected)}")

    return model
