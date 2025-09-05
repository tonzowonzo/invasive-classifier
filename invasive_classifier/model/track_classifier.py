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


# --- Configuration for the Detector ---
@dataclass
class DetectorConfig:
    """Configuration for the DinoV3 Object Detector."""

    num_classes: int  # Number of object classes (e.g., 1 for 'possum')
    backbone_name: str = "vit_base_patch16_224.dinov3.lvd142m"
    in_channels: int = 1  # Set to 1 for single-channel thermal images
    dropout: float = 0.1
    # Freeze the backbone by default is NOT recommended for domain transfer.
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 0
    feature_dim_override: Optional[int] = None


# --- A Simple Detection Head ---
class SimpleDetectionHead(nn.Module):
    """
    A simple detection head that takes backbone features and predicts
    class logits and bounding boxes for each feature token.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        # We add +1 to num_classes for the "no object" or background class.
        self.num_classes = num_classes
        self.class_head = nn.Linear(in_dim, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 4),  # 4 values for bbox: (cx, cy, w, h)
        )

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            feats (torch.Tensor): Features from the backbone of shape [B, N, D]
                                  where N is the number of patches/tokens.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with 'pred_logits' and 'pred_boxes'.
        """
        pred_logits = self.class_head(feats)
        pred_boxes = self.box_head(
            feats
        ).sigmoid()  # Sigmoid to keep box coords in [0, 1]
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


# --- The Main Detector Model ---
class DinoV3ObjectDetector(nn.Module):
    """
    An object detector using a DINOv3 Vision Transformer backbone, adapted for
    single-channel thermal input.
    """

    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.cfg = cfg
        if not _HAS_TIMM:
            raise ImportError("timm is required (`pip install timm`).")

        # Create backbone without its original classifier head
        self.backbone = timm.create_model(
            cfg.backbone_name, pretrained=True, num_classes=0
        )
        feat_dim = cfg.feature_dim_override or getattr(
            self.backbone, "num_features", 768
        )

        # Adapt backbone for thermal (single-channel) input
        self._adapt_input_channels(cfg.in_channels)

        # (Un)freeze backbone layers
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.unfreeze_last_n_blocks > 0:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("Backbone has no 'blocks' to unfreeze.")
            # Unfreeze the last N blocks
            for b in blocks[-cfg.unfreeze_last_n_blocks :]:
                for p in b.parameters():
                    p.requires_grad = True

        # Create the detection head
        self.detection_head = SimpleDetectionHead(feat_dim, cfg.num_classes)

    def _adapt_input_channels(self, in_channels: int):
        """
        Modifies the first layer of the ViT backbone to accept a different
        number of input channels.
        """
        if in_channels == 3:
            return  # No changes needed

        patch_embed = self.backbone.patch_embed
        original_conv = patch_embed.proj

        # Create a new Conv2d layer with the desired number of input channels
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # A common heuristic to initialize the new weights is to average the
        # original RGB weights.
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(
                dim=1, keepdim=True
            ).repeat(1, in_channels, 1, 1)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        # Replace the original patch embedding projection layer
        patch_embed.proj = new_conv
        print(f"✅ Backbone input layer adapted for {in_channels} channels.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of frames.

        Args:
            x (torch.Tensor): A batch of images, shape [B, C, H, W].
                              For thermal, C=1.

        Returns:
            Dict[str, torch.Tensor]: Predictions from the detection head.
        """
        # Get patch features from the backbone. This returns tokens,
        # including the [CLS] token at the beginning.
        # Shape: [B, num_patches + 1, feature_dim]
        feats = self.backbone.forward_features(x)

        # We typically discard the [CLS] token for detection tasks
        patch_tokens = feats[:, 1:, :]

        # Get predictions from the detection head
        return self.detection_head(patch_tokens)


# ---- Factory Function to Build the Detector ----
def build_dinov3_detector(
    num_classes: int,
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
        backbone_name=backbone_name,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    model = DinoV3ObjectDetector(cfg)

    # Optional: load local DINOv3 backbone weights if needed
    if local_checkpoint and os.path.isfile(local_checkpoint):
        sd = torch.load(local_checkpoint, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
        print(f"[DINOv3] Loaded local checkpoint: {local_checkpoint}")
        print(f"[DINOv3] missing={len(missing)} unexpected={len(unexpected)}")
    return model
