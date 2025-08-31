


import torch
from invasive_classifier.model.track_classifier import build_dinov3_track_classifier
from invasive_classifier.data.dataset import make_loader


if __name__ == "__main__":
    ROOT = "/home/timiles/tim/invasive-classifier/nz_thermal_data"
    dl, label_map = make_loader(ROOT, num_samples=24, size=224, include_false_positive=True)
    num_classes = len(label_map)

    model = build_dinov3_track_classifier(
        num_classes=num_classes,
        backbone_name="vit_base_patch16_224.dinov3.lvd142m",  # good default
        temporal="meanmax",   # or "gru" if you prefer a tiny recurrent head
        freeze_backbone=True, # start frozen
        unfreeze_last_n_blocks=0,
        dropout=0.2,
    ).cuda()

    xb, yb, meta = next(iter(dl))
    logits = model(xb.cuda().float())  # xb: [B,T,C,H,W] normalized to [0,1] by dataset
    print(logits.shape)  # [B, num_classes]
