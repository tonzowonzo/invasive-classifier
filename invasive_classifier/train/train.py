import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from invasive_classifier.config import get_config, get_class_mapping
from invasive_classifier.data.data_utils import (
    apply_class_mapping,
    load_split_ids,
    scan_labels_in_split,
)
from invasive_classifier.train.train_utils import evaluate, log_debug_plots
from invasive_classifier.model.track_classifier import build_dinov3_detector
from invasive_classifier.data.dataset_invasive import make_loader


def set_seed(seed=1337):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    CFG = get_config()
    set_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CFG["out"], exist_ok=True)
    os.makedirs(os.path.join(CFG["out"], "debug_plots"), exist_ok=True)

    # --- 1. Setup Labels and Classes ---
    train_ids, eval_ids = load_split_ids(CFG["splits_path"])
    indiv_dir = os.path.join(CFG["root"], "individual-metadata")
    train_cnt_raw = scan_labels_in_split(indiv_dir, train_ids)
    class_mapping = get_class_mapping()
    train_cnt = apply_class_mapping(train_cnt_raw, class_mapping)
    print("[labels] Applied class mapping to consolidate classes.")

    active_classes = sorted(list(train_cnt.keys()))
    label_map = {c: i for i, c in enumerate(active_classes)}
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    print(f"[labels] Active classes after merging: {len(label_map)}")
    print(f"[labels] active: {class_names}")

    # --- 2. Setup Dataloaders ---
    train_loader, _ = make_loader(
        CFG["root"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        num_samples=CFG["num_samples"],
        size=CFG["size"],
        allowed_clip_ids=train_ids,
        normalize="imagenet",
        label_map=label_map,
        class_mapping=class_mapping,
    )

    eval_loader, _ = make_loader(
        CFG["root"],
        batch_size=max(1, CFG["batch_size"] // 2),
        num_workers=CFG["num_workers"],
        num_samples=CFG["num_samples"],
        size=CFG["size"],
        allowed_clip_ids=eval_ids,
        normalize="imagenet",
        label_map=label_map,
        class_mapping=class_mapping,
    )
    print(
        f"[data] train tracks: {len(train_loader.dataset)} | eval tracks: {len(eval_loader.dataset)}"
    )

    if CFG["use_weighted_sampler"]:
        labels_idx = [it["label_id"] for it in train_loader.dataset.items]
        cnt = Counter(labels_idx)
        if len(cnt) > 1:
            sample_w = [
                1.0 / cnt.get(it["label_id"], 1e-9) for it in train_loader.dataset.items
            ]
            sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=CFG["batch_size"],
                sampler=sampler,
                num_workers=CFG["num_workers"],
                pin_memory=True,
            )

    # --- 3. Setup Model, Optimizer, and Loss ---
    # --- FIX: Pass the 'size' from the config to the model builder ---
    model = build_dinov3_detector(
        num_classes=len(label_map),
        img_size=CFG["size"],
        backbone_name="vit_base_patch16_224",
        in_channels=3,
        local_checkpoint=CFG["local_ckpt"],
        unfreeze_last_n_blocks=CFG["unfreeze_last_n_blocks"],
    ).to(device)

    back_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.detection_head.parameters() if p.requires_grad]
    groups = [
        {"params": back_params, "lr": CFG["lr_backbone"]},
        {"params": head_params, "lr": CFG["lr_head"]},
    ]
    optimizer = optim.AdamW(groups, weight_decay=CFG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    writer = SummaryWriter(CFG["logdir"])
    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_acc, seen = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}", ncols=120)
        for xb, yb, meta in pbar:
            xb, yb = xb.to(device).float(), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                B, T, C, H, W = xb.shape
                xb_frames = xb.view(B * T, C, H, W)
                predictions = model(xb_frames)
                pred_logits = predictions["pred_logits"]
                clip_logits = (
                    pred_logits.view(B, T, pred_logits.shape[1], -1)
                    .max(dim=2)
                    .values.max(dim=1)
                    .values
                )
                clip_logits = clip_logits[:, :-1]
                loss = criterion(clip_logits, yb)

            scaler.scale(loss).backward()
            if CFG["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            bs = yb.size(0)
            ep_loss += loss.item() * bs
            ep_acc += (clip_logits.argmax(1) == yb).sum().item()
            seen += bs

            lrs = [g["lr"] for g in optimizer.param_groups]
            pbar.set_postfix(
                loss=f"{ep_loss/max(1,seen):.4f}",
                acc=f"{ep_acc/max(1,seen):.3f}",
                lr="/".join(f"{lr:.2e}" for lr in lrs),
            )

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)

            if global_step > 0 and global_step % CFG["debug_every_steps"] == 0:
                log_debug_plots(
                    writer,
                    xb,
                    yb,
                    clip_logits,
                    meta,
                    class_names,
                    train_loader,
                    epoch,
                    global_step,
                    os.path.join(CFG["out"], "debug_plots"),
                )

            global_step += 1

        train_loss = ep_loss / max(1, seen)
        train_acc = ep_acc / max(1, seen)
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch_acc", train_acc, epoch)
        sched.step()

        eval_loss, eval_f1 = evaluate(
            model, eval_loader, device, label_map, writer, global_step, CFG["out"]
        )
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | eval_loss={eval_loss:.4f} macro_f1={eval_f1:.3f} | time {time.time()-t0:.1f}s"
        )

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(
                {"model": model.state_dict()}, os.path.join(CFG["out"], "best.ckpt")
            )
            print(f"✓ New best saved (macro_f1={best_f1:.3f})")

    writer.close()


if __name__ == "__main__":
    main()
