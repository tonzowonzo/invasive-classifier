import os
import time
import json
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from invasive_classifier.model.track_classifier import build_dinov3_detector
from invasive_classifier.data.dataset_invasive import make_loader
from invasive_classifier.utils import plots


# ---------------- helpers ----------------


def set_seed(seed=1337):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_class_mapping(counts: Counter, mapping: Dict[str, str]) -> Counter:
    new_counts = Counter()
    for label, count in counts.items():
        new_label = mapping.get(label, label)
        new_counts[new_label] += count
    return new_counts


def load_split_ids(path: str) -> Tuple[Set[int], Set[int]]:
    def extract_ids(obj) -> Set[int]:
        out = set()

        def visit(x):
            if x is None:
                return
            if isinstance(x, (list, tuple, set)):
                for xi in x:
                    visit(xi)
            elif isinstance(x, dict):
                for k in ("id", "clip_id", "clip", "clipId"):
                    if k in x:
                        try:
                            out.add(int(x[k]))
                            return
                        except Exception:
                            pass
                for v in x.values():
                    visit(v)
            else:
                try:
                    out.add(int(x))
                except Exception:
                    pass

        visit(obj)
        return out

    with open(path, "r") as f:
        data = json.load(f)

    def pick(keys):
        for k in keys:
            if k in data:
                ids = extract_ids(data[k])
                if ids:
                    return ids
        return set()

    train_ids = pick(["train", "train_ids", "train_clip_ids", "train_clips"])
    val_ids = pick(["val", "validation", "val_ids", "valid_ids", "validation_ids"])
    test_ids = pick(["test", "test_ids", "test_clip_ids", "test_clips"])
    eval_ids = val_ids if val_ids else test_ids
    if not train_ids or not eval_ids:
        raise ValueError(
            f"Bad splits json: train={len(train_ids)} eval={len(eval_ids)}"
        )
    return train_ids, eval_ids


def choose_label(tags: List[Dict]) -> str | None:
    if not tags:
        return None
    labs = [t.get("label") for t in tags if t.get("label")]
    if not labs:
        return None
    counts = Counter(labs).most_common()
    # --- FIX: Renamed 'l' to 'lab' ---
    tied = [lab for lab, c in counts if c == counts[0][1]]
    if len(tied) == 1:
        return tied[0]
    # --- FIX: Renamed 'l' to 'lab' ---
    conf = {
        lab: sum(t.get("confidence", 0.0) for t in tags if t.get("label") == lab)
        for lab in tied
    }
    return max(conf, key=conf.get)


def scan_labels_in_split(indiv_meta_dir: str, clip_ids: Set[int]) -> Counter:
    cnt = Counter()
    for cid in tqdm(clip_ids, desc="Scanning labels"):
        p = os.path.join(indiv_meta_dir, f"{cid}_metadata.json")
        if not os.path.exists(p):
            continue
        try:
            m = json.load(open(p))
            for tr in m.get("tracks", []):
                lab = choose_label(tr.get("tags", []))
                if lab:
                    cnt[lab] += 1
        except Exception:
            pass
    return cnt


def macro_f1_from_counts(tp, fp, fn, eps=1e-9):
    f1s = []
    for c in tp.keys():
        p = tp[c] / (tp[c] + fp[c] + eps)
        r = tp[c] / (tp[c] + fn[c] + eps)
        f1s.append(2 * p * r / (p + r + eps))
    return sum(f1s) / max(1, len(f1s))


def evaluate(model, loader, device, label_map, writer=None, step=0, save_dir=None):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n = 0.0, 0
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

    with torch.no_grad():
        for xb, yb, meta in tqdm(loader, desc="Evaluating", ncols=100):
            xb = xb.to(device).float()
            yb = yb.to(device)
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

            loss_sum += ce(clip_logits, yb).item()
            n += yb.size(0)
            pred = clip_logits.argmax(1)
            for t, p in zip(yb.tolist(), pred.tolist()):
                if t == p:
                    tp[t] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1

    avg_loss = loss_sum / max(1, n)
    mf1 = macro_f1_from_counts(tp, fp, fn)

    if writer:
        writer.add_scalar("eval/loss", avg_loss, step)
        writer.add_scalar("eval/macro_f1", mf1, step)

    if save_dir:
        inv = {v: k for k, v in label_map.items()}
        class_f1 = {}
        for c in inv:
            p = tp[c] / (tp[c] + fp[c] + 1e-9)
            r = tp[c] / (tp[c] + fn[c] + 1e-9)
            class_f1[inv[c]] = 2 * p * r / (p + r + 1e-9)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"class_f1_step{step}.json"), "w") as f:
            json.dump(class_f1, f, indent=2)
    return avg_loss, mf1


# ---------------- training ----------------


def main():
    CFG = dict(
        root="/home/timiles/tim/invasive-classifier/nz_thermal_data",
        splits_path="/home/timiles/tim/invasive-classifier/nz_thermal_data/new-zealand-wildlife-thermal-imaging-splits.json",
        local_ckpt="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        out="artifacts_detector_merged",
        logdir="runs/invasive_detector_merged",
        epochs=10,
        batch_size=4,
        num_workers=8,
        num_samples=8,
        size=224,
        lr_head=1e-4,
        lr_backbone=5e-5,
        weight_decay=0.05,
        grad_clip=1.0,
        unfreeze_last_n_blocks=0,
        use_weighted_sampler=True,
        debug_every_steps=250,
    )

    def class_names_from_map(label_map: Dict[str, int]):
        return [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    def find_dataset_item(ds, clip_id, track_index, video_path):
        for it in ds.items:
            if (
                it["clip_id"] == clip_id
                and it["track_index"] == track_index
                and it["video"] == video_path
            ):
                return it
        return None

    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CFG["out"], exist_ok=True)
    os.makedirs(os.path.join(CFG["out"], "debug_plots"), exist_ok=True)

    train_ids, eval_ids = load_split_ids(CFG["splits_path"])
    indiv_dir = os.path.join(CFG["root"], "individual-metadata")

    train_cnt_raw = scan_labels_in_split(indiv_dir, train_ids)

    class_mapping = {
        "mouse": "rodent",
        "rat": "rodent",
        "rabbit": "leporidae",
        "hare": "leporidae",
        "stoat": "mustelid",
        "ferret": "mustelid",
        "deer": "large_animal",
        "goat": "large_animal",
        "sheep": "large_animal",
        "wallaby": "large_animal",
        "pig": "large_animal",
        "sealion": "large_animal",
        "unidentified": "unidentified/other",
        "other": "unidentified/other",
        "poor tracking": "unidentified/other",
        "lizard": "unidentified/other",
        "kiwi": "bird_ground_large",
        "north island brown kiwi": "bird_ground_large",
        "little spotted kiwi": "bird_ground_large",
        "pukeko": "bird_ground_large",
        "chicken": "bird_ground_large",
        "penguin": "bird_ground_large",
        "black swan": "bird_waterfowl",
        "duck": "bird_waterfowl",
        "brown teal": "bird_waterfowl",
        "brown quail": "bird_small_other",
        "pheasant": "bird_small_other",
        "new zealand fantail": "bird_small_other",
        "song thrush": "bird_small_other",
        "california quail": "bird_small_other",
        "partridge": "bird_small_other",
        "quail": "bird_small_other",
        "morepork": "bird_small_other",
    }
    train_cnt = apply_class_mapping(train_cnt_raw, class_mapping)
    print("[labels] Applied class mapping to consolidate classes.")

    train_seen = set(train_cnt.keys())
    active_classes = sorted(list(train_seen))
    label_map = {c: i for i, c in enumerate(active_classes)}
    class_names = class_names_from_map(label_map)
    print(f"[labels] Active classes after merging: {len(label_map)}")
    print(f"[labels] active: {class_names}")

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

    model = build_dinov3_detector(
        num_classes=len(label_map),
        backbone_name="vit_base_patch16_224",
        local_checkpoint=CFG["local_ckpt"],
        unfreeze_last_n_blocks=CFG["unfreeze_last_n_blocks"],
    ).to(device)

    back_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.detection_head.parameters() if p.requires_grad]
    groups = []
    if back_params:
        groups.append({"params": back_params, "lr": CFG["lr_backbone"]})
    if head_params:
        groups.append({"params": head_params, "lr": CFG["lr_head"]})
    optimizer = optim.AdamW(groups, weight_decay=CFG["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(CFG["logdir"])
    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0
        ep_acc = 0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}", ncols=120)
        for xb, yb, meta in pbar:
            xb = xb.to(device).float()
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
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

            curr_loss = ep_loss / max(1, seen)
            curr_acc = ep_acc / max(1, seen)
            lrs = [g["lr"] for g in optimizer.param_groups]
            pbar.set_postfix(
                loss=f"{curr_loss:.4f}",
                acc=f"{curr_acc:.3f}",
                lr="/".join(f"{lr:.2e}" for lr in lrs),
            )

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                for i, g in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], global_step)

            if global_step > 0 and global_step % CFG["debug_every_steps"] == 0:
                save_dir = os.path.join(CFG["out"], "debug_plots")
                pred_idx = int(clip_logits[0].argmax().item())
                fig1 = plots.show_track_crops(
                    xb[0].detach().cpu(),
                    label=f"GT: {meta['label_str'][0]} | pred: {class_names[pred_idx]}",
                    denorm=True,
                )
                fig1.savefig(
                    os.path.join(save_dir, f"crops_e{epoch}_s{global_step}.png"),
                    dpi=120,
                )
                writer.add_figure("debug/crops", fig1, global_step)
                plt.close(fig1)

                fig2 = plots.batch_grid(
                    xb.detach().cpu(),
                    yb.detach().cpu(),
                    clip_logits.detach().cpu(),
                    meta,
                    class_names,
                    max_items=16,
                    denorm=True,
                    correct_only=False,
                )
                fig2.savefig(
                    os.path.join(
                        save_dir, f"batch_mistakes_e{epoch}_s{global_step}.png"
                    ),
                    dpi=120,
                )
                writer.add_figure("debug/batch_mistakes", fig2, global_step)
                plt.close(fig2)

                fig3 = plots.batch_grid(
                    xb.detach().cpu(),
                    yb.detach().cpu(),
                    clip_logits.detach().cpu(),
                    meta,
                    class_names,
                    max_items=16,
                    denorm=True,
                    correct_only=True,
                )
                fig3.savefig(
                    os.path.join(
                        save_dir, f"batch_correct_e{epoch}_s{global_step}.png"
                    ),
                    dpi=120,
                )
                writer.add_figure("debug/batch_correct", fig3, global_step)
                plt.close(fig3)

                try:
                    item = find_dataset_item(
                        train_loader.dataset,
                        clip_id=meta["clip_id"][0],
                        track_index=int(meta["track_index"][0]),
                        video_path=meta["video_path"][0],
                    )
                    if item is not None:
                        probs = (
                            torch.softmax(clip_logits[0], dim=0).detach().cpu().numpy()
                        )
                        fig4 = plots.track_debug_panel(
                            xb[0].detach().cpu(),
                            meta={
                                "label_str": meta["label_str"][0],
                                "clip_id": int(meta["clip_id"][0]),
                                "track_index": int(meta["track_index"][0]),
                                "frames": item.get("frames", None),
                            },
                            class_probs=probs,
                            class_names=class_names,
                            video_path=item["video"],
                            boxes=item["boxes"],
                            denorm=True,
                        )
                        fig4.savefig(
                            os.path.join(
                                save_dir, f"track_panel_e{epoch}_s{global_step}.png"
                            ),
                            dpi=120,
                        )
                        writer.add_figure("debug/track_panel", fig4, global_step)
                        plt.close(fig4)
                except Exception as e:
                    print(f"[debug] track panel failed: {e}")
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
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} "
            f"| eval_loss={eval_loss:.4f} macro_f1={eval_f1:.3f} | time {time.time()-t0:.1f}s"
        )

        last_path = os.path.join(CFG["out"], "last.ckpt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "best_f1": best_f1,
                "global_step": global_step,
                "epoch": epoch,
                "label_map": label_map,
                "config": CFG,
            },
            last_path,
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
