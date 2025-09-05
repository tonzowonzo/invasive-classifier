import os
import json
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from invasive_classifier.utils import plots


def macro_f1_from_counts(tp, fp, fn, eps=1e-9):
    """Calculates the macro F1 score from true/false positive/negative counts."""
    f1s = []
    for c in tp.keys():
        p = tp[c] / (tp[c] + fp[c] + eps)
        r = tp[c] / (tp[c] + fn[c] + eps)
        f1s.append(2 * p * r / (p + r + eps))
    return sum(f1s) / max(1, len(f1s))


def evaluate(model, loader, device, label_map, writer=None, step=0, save_dir=None):
    """Runs a full evaluation loop on the given model and dataloader."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n = 0.0, 0
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

    with torch.no_grad():
        for xb, yb, meta in tqdm(loader, desc="Evaluating", ncols=100):
            xb, yb = xb.to(device).float(), yb.to(device)
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


def log_debug_plots(
    writer: SummaryWriter,
    xb: torch.Tensor,
    yb: torch.Tensor,
    clip_logits: torch.Tensor,
    meta: Dict,
    class_names: list,
    train_loader,
    epoch: int,
    global_step: int,
    save_dir: str,
):
    """Logs a rich set of debug images to TensorBoard and disk."""

    def find_dataset_item(ds, clip_id, track_index, video_path):
        for it in ds.items:
            if (
                it["clip_id"] == clip_id
                and it["track_index"] == track_index
                and it["video"] == video_path
            ):
                return it
        return None

    # 1) Crops timeline
    pred_idx = int(clip_logits[0].argmax().item())
    fig1 = plots.show_track_crops(
        xb[0].detach().cpu(),
        label=f"GT: {meta['label_str'][0]} | pred: {class_names[pred_idx]}",
        denorm=True,
    )
    fig1.savefig(os.path.join(save_dir, f"crops_e{epoch}_s{global_step}.png"), dpi=120)
    writer.add_figure("debug/crops", fig1, global_step)
    plt.close(fig1)

    # 2) Batch montage: mistakes
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
        os.path.join(save_dir, f"batch_mistakes_e{epoch}_s{global_step}.png"), dpi=120
    )
    writer.add_figure("debug/batch_mistakes", fig2, global_step)
    plt.close(fig2)

    # 3) Batch montage: correct
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
        os.path.join(save_dir, f"batch_correct_e{epoch}_s{global_step}.png"), dpi=120
    )
    writer.add_figure("debug/batch_correct", fig3, global_step)
    plt.close(fig3)

    # 4) Single rich track panel
    try:
        item = find_dataset_item(
            train_loader.dataset,
            clip_id=meta["clip_id"][0],
            track_index=int(meta["track_index"][0]),
            video_path=meta["video_path"][0],
        )
        if item is not None:
            probs = torch.softmax(clip_logits[0], dim=0).detach().cpu().numpy()
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
                os.path.join(save_dir, f"track_panel_e{epoch}_s{global_step}.png"),
                dpi=120,
            )
            writer.add_figure("debug/track_panel", fig4, global_step)
            plt.close(fig4)
    except Exception as e:
        print(f"[debug] track panel failed: {e}")
