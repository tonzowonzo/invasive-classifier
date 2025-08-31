# invasive_classifier/utils/plots.py
import os
from typing import Dict, List, Tuple, Optional, Sequence

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------- Normalization helpers ----------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def denorm_img(x: torch.Tensor,
               mean: Sequence[float] = IMAGENET_MEAN,
               std: Sequence[float]  = IMAGENET_STD) -> torch.Tensor:
    """
    x: [C,H,W] or [T,C,H,W] (float, ImageNet-normalized). Returns [0,1].
    If your tensor is already [0,1], pass mean=(0,0,0), std=(1,1,1).
    """
    if x.ndim == 3:
        x = x.clone()
        for c in range(min(3, x.size(0))):
            x[c] = x[c] * std[c] + mean[c]
        return x.clamp(0, 1)
    elif x.ndim == 4:
        x = x.clone()
        for c in range(min(3, x.size(1))):
            x[:, c] = x[:, c] * std[c] + mean[c]
        return x.clamp(0, 1)
    else:
        raise ValueError(f"denorm_img expects [C,H,W] or [T,C,H,W], got {x.shape}")

def to_gray01(x: torch.Tensor) -> torch.Tensor:
    """
    x: [C,H,W] or [T,C,H,W] in [0,1] with replicated grayscale channels.
    Returns a single-channel [H,W] or [T,H,W] in [0,1].
    """
    if x.ndim == 3:
        return x.mean(0, keepdim=False)
    elif x.ndim == 4:
        return x.mean(1, keepdim=False)
    else:
        raise ValueError(f"to_gray01 expects [C,H,W] or [T,C,H,W], got {x.shape}")

# ---------------------- Simple crops grid ----------------------

def show_track_crops(track_tensor: torch.Tensor, label: str = "",
                     max_cols: int = 8, denorm: bool = True,
                     mean: Sequence[float] = IMAGENET_MEAN,
                     std: Sequence[float]  = IMAGENET_STD):
    """
    Build a Figure showing crops from one track.
    track_tensor: [T, C, H, W] (normalized or [0,1])
    """
    T, C, H, W = track_tensor.shape
    x = denorm_img(track_tensor, mean, std) if denorm else track_tensor
    g = to_gray01(x)  # thermal is effectively grayscale

    cols = max(1, min(max_cols, T))
    rows = int(np.ceil(T / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < T:
            ax.imshow(g[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"t{i}", fontsize=8)
    if label:
        fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    return fig

# ---------------------- Video sampling with boxes ----------------------

def show_video_with_boxes(video_path: str,
                          boxes: Dict[int, Tuple[int,int,int,int]],
                          frames: Optional[List[int]] = None,
                          step: int = 5,
                          label: str = ""):
    """
    Sample raw frames and draw a single track's boxes on top.
    Returns a Figure (does not show).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None:
        frames = list(range(0, total_frames, max(1, step)))
    frames = frames[:12]  # compact panel

    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames)*2.2, 2.2))
    if len(frames) == 1:
        axes = [axes]

    for ax, f in zip(axes, frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            ax.axis("off"); continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if f in boxes:
            x1,y1,x2,y2 = boxes[f]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        ax.imshow(frame); ax.axis("off"); ax.set_title(f"f{f}", fontsize=8)
    if label:
        fig.suptitle(f"{os.path.basename(video_path)} | {label}", fontsize=11)
    fig.tight_layout()
    cap.release()
    return fig

# ---------------------- Curves ----------------------

def _motion_curve(track_denorm: torch.Tensor) -> np.ndarray:
    """
    Motion: mean |frame[t]-frame[t-1]| across pixels.
    track_denorm: [T,C,H,W] -> internally averaged to grayscale.
    Returns [T] with a zero at t=0 for alignment.
    """
    g = to_gray01(track_denorm)  # [T,H,W]
    T = g.size(0)
    if T <= 1:
        return np.zeros(1, dtype=np.float32)
    diffs = (g[1:] - g[:-1]).abs().mean(dim=(1,2)).cpu().numpy()
    return np.concatenate([[0.0], diffs])  # align t indices

def _bbox_area_curve(boxes: Dict[int, Tuple[int,int,int,int]],
                     frames: Sequence[int]) -> np.ndarray:
    areas = []
    for f in frames:
        b = boxes.get(int(f))
        if b is None:
            areas.append(np.nan)
        else:
            x1,y1,x2,y2 = b
            areas.append(max(0, x2-x1) * max(0, y2-y1))
    return np.array(areas, dtype=np.float32)

# ---------------------- Rich debug panel ----------------------

def _shorten(name: str, max_chars: int = 24) -> str:
    if len(name) <= max_chars:
        return name
    return name[:max_chars-1] + "…"

def track_debug_panel(track_tensor: torch.Tensor,
                      meta: Dict,
                      class_probs: Optional[np.ndarray] = None,
                      class_names: Optional[List[str]] = None,
                      video_path: Optional[str] = None,
                      boxes: Optional[Dict[int, Tuple[int,int,int,int]]] = None,
                      frames: Optional[List[int]] = None,
                      denorm: bool = True,
                      mean: Sequence[float] = IMAGENET_MEAN,
                      std: Sequence[float]  = IMAGENET_STD,
                      topk: int = 5):
    """
    A single panel for one track:
      Left (two rows): crops timeline (grayscale) + sampled raw frames w/ boxes
      Right (stack): top-k probabilities bar, motion curve, bbox-area curve
    Returns fig.
    """
    T = track_tensor.size(0)
    x_den = denorm_img(track_tensor, mean, std) if denorm else track_tensor

    # Prepare frame indices for raw panel/bbox curve
    if frames is None:
        # map 0..T-1 to approx real frames if provided
        frames = meta.get("frames", list(range(T)))
    frames = list(frames)[:max(1, min(T, 12))]

    # --- layout via GridSpec (no overlaps)
    # width ratios: crops area, raw area, right column
    fig = plt.figure(figsize=(18, 4.0), dpi=120)
    gs = GridSpec(nrows=2, ncols=3, width_ratios=[3.5, 3.5, 2.2],
                  height_ratios=[1.0, 1.0], figure=fig, wspace=0.25, hspace=0.35)

    # Row 1, Col 0: crops timeline (wrap to two lines if many)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_grid.axis("off")
    cols = min(10, max(4, T))
    rows = int(np.ceil(T / cols))
    inner_gs = GridSpec(nrows=rows, ncols=cols, figure=fig,
                        left=ax_grid.get_position(fig).x0,
                        right=ax_grid.get_position(fig).x1,
                        bottom=ax_grid.get_position(fig).y0,
                        top=ax_grid.get_position(fig).y1, wspace=0.05, hspace=0.15)
    g = to_gray01(x_den)  # [T,H,W]
    for i in range(T):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(inner_gs[r, c])
        ax.imshow(g[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t{i}", fontsize=7)
        ax.axis("off")

    # Row 2, Col 0: sampled raw frames with boxes (if provided)
    if video_path is not None and boxes is not None:
        ax_raw = fig.add_subplot(gs[1, 0])
        ax_raw.axis("off")
        # small thumbnails inline
        cap = cv2.VideoCapture(video_path)
        thumbs = min(len(frames), 10)
        sub_gs = GridSpec(nrows=1, ncols=thumbs, figure=fig,
                          left=ax_raw.get_position(fig).x0,
                          right=ax_raw.get_position(fig).x1,
                          bottom=ax_raw.get_position(fig).y0,
                          top=ax_raw.get_position(fig).y1, wspace=0.05)
        for i, f in enumerate(frames[:thumbs]):
            ax = fig.add_subplot(sub_gs[0, i])
            ax.axis("off")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if int(f) in boxes:
                x1,y1,x2,y2 = boxes[int(f)]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            ax.imshow(frame); ax.set_title(f"f{f}", fontsize=7)
        cap.release()
    else:
        ax_placeholder = fig.add_subplot(gs[1, 0])
        ax_placeholder.axis("off")
        ax_placeholder.text(0.5, 0.5, "raw frames unavailable", ha="center", va="center", fontsize=9)

    # Row 0, Col 1: Top-k probabilities
    ax_p = fig.add_subplot(gs[0, 1])
    ax_p.set_title("Top-k probs", fontsize=10)
    if class_probs is not None and class_names is not None:
        probs = np.asarray(class_probs, dtype=np.float32)
        k = int(min(topk, len(probs)))
        idx = np.argsort(-probs)[:k]
        # shorten/wrap labels
        labs = [_shorten(class_names[i], 28) for i in idx][::-1]
        vals = probs[idx][::-1]
        ax_p.barh(range(k), vals)
        ax_p.set_yticks(range(k)); ax_p.set_yticklabels(labs, fontsize=8)
        ax_p.set_xlim(0, 1.0)
        for j, v in enumerate(vals):
            ax_p.text(min(v + 0.02, 0.98), j, f"{v:.2f}", va="center", fontsize=8)
    else:
        ax_p.text(0.5, 0.5, "no probs", ha="center", va="center")
        ax_p.set_yticks([]); ax_p.set_xticks([])

    # Row 1, Col 1: Motion curve
    ax_m = fig.add_subplot(gs[1, 1])
    mot = _motion_curve(x_den)
    ax_m.plot(np.arange(len(mot)), mot, lw=1.5)
    ax_m.set_title("Motion Δ (mean|Δ|)", fontsize=10)
    ax_m.set_xlabel("t"); ax_m.grid(True, alpha=0.25)

    # Col 2 (full height): BBox area curve
    ax_a = fig.add_subplot(gs[:, 2])
    if boxes is not None:
        # prefer explicit meta frames else fallback to sorted box keys
        frames_list = meta.get("frames") or sorted(boxes.keys())
        areas = _bbox_area_curve(boxes, frames_list[:T])
        ax_a.plot(np.arange(len(areas)), areas, lw=1.7, color="tab:orange")
        ax_a.set_title("BBox area", fontsize=10)
        ax_a.set_xlabel("t"); ax_a.grid(True, alpha=0.25)
    else:
        ax_a.axis("off")
        ax_a.text(0.5, 0.5, "no boxes", ha="center", va="center")

    title = f"{meta.get('label_str','?')} | clip {meta.get('clip_id','?')} | track {meta.get('track_index','?')}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig

# ---------------------- Batch montage ----------------------

def batch_grid(xb: torch.Tensor, yb: torch.Tensor, logits: torch.Tensor,
               metas: Dict[str, List], class_names: List[str],
               max_items: int = 16, denorm: bool = True,
               mean: Sequence[float] = IMAGENET_MEAN,
               std: Sequence[float]  = IMAGENET_STD,
               correct_only: Optional[bool] = None):
    """
    Grid of first-frame crops with GT/PRED + confidence.
    """
    B, T, C, H, W = xb.shape
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = logits.argmax(1).detach().cpu().numpy()
    gts   = yb.detach().cpu().numpy()

    # filter indices
    idxs = []
    for i in range(B):
        ok = (preds[i] == gts[i])
        if correct_only is True  and not ok: continue
        if correct_only is False and ok:     continue
        idxs.append(i)
    idxs = idxs[:max_items]
    if not idxs:
        fig = plt.figure(figsize=(6,2)); ax = fig.add_subplot(111)
        ax.text(0.5,0.5,"No items for this filter", ha="center", va="center"); ax.axis("off")
        return fig

    x0 = xb[:,0]
    x0 = denorm_img(x0, mean, std) if denorm else x0
    g0 = to_gray01(x0)

    cols = min(8, len(idxs))
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.6))
    axes = np.atleast_1d(axes).flatten()

    for ax,i in zip(axes, idxs):
        ax.axis("off")
        ax.imshow(g0[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        gt = class_names[gts[i]]; pr = class_names[preds[i]]
        conf = probs[i, preds[i]]
        color = "green" if preds[i]==gts[i] else "red"
        ax.set_title(f"{pr} ({conf:.2f})\nGT: {gt}", color=color, fontsize=9)
    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig

# ---------------------- Distributions & PCA ----------------------

def plot_class_distribution(label_list: List[str], top_n: int = 30, title: str = "Class distribution"):
    from collections import Counter
    counts = Counter(label_list)
    labs, vals = zip(*counts.most_common(top_n))
    fig = plt.figure(figsize=(min(12, top_n*0.6), 4))
    ax = fig.add_subplot(111)
    ax.bar(labs, vals)
    ax.set_xticklabels(labs, rotation=90, fontsize=8)
    ax.set_title(title); ax.set_ylabel("tracks")
    fig.tight_layout()
    return fig

def prediction_histogram(preds: np.ndarray, class_names: List[str], title: str = "Predictions per class"):
    vals = np.bincount(preds, minlength=len(class_names))
    fig = plt.figure(figsize=(min(12, len(class_names)*0.6), 4))
    ax = fig.add_subplot(111)
    ax.bar(class_names, vals)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_title(title); ax.set_ylabel("#predictions")
    fig.tight_layout()
    return fig

def pca_scatter(embeddings: np.ndarray,
                targets: np.ndarray,
                class_names: List[str],
                title: str = "PCA of embeddings"):
    """
    embeddings: [N,D]  targets: [N]
    """
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    for cid in np.unique(targets):
        mask = (targets == cid)
        ax.scatter(xy[mask,0], xy[mask,1], s=8, alpha=0.6, label=_shorten(class_names[int(cid)], 20))
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return fig
