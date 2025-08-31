import os
import json
from typing import Dict, List, Tuple, Optional

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_track_crops(track_tensor: torch.Tensor, label: str = "", max_cols: int = 8):
    """
    Build a matplotlib Figure showing crops from one track.
    DOES NOT call plt.show(); caller should save/close the figure.
    Args:
        track_tensor: [T, C, H, W] in [0,1]
        label: optional title
        max_cols: images per row
    Returns:
        fig (matplotlib.figure.Figure)
    """
    t, c, h, w = track_tensor.shape
    cols = min(max_cols, t if t > 0 else 1)
    rows = int(np.ceil(t / cols)) if t > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_1d(axes).flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < t:
            img = track_tensor[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.set_title(f"f{i}", fontsize=8)
    if label:
        fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    return fig


def show_video_with_boxes(video_path: str,
                          boxes: Dict[int, Tuple[int,int,int,int]],
                          frames: Optional[List[int]] = None,
                          step: int = 5,
                          label: str = ""):
    """
    Show sampled frames from a video with the track's bounding boxes drawn.
    Args:
        video_path: path to .mp4
        boxes: dict frame_index -> (x1,y1,x2,y2)
        frames: explicit list of frames to show (if None, sample every `step`)
        step: frame stride if frames=None
        label: optional text to display
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None:
        frames = list(range(0, total_frames, step))

    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames)*3, 3))
    if len(frames) == 1:
        axes = [axes]
    for ax, f in zip(axes, frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if f in boxes:
            x1,y1,x2,y2 = boxes[f]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        ax.imshow(frame)
        ax.set_title(f"f{f}")
        ax.axis("off")
    if label:
        fig.suptitle(f"Video: {os.path.basename(video_path)} | Label: {label}")
    plt.tight_layout()
    plt.show()
    cap.release()


def plot_class_distribution(label_list: List[str], top_n: int = 30):
    """
    Plot histogram of label frequencies (for debugging imbalance).
    Args:
        label_list: list of label strings (all tracks scanned)
        top_n: number of classes to display
    """
    from collections import Counter
    counts = Counter(label_list)
    most_common = counts.most_common(top_n)
    labs, vals = zip(*most_common)
    plt.figure(figsize=(10,4))
    plt.bar(labs, vals)
    plt.xticks(rotation=90)
    plt.title(f"Top-{top_n} class distribution")
    plt.show()