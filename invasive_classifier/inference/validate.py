import os
import json
from typing import List, Dict

import torch
import torch.nn as nn

# plotting (headless)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from invasive_classifier.data.dataset import make_loader
from invasive_classifier.model.track_classifier import build_dinov3_track_classifier


# ---------------- splits ----------------
def load_split_ids(path):
    """
    Reads a splits JSON and returns (train_ids, eval_ids).
    Accepts ints/strings or dicts with 'id'/'clip_id' fields; flattens nested containers.
    Prefers 'val' if present, else 'test'.
    """

    def extract_ids(obj):
        out = []

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
                            out.append(int(x[k]))
                            return
                        except Exception:
                            pass
                for v in x.values():
                    visit(v)
            else:
                try:
                    out.append(int(x))
                except Exception:
                    pass

        visit(obj)
        return set(out)

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
            f"Could not extract non-empty train/eval ids from split file: {path}"
        )
    return train_ids, eval_ids


# ---------------- confusion matrix helpers ----------------
def confusion_matrix(
    preds: np.ndarray, targets: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Simple confusion matrix (counts). Shape [C, C] with rows = true, cols = pred.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str = "Confusion matrix",
):
    """
    cm: counts matrix [C,C]. We will annotate with normalized row percentages.
    """
    # normalize rows to percentages (avoid div by zero)
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    norm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums > 0)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate with % (from normalized matrix)
    thresh = cm.max() * 0.5 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = norm[i, j] * 100.0
            txt = f"{val:.0f}" if val >= 1.0 else ""  # show only >=1%
            plt.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------- main validation ----------------
def main():
    # ---- CONFIG ----
    CFG = dict(
        root="/home/timiles/tim/invasive-classifier/nz_thermal_data",
        splits_path="/home/timiles/tim/invasive-classifier/nz_thermal_data/new-zealand-wildlife-thermal-imaging-splits.json",
        artifacts_dir="artifacts",
        ckpt_path="artifacts/best.ckpt",  # or "artifacts/last.ckpt"
        batch_size=8,
        num_workers=8,
        num_samples=24,
        size=224,
        local_dinov3_ckpt="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",  # same as training
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint (for model + label_map)
    ckpt = torch.load(CFG["ckpt_path"], map_location="cpu")
    label_map: Dict[str, int] = ckpt["label_map"]
    class_names = [
        k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])
    ]  # index order
    num_classes = len(class_names)

    # build model and load weights
    model = build_dinov3_track_classifier(
        num_classes=num_classes,
        backbone_name="vit_base_patch16_224",
        temporal="meanmax",
        freeze_backbone=True,  # as used in training
        unfreeze_last_n_blocks=0,
        dropout=0.2,
        local_checkpoint=CFG["local_dinov3_ckpt"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # eval split
    _, eval_ids = load_split_ids(CFG["splits_path"])

    # loader (reuse label_map so class indices align)
    eval_loader, _ = make_loader(
        CFG["root"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        num_samples=CFG["num_samples"],
        size=CFG["size"],
        include_false_positive=True,
        allowed_clip_ids=eval_ids,
        label_map=label_map,
        normalize="imagenet",
    )

    print(f"[validate] Eval items: {len(eval_loader.dataset)} | classes: {num_classes}")

    # run prediction
    all_preds, all_tgts = [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n = 0.0, 0

    with torch.no_grad():
        for xb, yb, meta in eval_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb).item()
            loss_sum += loss
            n += yb.size(0)
            preds = logits.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_tgts = np.concatenate(all_tgts, axis=0)
    avg_loss = loss_sum / max(1, n)
    acc = float((all_preds == all_tgts).mean())
    print(f"[validate] loss={avg_loss:.4f}  acc={acc:.3f}")

    # confusion matrix
    cm = confusion_matrix(all_preds, all_tgts, num_classes=num_classes)
    cm_path = os.path.join(CFG["artifacts_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, title="Confusion matrix")
    print(f"[validate] saved: {cm_path}")

    # also save raw cm + class names for later
    npy_path = os.path.join(CFG["artifacts_dir"], "confusion_matrix.npy")
    json_path = os.path.join(CFG["artifacts_dir"], "class_names.json")
    os.makedirs(CFG["artifacts_dir"], exist_ok=True)
    np.save(npy_path, cm)
    with open(json_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[validate] saved: {npy_path}, {json_path}")


if __name__ == "__main__":
    main()
