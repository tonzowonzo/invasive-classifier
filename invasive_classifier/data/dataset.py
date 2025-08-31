# invasive_classifier/data/dataset.py
import os, json, glob
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# -------------------- helpers --------------------

def boxes_from_points(points: List[Tuple[int,int,int]], pad=12, W=None, H=None):
    by_f = defaultdict(list)
    for x, y, f in points:
        by_f[int(f)].append((float(x), float(y)))
    boxes = {}
    for f, xy in by_f.items():
        xs, ys = zip(*xy)
        x1, y1 = int(min(xs) - pad), int(min(ys) - pad)
        x2, y2 = int(max(xs) + pad), int(max(ys) + pad)
        if W is not None and H is not None:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 > x1 and y2 > y1:
            boxes[f] = (x1, y1, x2, y2)
    return boxes


def choose_label(tags: List[Dict]) -> Optional[str]:
    if not tags:
        return None
    labs = [t.get("label") for t in tags if t.get("label")]
    if not labs:
        return None
    counts = Counter(labs).most_common()
    tied = [l for l, c in counts if c == counts[0][1]]
    if len(tied) == 1:
        return tied[0]
    # tie-break by summed confidence
    conf = {l: sum(t.get("confidence", 0.0) for t in tags if t.get("label") == l) for l in tied}
    return max(conf, key=conf.get)


def uniform_indices(frames: List[int], num: int) -> List[int]:
    if len(frames) == 0:
        return []
    if len(frames) == 1:
        return [frames[0]] * num
    return [frames[round(i * (len(frames) - 1) / (num - 1))] for i in range(num)]


# ImageNet normalization for DINOv3/VIT backbones
IMAGENET_NORM = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


# -------------------- dataset --------------------

class NZThermalTracksNoFiltered(Dataset):
    """
    Layout:
      root/
        videos/                 # contains <id>.mp4 and maybe <id>_filtered.mp4 (ignored)
        individual-metadata/    # <id>_metadata.json with track 'points'
        new-zealand-wildlife-thermal-imaging.json (optional)

    Returns:
      x: [T, 3, H, W]  float32
      y: int64
      meta: dict
    """
    def __init__(
        self,
        root: str,
        num_samples: int = 24,
        size: int = 224,
        label_map: Optional[Dict[str, int]] = None,
        allowed_clip_ids: Optional[Set[int]] = None,
        include_false_positive: bool = True,
        min_track_len: int = 4,
        pad: int = 12,
        drop_calibration_frames: bool = True,
        normalize: str = "imagenet",  # "imagenet" or "none"
    ):
        self.root = root
        self.vdir = os.path.join(root, "videos")
        self.mdir = os.path.join(root, "individual-metadata")
        self.num_samples, self.size = num_samples, size
        self.pad = pad
        self.drop_calibration_frames = drop_calibration_frames

        # choose normalizer
        norm = (normalize or "none").lower()
        if norm == "imagenet":
            self.normalizer = IMAGENET_NORM
        elif norm == "none":
            self.normalizer = None
        else:
            raise ValueError(f"Unknown normalize='{normalize}', use 'imagenet' or 'none'.")

        metas = sorted(glob.glob(os.path.join(self.mdir, "*_metadata.json")))
        if not metas:
            raise FileNotFoundError("No *_metadata.json found in individual-metadata/")

        # build label_map if not given
        if label_map is None:
            seen = []
            for p in metas[:5000]:  # quick scan
                try:
                    m = json.load(open(p))
                    for tr in m.get("tracks", []):
                        lab = choose_label(tr.get("tags", []))
                        if lab:
                            seen.append(lab)
                except Exception:
                    pass
            classes = sorted(set(seen))
            if not include_false_positive:
                classes = [c for c in classes if "false" not in c.lower()]
            self.label_map = {c: i for i, c in enumerate(classes)}
        else:
            self.label_map = label_map

        # index items (track-level)
        self.items = []
        for p in metas:
            try:
                m = json.load(open(p))
            except Exception:
                continue
            clip_id = m.get("id") or m.get("clip_id")
            if allowed_clip_ids is not None and clip_id not in allowed_clip_ids:
                continue

            # ONLY use non-filtered mp4
            base = os.path.splitext(os.path.basename(p))[0].replace("_metadata", "")  # e.g., '390'
            vpath = os.path.join(self.vdir, f"{base}.mp4")
            if not os.path.exists(vpath):
                # strictly ignore filtered videos as requested
                continue

            W, H = m.get("width"), m.get("height")
            calib = set(m.get("calibration_frames", []) or [])

            for ti, tr in enumerate(m.get("tracks", [])):
                lab = choose_label(tr.get("tags", []))
                if not lab or lab not in self.label_map:
                    continue
                pts = tr.get("points", [])
                if not pts:
                    continue
                pts = [(int(p[0]), int(p[1]), int(p[2])) for p in pts]
                boxes = boxes_from_points(pts, self.pad, W, H)
                frames = sorted(boxes.keys())
                if self.drop_calibration_frames and calib:
                    frames = [f for f in frames if f not in calib]
                if len(frames) < min_track_len:
                    continue
                self.items.append(
                    {
                        "video": vpath,
                        "boxes": boxes,
                        "frames": frames,
                        "label_id": self.label_map[lab],
                        "label_str": lab,
                        "clip_id": clip_id,
                        "track_index": ti,
                        "W": W,
                        "H": H,
                    }
                )

        if not self.items:
            raise RuntimeError("No usable tracks found (maybe only _filtered.mp4 present?).")

    def __len__(self):
        return len(self.items)

    def _read_frame(self, cap, fidx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            return None
        # Convert to grayscale then 3-channel for backbones expecting 3ch
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.stack([frame, frame, frame], axis=2)  # H,W,3
        return frame

    def __getitem__(self, i):
        it = self.items[i]
        cap = cv2.VideoCapture(it["video"])
        if not cap.isOpened():
            raise IOError(f"Could not open video: {it['video']}")
        idxs = uniform_indices(it["frames"], self.num_samples)

        crops = []
        for f in idxs:
            frame = self._read_frame(cap, f)
            if frame is None:
                frame = np.zeros((it["H"], it["W"], 3), dtype=np.uint8)
            x1, y1, x2, y2 = it["boxes"].get(f, it["boxes"][it["frames"][0]])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
            crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
            crop = crop.astype(np.float32) / 255.0                 # [H,W,3] in [0,1]
            tensor = torch.from_numpy(crop).permute(2, 0, 1)       # [3,H,W]
            if self.normalizer is not None:
                tensor = self.normalizer(tensor)                   # ImageNet norm for DINOv3
            crops.append(tensor)
        cap.release()

        x = torch.stack(crops, 0)              # [T,3,H,W]
        y = torch.tensor(it["label_id"]).long()
        meta = {
            "label_str": it["label_str"],
            "clip_id": it["clip_id"],
            "track_index": it["track_index"],
            "video_path": it["video"],
        }
        return x, y, meta


# -------------------- loader factory --------------------

def make_loader(root, batch_size=8, num_workers=4, **kwargs):
    """
    kwargs are passed to NZThermalTracksNoFiltered, e.g.:
      num_samples, size, include_false_positive, allowed_clip_ids, normalize
    """
    ds = NZThermalTracksNoFiltered(root, **kwargs)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl, ds.label_map


# -------------------- quick test --------------------

if __name__ == "__main__":
    ROOT = "/home/timiles/tim/invasive-classifier/nz_thermal_data"

    loader, label_map = make_loader(
        ROOT,
        batch_size=2,
        num_workers=4,
        num_samples=24,
        size=224,
        include_false_positive=True,
        normalize="imagenet",  # or "none"
    )

    print(f"classes: {len(label_map)}", list(label_map)[:10], "...")
    xb, yb, meta = next(iter(loader))
    print("xb:", xb.shape, "yb:", yb.shape, "labels:", meta["label_str"][:2])
