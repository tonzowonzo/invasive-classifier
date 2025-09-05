import os
import json
from collections import Counter
from typing import Dict, Set, Tuple, List

from tqdm import tqdm


def load_split_ids(path: str) -> Tuple[Set[int], Set[int]]:
    """Loads train and eval clip IDs from a JSON splits file."""

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
    """Selects the best label from a list of tags based on majority and confidence."""
    if not tags:
        return None
    labs = [t.get("label") for t in tags if t.get("label")]
    if not labs:
        return None
    counts = Counter(labs).most_common()
    tied = [lab for lab, c in counts if c == counts[0][1]]
    if len(tied) == 1:
        return tied[0]
    conf = {
        lab: sum(t.get("confidence", 0.0) for t in tags if t.get("label") == lab)
        for lab in tied
    }
    return max(conf, key=conf.get)


def scan_labels_in_split(indiv_meta_dir: str, clip_ids: Set[int]) -> Counter:
    """Scans all metadata files in a split and counts the occurrences of each label."""
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


def apply_class_mapping(counts: Counter, mapping: Dict[str, str]) -> Counter:
    """Applies a mapping to merge class labels in a Counter object."""
    new_counts = Counter()
    for label, count in counts.items():
        new_label = mapping.get(label, label)
        new_counts[new_label] += count
    return new_counts
