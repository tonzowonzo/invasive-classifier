# scripts/check_classes.py
import os, json, glob
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple, List
from invasive_classifier.utils.classmap import label_map_from_counts


def load_split_ids(path: str) -> Tuple[Set[int], Set[int]]:
    def extract_ids(obj) -> Set[int]:
        out = set()
        def visit(x):
            if x is None: return
            if isinstance(x, (list, tuple, set)):
                for xi in x: visit(xi)
            elif isinstance(x, dict):
                for k in ("id","clip_id","clip","clipId"):
                    if k in x:
                        try: out.add(int(x[k])); return
                        except Exception: pass
                for v in x.values(): visit(v)
            else:
                try: out.add(int(x))
                except Exception: pass
        visit(obj); return out

    with open(path, "r") as f:
        data = json.load(f)
    def pick(keys):
        for k in keys:
            if k in data:
                ids = extract_ids(data[k])
                if ids: return ids
        return set()
    train_ids = pick(["train","train_ids","train_clip_ids","train_clips"])
    val_ids   = pick(["val","validation","val_ids","valid_ids","validation_ids"])
    test_ids  = pick(["test","test_ids","test_clip_ids","test_clips"])
    eval_ids  = val_ids if val_ids else test_ids
    return train_ids, eval_ids

def choose_label(tags: List[Dict]) -> str|None:
    if not tags: return None
    labs = [t.get("label") for t in tags if t.get("label")]
    if not labs: return None
    # majority; break ties by summed confidence
    from collections import Counter
    counts = Counter(labs).most_common()
    tied = [l for l,c in counts if c == counts[0][1]]
    if len(tied) == 1: return tied[0]
    conf = {l: sum(t.get("confidence",0.0) for t in tags if t.get("label")==l) for l in tied}
    return max(conf, key=conf.get)

def scan_split_counts(indiv_meta_dir: str, clip_ids: Set[int]) -> Counter:
    cnt = Counter()
    for cid in clip_ids:
        p = os.path.join(indiv_meta_dir, f"{cid}_metadata.json")
        if not os.path.exists(p): 
            continue
        try:
            m = json.load(open(p))
            for tr in m.get("tracks", []):
                lab = choose_label(tr.get("tags", []))
                if lab: cnt[lab] += 1
        except Exception:
            pass
    return cnt

def main():
    ROOT = "/home/timiles/tim/invasive-classifier/nz_thermal_data"
    SPLITS = os.path.join(ROOT, "new-zealand-wildlife-thermal-imaging-splits.json")
    COUNTS_CSV = "/home/timiles/tim/invasive-classifier/new-zealand-wildlife-thermal-imaging-counts.csv"  # adjust if needed

    # label_map from counts CSV (no filtering here; change if you want)
    from invasive_classifier.utils.classmap import label_map_from_counts
    label_map, csv_counts = label_map_from_counts(COUNTS_CSV, include_false_positive=True, min_count=None, sort_by="name")
    classes = list(label_map.keys())
    print(f"[classes] {len(classes)} total from CSV")

    train_ids, eval_ids = load_split_ids(SPLITS)
    indiv_dir = os.path.join(ROOT, "individual-metadata")

    train_cnt = scan_split_counts(indiv_dir, train_ids)
    eval_cnt  = scan_split_counts(indiv_dir, eval_ids)

    train_set = set(train_cnt.keys())
    eval_set  = set(eval_cnt.keys())
    csv_set   = set(classes)

    missing_in_train = eval_set - train_set
    missing_in_labelmap = (train_set | eval_set) - csv_set

    print("\n=== Coverage check ===")
    print(f"train classes seen: {len(train_set)}")
    print(f"eval  classes seen: {len(eval_set)}")
    if missing_in_train:
        print(f"[WARN] classes present in EVAL but absent in TRAIN: {sorted(missing_in_train)}")
    else:
        print("No classes present only in eval — good.")

    if missing_in_labelmap:
        print(f"[WARN] classes present in data but not in CSV label_map: {sorted(missing_in_labelmap)}")
    else:
        print("All observed classes exist in the CSV label_map — good.")

    # Save per-split distributions
    os.makedirs("artifacts", exist_ok=True)
    import csv
    with open("artifacts/train_class_distribution.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","tracks"])
        for k,v in sorted(train_cnt.items()): w.writerow([k,v])
    with open("artifacts/eval_class_distribution.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","tracks"])
        for k,v in sorted(eval_cnt.items()): w.writerow([k,v])

    print("\nSaved: artifacts/train_class_distribution.csv, artifacts/eval_class_distribution.csv")

if __name__ == "__main__":
    main()
