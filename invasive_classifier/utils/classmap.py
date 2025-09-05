# invasive_classifier/utils/classmap.py
import csv
from typing import Dict, List, Optional, Tuple


def label_map_from_counts(
    csv_path: str,
    include_false_positive: bool = True,
    min_count: Optional[int] = None,  # set e.g. 10 to drop ultra-rare
    extra_classes: Optional[List[str]] = None,
    sort_by: str = "name",  # "name" or "count"
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns (label_map, counts_dict). Reads a CSV with columns ['label','count'].
    - include_false_positive: keep 'false-positive' class or drop it
    - min_count: drop classes with total < min_count (use with care)
    - extra_classes: ensure these are included even if they don't meet min_count
    - sort_by: index order ('name' alphabetic or 'count' descending)
    """
    counts: Dict[str, int] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Accept variants of header names
        cols = {k.lower(): k for k in reader.fieldnames or []}
        lab_col = cols.get("label") or cols.get("class") or cols.get("category")
        cnt_col = (
            cols.get("count") or cols.get("n") or cols.get("num") or cols.get("samples")
        )
        if not (lab_col and cnt_col):
            raise ValueError(
                f"CSV must have 'label' and 'count' columns; got: {reader.fieldnames}"
            )

        for row in reader:
            lab = (row[lab_col] or "").strip()
            if not lab:
                continue
            try:
                c = int(float(row[cnt_col]))
            except Exception:
                c = 0
            counts[lab] = counts.get(lab, 0) + c

    # optional filtering
    classes = list(counts.keys())
    if not include_false_positive:
        classes = [c for c in classes if "false" not in c.lower()]
    if min_count is not None:
        classes = [c for c in classes if counts.get(c, 0) >= min_count]
    if extra_classes:
        for c in extra_classes:
            if c not in classes:
                classes.append(c)

    if sort_by == "count":
        classes = sorted(classes, key=lambda c: (-counts.get(c, 0), c))
    else:
        classes = sorted(classes)

    label_map = {c: i for i, c in enumerate(classes)}
    return label_map, counts
