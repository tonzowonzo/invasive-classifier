from typing import Dict


def get_config() -> Dict:
    """Returns the main configuration dictionary for the training script."""
    return dict(
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


def get_class_mapping() -> Dict[str, str]:
    """Returns the mapping from raw labels to consolidated classes."""
    return {
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
