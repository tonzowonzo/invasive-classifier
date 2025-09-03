# Invasive Species Classifier (Thermal Video, New Zealand Wildlife)

This is a WIP - I'll likely change the architecture significantly.

This repository contains code and training utilities for building a **track-level classifier** of invasive species (e.g., possums, rodents, cats, hedgehogs) using thermal camera trap videos from **[The Cacophony Project](https://cacophony.org.nz/)**.  

The goal is to support conservation by automatically detecting and classifying animals in thermal footage, enabling better monitoring and control of invasive species.

---

## Dataset

We use the public **NZ Thermal Wildlife dataset**:

- **121,190 thermal videos**, recorded mostly at night across New Zealand.
- Each video is annotated with one or more **tracks** (continuous movement).
- Tracks include:
  - bounding boxes (`points`)
  - frame indices
  - species labels (45 categories, including “false positive”).
- We use only the **non-filtered mp4s** (not the background-subtracted ones).
- Data layout (after download):
  ```
  nz_thermal_data/
    videos/                  # <id>.mp4
    individual-metadata/     # <id>_metadata.json with tracks + points
    new-zealand-wildlife-thermal-imaging.json
  ```

See the [official dataset page](https://lila.science/datasets/new-zealand-wildlife-thermal-imaging/) for download instructions.

---

## Model

We build a **track-level classifier**:

- **Backbone**: [DINOv3 ViT-B/16](https://github.com/facebookresearch/dinov3), loaded from a local checkpoint (`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`).
- **Frozen backbone** (optionally unfreeze last N blocks).
- **Temporal pooling**:
  - Mean+Max pooling (`meanmax`, default), or
  - GRU head (`gru`).
- Outputs per-track species classification logits.

Normalization:  
- Grayscale frames are replicated to **3 channels**.  
- Resized to **224×224**.  
- Normalized with **ImageNet mean/std** (`0.485, 0.456, 0.406` / `0.229, 0.224, 0.225`) to match DINOv3 pretraining.

---

## Installation

We use **Conda for system deps** (Python, PyTorch, CUDA) and **Poetry** for Python deps.

### Conda env

```bash
conda create -n invasive-classifier python=3.11 pytorch torchvision torchaudio -c pytorch -c nvidia -c conda-forge
conda activate invasive-classifier
```

### Poetry deps

```bash
pip install poetry
poetry install
```

Key Python deps:  
- `timm` (ViT + DINOv3 models)  
- `torchvision`  
- `opencv-python-headless`  
- `matplotlib`, `pandas`, `tqdm`, `tensorboard`

---

## Training

Run training:

```bash
python invasive_classifier/train/train.py
```

This will:
- Build the dataset + dataloaders.
- Load DINOv3 backbone with local weights.
- Train a temporal classifier head.
- Log training progress with **tqdm** and **TensorBoard**.
- Save checkpoints under `artifacts/`.

### Config (inside `train.py`)

```python
CFG = dict(
    root="PATH/TO/nz_thermal_data",
    local_ckpt="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    out="artifacts",
    logdir="runs/invasive",
    epochs=10,
    batch_size=8,
    num_workers=8,
    num_samples=24,
    size=224,
    lr_head=1e-4,
    lr_backbone=5e-5,
    weight_decay=0.05,
    grad_clip=1.0,
    unfreeze_last_n_blocks=0,
    use_weighted_sampler=True,
)
```

---

## Debugging

- **Debug crops** are saved to `artifacts/debug_plots/` and logged to TensorBoard.
- **Class F1 scores** are written per eval step into JSON under `artifacts/`.

---

## License

This repository is for research purposes. Dataset: **Community Data License Agreement (permissive variant)**.  
See [The Cacophony Project](https://cacophony.org.nz/) for details.

---
