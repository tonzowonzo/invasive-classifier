# invasive_classifier/train/train.py
import os, math, time, json
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from invasive_classifier.data.dataset import make_loader
from invasive_classifier.model.track_classifier import build_dinov3_track_classifier
from invasive_classifier.utils import plots


# ---------------- utils ----------------

def set_seed(seed=1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_class_weights(label_map, dataset_items):
    labels = [it["label_id"] for it in dataset_items]
    cnt = Counter(labels)
    num = len(dataset_items)
    weights = {lbl: num / (len(cnt) * c) for lbl, c in cnt.items()}
    return torch.tensor([weights[i] for i in range(len(label_map))], dtype=torch.float)

def macro_f1_from_counts(tp, fp, fn, eps=1e-9):
    f1s = []
    for c in tp.keys():
        p = tp[c] / (tp[c] + fp[c] + eps)
        r = tp[c] / (tp[c] + fn[c] + eps)
        f1 = 2*p*r/(p+r+eps)
        f1s.append(f1)
    return sum(f1s)/max(1,len(f1s))

def evaluate(model, loader, device, label_map, writer=None, step=0, save_dir=None):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

    with torch.no_grad():
        for xb,yb,meta in loader:
            xb = xb.to(device).float(); yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            loss_sum += loss.item()*yb.size(0); n += yb.size(0)
            pred = logits.argmax(1)
            for t,p in zip(yb.tolist(), pred.tolist()):
                if t==p: tp[t]+=1
                else: fp[p]+=1; fn[t]+=1

    avg_loss = loss_sum/max(1,n)
    mf1 = macro_f1_from_counts(tp,fp,fn)

    if writer:
        writer.add_scalar("eval/loss", avg_loss, step)
        writer.add_scalar("eval/macro_f1", mf1, step)

    if save_dir:
        inv_map={v:k for k,v in label_map.items()}
        class_f1={}
        for c in inv_map:
            p=tp[c]/(tp[c]+fp[c]+1e-9); r=tp[c]/(tp[c]+fn[c]+1e-9)
            f1=2*p*r/(p+r+1e-9); class_f1[inv_map[c]]=f1
        os.makedirs(save_dir,exist_ok=True)
        with open(os.path.join(save_dir,f"class_f1_step{step}.json"),"w") as f: json.dump(class_f1,f,indent=2)
    return avg_loss,mf1


# ---------------- training ----------------

def main():
    # -------- config --------
    CFG = dict(
        root="/home/timiles/tim/invasive-classifier/nz_thermal_data",
        local_ckpt="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        out="artifacts", logdir="runs/invasive",
        epochs=10, batch_size=8, num_workers=8,
        num_samples=24, size=224,
        lr_head=1e-4, lr_backbone=5e-5, weight_decay=0.05,
        grad_clip=1.0, unfreeze_last_n_blocks=0,
        use_weighted_sampler=True,
        debug_every_steps=250,
    )

    set_seed(1337)
    device="cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_loader,label_map = make_loader(CFG["root"],
        batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
        num_samples=CFG["num_samples"], size=CFG["size"], include_false_positive=True)
    eval_loader,_ = make_loader(CFG["root"],
        batch_size=max(1,CFG["batch_size"]//2), num_workers=CFG["num_workers"],
        num_samples=CFG["num_samples"], size=CFG["size"], include_false_positive=True)

    if CFG["use_weighted_sampler"]:
        labels=[it["label_id"] for it in train_loader.dataset.items]
        cnt=Counter(labels)
        sample_w=[1.0/cnt[it["label_id"]] for it in train_loader.dataset.items]
        sampler=WeightedRandomSampler(sample_w,len(sample_w),replacement=True)
        train_loader=DataLoader(train_loader.dataset,batch_size=CFG["batch_size"],
                                sampler=sampler,num_workers=CFG["num_workers"],pin_memory=True)

    # model
    model=build_dinov3_track_classifier(
        num_classes=len(label_map), backbone_name="vit_base_patch16_224",
        temporal="meanmax", freeze_backbone=(CFG["unfreeze_last_n_blocks"]==0),
        unfreeze_last_n_blocks=CFG["unfreeze_last_n_blocks"],
        dropout=0.2, local_checkpoint=CFG["local_ckpt"]).to(device)

    back_params=[p for p in model.backbone.parameters() if p.requires_grad]
    head_params=[p for p in model.temporal.parameters() if p.requires_grad]
    groups=[]
    if back_params: groups.append({"params":back_params,"lr":CFG["lr_backbone"]})
    if head_params: groups.append({"params":head_params,"lr":CFG["lr_head"]})
    optimizer=optim.AdamW(groups,weight_decay=CFG["weight_decay"])

    # Epoch-level cosine schedule (cleaner; avoids early warnings)
    sched=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG["epochs"])

    # AMP (new API)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    class_w=make_class_weights(label_map,train_loader.dataset.items).to(device)
    criterion=nn.CrossEntropyLoss(weight=class_w)

    writer=SummaryWriter(CFG["logdir"])
    os.makedirs(CFG["out"],exist_ok=True)
    best_f1=-1.0; global_step=0

    for epoch in range(1,CFG["epochs"]+1):
        model.train(); t0=time.time()
        ep_loss=0; ep_acc=0; seen=0

        # tqdm progress bar over the training loader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}", ncols=120)
        for xb,yb,meta in pbar:
            xb=xb.to(device).float(); yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits=model(xb); loss=criterion(logits,yb)
            scaler.scale(loss).backward()
            if CFG["grad_clip"]>0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),CFG["grad_clip"])
            scaler.step(optimizer); scaler.update()

            # running stats
            bs=yb.size(0)
            ep_loss+=loss.item()*bs
            ep_acc+=(logits.argmax(1)==yb).sum().item()
            seen+=bs

            # update tqdm line
            curr_loss = ep_loss/max(1,seen)
            curr_acc  = ep_acc/max(1,seen)
            lrs = [g["lr"] for g in optimizer.param_groups]
            pbar.set_postfix(loss=f"{curr_loss:.4f}", acc=f"{curr_acc:.3f}", lr="/".join(f"{lr:.2e}" for lr in lrs))

            # TB scalars every few steps
            if global_step%10==0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                for i,g in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], global_step)

            # Debug crops
            if global_step % CFG["debug_every_steps"] == 0:
                save_dir=os.path.join(CFG["out"],"debug_plots"); os.makedirs(save_dir,exist_ok=True)
                fig = plots.show_track_crops(xb[0].detach().cpu(), label=f"y={meta['label_str'][0]}")
                fig_path = os.path.join(save_dir, f"crops_e{epoch}_s{global_step}.png")
                fig.savefig(fig_path, dpi=120); plt.close(fig)

                # Log to TensorBoard
                img = np.array(Image.open(fig_path))
                writer.add_image("debug/crops", torch.from_numpy(img).permute(2,0,1), global_step)

            global_step+=1

        # end epoch stats
        train_loss=ep_loss/max(1,seen); train_acc=ep_acc/max(1,seen)
        writer.add_scalar("train/epoch_loss",train_loss,epoch)
        writer.add_scalar("train/epoch_acc",train_acc,epoch)

        # step scheduler once per epoch
        sched.step()

        # eval
        eval_loss,eval_f1=evaluate(model,eval_loader,device,label_map,writer,global_step,CFG["out"])
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} "
              f"| eval_loss={eval_loss:.4f} macro_f1={eval_f1:.3f} "
              f"| time {time.time()-t0:.1f}s")

        # save ckpts
        last_path=os.path.join(CFG["out"],"last.ckpt")
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "scheduler":sched.state_dict(),"scaler":scaler.state_dict(),
                    "best_f1":best_f1,"global_step":global_step,"epoch":epoch,
                    "label_map":label_map,"config":CFG},last_path)
        if eval_f1>best_f1:
            best_f1=eval_f1
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "scheduler":sched.state_dict(),"scaler":scaler.state_dict(),
                        "best_f1":best_f1,"global_step":global_step,"epoch":epoch,
                        "label_map":label_map,"config":CFG},
                       os.path.join(CFG["out"],"best.ckpt"))
            print(f"✓ New best saved (macro_f1={best_f1:.3f})")

    writer.close()


if __name__=="__main__":
    main()
