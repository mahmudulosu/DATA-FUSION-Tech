# -*- coding: utf-8 -*-
"""
EEG-only Age Classification (Young vs Old)
------------------------------------------
Inputs:
- EEG scalograms (.npy) shaped (224,224,17), filename prefix is participant_id
- CSV with columns: participant_id, age ∈ {"20-25","25-30","60-65","65-70","70-75"}

Model:
- EEG encoder: ConvNeXt-Tiny (ImageNet init) with 17-ch stem + Electrode-Aware Attention
- Head: MLP → 2 classes

Training:
- Loss: Cross-Entropy
- Optim: AdamW + CosineAnnealingLR
- Mixed precision (CUDA)
- Class-balanced sampler

Outputs (./age_eeg/):
- Train/Val curves (PNG) + history CSV
- Test predictions CSV
- Confusion matrix (PNG) + ROC curves (PNG)
- t-SNE of test embeddings (PNG)
- Paper metrics CSV + Glossary TXT

Checkpoint:
- ./checkpoints/eeg_age_convnext.pth
"""

import os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_curve, roc_auc_score,
                             average_precision_score)
from sklearn.manifold import TSNE

# ====================== Config & Device ======================
SEED = 1337
random.seed(SEED); np.random.seed(SEED := 1337)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------- EDIT THESE ---------
EEG_DIR = r"C:\research\EEG_Domain\eye_openscalo"               # .npy, shape (224,224,17)
CSV     = r"C:/research/MRI/participants_LSD_andLEMON.csv"      # has: participant_id, age
# ------------------------------

OUT_DIR = "age_eeg"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Dataloader knobs (Windows-safe)
NUM_WORKERS = 0
PIN_MEMORY  = True

# Training knobs
BATCH = 8
EPOCHS = 40
LR_BACKBONE  = 1e-4
LR_HEAD      = 2e-4
WEIGHT_DECAY = 1e-4
USE_TTA      = True  # simple flips at test time

# ====================== Label prep (Age → young/old) ======================

def build_age_label_map(csv_path: str):
    df = pd.read_csv(csv_path)
    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group = {"20-25": "young", "25-30": "young",
                 "60-65": "old",   "65-70": "old",   "70-75": "old"}
    df = df[df["age"].isin(valid_ages)].copy()
    label_map = {"young": 0, "old": 1}
    id2label = {row["participant_id"]: label_map[age_group[row["age"]]]
                for _, row in df.iterrows()}
    return id2label, ["young","old"]

age_map, CLASS_NAMES = build_age_label_map(CSV)

# ====================== Utils ======================

def extract_pid(path_or_name: str) -> str:
    return os.path.basename(path_or_name).split("_")[0]

def zscore(x, eps: float = 1e-6):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def specaugment_like(x: torch.Tensor, time_mask_p=0.08, freq_mask_p=0.08):
    # x: [C=17, H=224, W=224]
    C, H, W = x.shape
    x = x.clone()
    if random.random() < 0.8:
        tw = int(W * time_mask_p)
        if tw > 0:
            t0 = random.randint(0, max(0, W - tw))
            x[:, :, t0:t0 + tw] = 0
        fh = int(H * freq_mask_p)
        if fh > 0:
            f0 = random.randint(0, max(0, H - fh))
            x[:, f0:f0 + fh, :] = 0
    return x

# ====================== Data ======================

class EEGDataset(Dataset):
    def __init__(self, eeg_dir, id2label, train=True):
        self.samples = []
        for fn in os.listdir(eeg_dir):
            if fn.endswith(".npy"):
                pid = extract_pid(fn)
                if pid in id2label:
                    self.samples.append((os.path.join(eeg_dir, fn), int(id2label[pid]), pid))
        self.train = train

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        arr = np.load(path).astype(np.float32)      # (224,224,17)
        # per-channel z-score
        for c in range(arr.shape[-1]):
            arr[..., c] = zscore(arr[..., c])
        x = torch.from_numpy(arr).permute(2,0,1)    # [17,224,224]
        if self.train:
            x = specaugment_like(x, 0.08, 0.08)
        return x, torch.tensor(y).long(), pid

def make_loader(ds, batch_size):
    labels = [int(l) for (_, l, _) in ds.samples]
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sample_w = [float(weights[l]) for l in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ====================== Model ======================

class ElectrodeAttention(nn.Module):
    def __init__(self, C=17, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Conv2d(C, max(1, C//r), 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, C//r), C, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, 17, 224, 224]
        w = self.mlp(self.pool(x))
        return x * w

class EEGEncoderConvNeXt(nn.Module):
    def __init__(self, out_dim=512, unfreeze_stages=(3,4)):
        super().__init__()
        self.eaa = ElectrodeAttention(C=17, r=4)
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        stem = base.features[0][0]  # Conv2d(3, 96, ...)
        new_stem = nn.Conv2d(17, stem.out_channels,
                             kernel_size=stem.kernel_size,
                             stride=stem.stride, padding=stem.padding, bias=False)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.mean(1, keepdim=True).repeat(1,17,1,1))
        base.features[0][0] = new_stem
        self.backbone = base.features
        self.backbone_ln = nn.LayerNorm(768, eps=1e-6)
        self.proj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(768, out_dim))

        # freeze all then unfreeze selected
        for p in self.parameters(): p.requires_grad = False
        for i, block in enumerate(self.backbone):
            if i in unfreeze_stages:
                for p in block.parameters(): p.requires_grad = True
        for p in self.backbone_ln.parameters(): p.requires_grad = True
        for p in self.proj.parameters(): p.requires_grad = True
        for p in self.eaa.parameters(): p.requires_grad = True

    def forward(self, x):
        # x: [B, 17, 224, 224]
        x = self.eaa(x)
        x = self.backbone(x)          # [B,768,h,w]
        x = x.mean([-2, -1])          # [B,768]
        x = self.backbone_ln(x)
        x = self.proj(x)              # [B,512]
        return F.normalize(x, dim=-1)

class EEGClassifier(nn.Module):
    def __init__(self, dim=512, n_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 2*dim), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(2*dim, n_classes)
        )
    def forward(self, z):
        return self.head(z)

# ====================== Eval helpers ======================

def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs); labels = np.asarray(labels)
    conf = probs[:,1]
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i>0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask): continue
        acc_bin = ( (conf[mask] >= 0.5).astype(int) == labels[mask] ).mean()
        conf_bin = conf[mask].mean()
        ece += np.abs(acc_bin - conf_bin) * (mask.mean())
    return float(ece)

@torch.no_grad()
def eval_epoch(model_enc, model_cls, loader, split_name="val", tta=False):
    model_enc.eval(); model_cls.eval()
    preds, gts, probs = [], [], []
    ce_loss_sum, n_ce = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if not tta:
                z = model_enc(x)
                logits = model_cls(z)
            else:
                logits_all = []
                z0 = model_enc(x); l0 = model_cls(z0); logits_all.append(l0)
                z1 = model_enc(x.flip(-1)); l1 = model_cls(z1); logits_all.append(l1)
                z2 = model_enc(x.flip(-2)); l2 = model_cls(z2); logits_all.append(l2)
                logits = torch.stack(logits_all, 0).mean(0)

        prob = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        ce_loss_sum += float(ce(logits, y).item()); n_ce += y.size(0)
        preds.append(pred.cpu()); gts.append(y.cpu()); probs.append(prob.cpu())

    if not preds:
        print(f"[{split_name}] No samples.")
        return {"acc": 0.0, "loss_ce": 0.0}

    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    probs = torch.cat(probs).numpy()
    acc = (preds == gts).mean()
    val_loss = ce_loss_sum / max(1, n_ce)

    print(f"[{split_name}] CE loss: {val_loss:.4f} | Acc: {acc:.4f}")
    print(confusion_matrix(gts, preds, labels=[0,1]))
    print(classification_report(gts, preds, labels=[0,1], target_names=CLASS_NAMES, digits=4, zero_division=0))
    return {"acc": float(acc), "loss_ce": float(val_loss), "preds": preds, "gts": gts, "probs": probs}

# ====================== Plotting / Reporting ======================

def plot_curves(history, out_base_png):
    ep = np.arange(1, len(history)+1)
    tr_loss = [h["train_loss"] for h in history]
    tr_acc  = [h["train_acc"]  for h in history]
    va_loss = [h["val_loss"]   for h in history]
    va_acc  = [h["val_acc"]    for h in history]

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_loss, label="Train Loss")
    plt.plot(ep, va_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_base_png.replace(".png","_loss.png"), dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_acc, label="Train Acc")
    plt.plot(ep, va_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_base_png.replace(".png","_acc.png"), dpi=200); plt.close()

def save_history_csv(history, out_csv):
    df = pd.DataFrame(history); df.index = np.arange(1, len(df)+1); df.index.name = "epoch"; df.to_csv(out_csv)

def plot_confusion_and_roc(gts, probs, out_cm, out_roc):
    preds = probs.argmax(1)
    cm = confusion_matrix(gts, preds, labels=[0,1])

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest'); plt.title("Confusion Matrix"); plt.colorbar()
    tick = np.arange(2); plt.xticks(tick, CLASS_NAMES); plt.yticks(tick, CLASS_NAMES)
    th = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > th else "black")
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout(); plt.savefig(out_cm, dpi=200); plt.close()

    y_true_bin = np.eye(2)[gts]
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(6,5))
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], probs[:, i])
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{CLASS_NAMES[i]} AUC={roc_auc[i]:.3f}")
    fpr_m, tpr_m, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    auc_m = roc_auc_score(y_true_bin.ravel(), probs.ravel())
    plt.plot(fpr_m, tpr_m, lw=2.0, linestyle="--", label=f"Micro AUC={auc_m:.3f}")
    plt.plot([0,1],[0,1], linestyle=":", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_roc, dpi=200); plt.close()

def plot_tsne(emb, gts, out_png, perplexity=30, random_state=SEED):
    emb = np.asarray(emb); gts = np.asarray(gts)
    if emb.ndim != 2 or emb.shape[0] < 3:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5, 0.5, "t-SNE skipped (not enough samples)", ha='center', va='center')
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(); return
    valid = np.isfinite(emb).all(axis=1)
    emb = emb[valid]; gts = gts[valid]; n = emb.shape[0]
    p = max(2, min(perplexity, n-1))
    print(f"[t-SNE] n={n}, perplexity={p}")
    tsne = TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto", random_state=random_state)
    z = tsne.fit_transform(emb)
    plt.figure(figsize=(6,5))
    for i, name in enumerate(CLASS_NAMES):
        idx = (gts == i)
        if idx.any(): plt.scatter(z[idx,0], z[idx,1], s=18, marker=("o" if i==0 else "^"), label=name, alpha=0.8)
    plt.legend(); plt.title("t-SNE of EEG Test Embeddings"); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def write_metrics_glossary(path_txt):
    txt = (
        "Metrics Glossary\n"
        "-----------------\n"
        "Acc → Accuracy (overall proportion of correct predictions)\n"
        "Macro-F1 → Macro-averaged F1 Score (F1 computed per class, then averaged equally)\n"
        "AUROC → Area Under the Receiver Operating Characteristic Curve\n"
        "AUPRC → Area Under the Precision–Recall Curve\n"
        "ECE ↓ → Expected Calibration Error (lower is better)\n"
        "Params (M) → Number of trainable parameters (in millions)\n"
        "ms/sample → Inference time per sample (milliseconds per subject/image)\n"
    )
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(txt)

def compute_and_save_paper_metrics(stats, enc, head, out_csv, out_txt):
    gts, probs = stats["gts"], stats["probs"]
    preds = probs.argmax(1)
    acc = (preds == gts).mean()
    macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(gts, probs[:,1])
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(gts, probs[:,1])
    except Exception:
        auprc = float("nan")
    ece = expected_calibration_error(probs, gts, n_bins=15)

    def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    params_m = (count_trainable(enc) + count_trainable(head)) / 1e6
    ms_per_sample = float("nan")

    pd.DataFrame([{
        "Acc": acc,
        "Macro-F1": macro_f1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "ECE": ece,
        "Params (M)": params_m,
        "ms/sample": ms_per_sample
    }]).to_csv(out_csv, index=False)
    write_metrics_glossary(out_txt)

# ====================== Train / Test ======================

def build_subject_list(eeg_dir, id2label):
    ds = EEGDataset(eeg_dir, id2label, train=True)
    ids = [pid for (_,_,pid) in ds.samples]
    return ds, sorted(list(set(ids)))

def stratified_subject_split(common_ids, id2label, test_frac=0.20, seed=SEED):
    rng = random.Random(seed)
    by_cls = {0: [], 1: []}
    for pid in common_ids:
        if pid in id2label:
            by_cls[id2label[pid]].append(pid)
    for k in by_cls: rng.shuffle(by_cls[k])
    test_ids = set()
    for k, lst in by_cls.items():
        n_k = len(lst); t_k = max(1, int(round(test_frac * n_k))) if n_k>0 else 0
        test_ids.update(lst[:t_k])
    train_ids = set(pid for pid in common_ids if pid not in test_ids)
    print(f"[Stratified Split] Class0={len(by_cls[0])} Class1={len(by_cls[1])} | Test={len(test_ids)}")
    return train_ids, test_ids

def subset_dataset(ds, keep_ids, is_train):
    keep = []
    for (p,l,pid) in ds.samples:
        if pid in keep_ids:
            keep.append((p,l,pid))
    new = EEGDataset(EEG_DIR, age_map, train=is_train)
    new.samples = keep
    return new

def count_ids(ids, id2label):
    c0 = sum(1 for pid in ids if id2label.get(pid, 1) == 0)
    c1 = sum(1 for pid in ids if id2label.get(pid, 1) == 1)
    return c0, c1

def print_split_stats(train_ids, test_ids, id2label):
    y_tr0, y_tr1 = count_ids(train_ids, id2label)
    y_te0, y_te1 = count_ids(test_ids,  id2label)
    print(f"[Subjects] Train: {CLASS_NAMES[0]}={y_tr0}  {CLASS_NAMES[1]}={y_tr1}  (Total={len(train_ids)})")
    print(f"[Subjects] Test : {CLASS_NAMES[0]}={y_te0}  {CLASS_NAMES[1]}={y_te1}  (Total={len(test_ids)})")

def train_eeg_only(train_ds, test_ds):
    tr_loader = make_loader(train_ds, BATCH)
    te_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    enc = EEGEncoderConvNeXt(out_dim=512, unfreeze_stages=(3,4)).to(device)
    cls = EEGClassifier(dim=512, n_classes=2).to(device)

    params = [
        {'params': [p for p in enc.parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': [p for p in cls.parameters() if p.requires_grad], 'lr': LR_HEAD},
    ]
    opt = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ce = nn.CrossEntropyLoss()

    history = []
    best = 0.0

    for ep in range(1, EPOCHS+1):
        enc.train(); cls.train()
        tot, corr, n = 0.0, 0, 0
        for x, y, _ in tqdm(tr_loader, desc=f"Epoch {ep}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                z = enc(x)
                logits = cls(z)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(cls.parameters(), 5.0)
            scaler.step(opt); scaler.update()

            tot += float(loss.item()) * y.size(0)
            pred = logits.argmax(1)
            corr += int((pred==y).sum().item()); n += y.size(0)

        sched.step()
        tr_loss = tot/max(1,n); tr_acc = corr/max(1,n)

        val_stats = eval_epoch(enc, cls, te_loader, split_name="val/test", tta=False)
        te_acc = val_stats["acc"]; te_loss = val_stats["loss_ce"]
        best = max(best, te_acc)
        print(f"[Train] ep {ep} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {te_loss:.4f} acc {te_acc:.4f} | best {best:.4f}")

        history.append({
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(te_loss),
            "val_acc": float(te_acc),
        })

    # Save training curves/history
    save_history_csv(history, os.path.join(OUT_DIR, "history_metrics.csv"))
    plot_curves(history, os.path.join(OUT_DIR, "curves_train_val.png"))

    print("Evaluating with light TTA & collecting test outputs...")
    test_stats, emb, pids = collect_test_outputs(enc, cls, te_loader, use_tta=USE_TTA)

    gts   = test_stats["gts"]; probs = test_stats["probs"]; preds = probs.argmax(1)
    pd.DataFrame({
        "pid": pids,
        "y_true": gts,
        "y_pred": preds,
        "prob_"+CLASS_NAMES[0]: probs[:,0],
        "prob_"+CLASS_NAMES[1]: probs[:,1],
    }).to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    plot_confusion_and_roc(gts, probs,
                           out_cm=os.path.join(OUT_DIR, "confusion_matrix.png"),
                           out_roc=os.path.join(OUT_DIR, "roc_curve.png"))

    plot_tsne(emb, gts, out_png=os.path.join(OUT_DIR, "tsne_test_embeddings.png"), perplexity=30)

    compute_and_save_paper_metrics(
        test_stats, enc, cls,
        out_csv=os.path.join(OUT_DIR, "paper_metrics.csv"),
        out_txt=os.path.join(OUT_DIR, "metrics_glossary.txt")
    )

    torch.save({
        'encoder': enc.state_dict(),
        'classifier': cls.state_dict(),
        'config': {
            'classes': CLASS_NAMES
        }
    }, os.path.join("checkpoints", "eeg_age_convnext.pth"))
    print("✅ Saved: checkpoints/eeg_age_convnext.pth")
    print(f"✅ Curves/plots/CSVs saved under ./{OUT_DIR}/")
    return enc, cls

@torch.no_grad()
def collect_test_outputs(enc, cls, loader, use_tta=True):
    enc.eval(); cls.eval()
    all_probs, all_gts, all_pids, emb_list = [], [], [], []

    for x, y, pids in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if not use_tta:
                z = enc(x)
                logits = cls(z)
            else:
                logits_all, emb_all = [], []
                z0 = enc(x);          l0 = cls(z0); logits_all.append(l0); emb_all.append(z0)
                z1 = enc(x.flip(-1)); l1 = cls(z1); logits_all.append(l1); emb_all.append(z1)
                z2 = enc(x.flip(-2)); l2 = cls(z2); logits_all.append(l2); emb_all.append(z2)
                logits = torch.stack(logits_all, 0).mean(0)
                z      = torch.stack(emb_all, 0).mean(0)

        prob = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(prob); all_gts.append(y.numpy()); all_pids.extend(list(pids))
        emb_list.append(z.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,2))
    gts   = np.concatenate(all_gts, axis=0)   if all_gts else np.array([])
    emb   = np.concatenate(emb_list, axis=0)  if emb_list else np.zeros((0,512))
    return {"probs": probs, "gts": gts}, emb, all_pids

# ====================== Main ======================
if __name__ == "__main__":
    # Build dataset & subject split (stratified by age group)
    eeg_full, ids = build_subject_list(EEG_DIR, age_map)
    train_ids, test_ids = stratified_subject_split(ids, age_map, test_frac=0.20, seed=SEED)

    # Quick stats
    n_y_tr = sum(age_map[pid]==0 for pid in train_ids)
    n_o_tr = sum(age_map[pid]==1 for pid in train_ids)
    n_y_te = sum(age_map[pid]==0 for pid in test_ids)
    n_o_te = sum(age_map[pid]==1 for pid in test_ids)
    print(f"[Subjects] Train: young={n_y_tr} old={n_o_tr} | Test: young={n_y_te} old={n_o_te}")

    eeg_tr = subset_dataset(eeg_full, train_ids, True)
    eeg_te = subset_dataset(eeg_full, test_ids,  False)

    # Train + evaluate
    enc, cls = train_eeg_only(eeg_tr, eeg_te)
