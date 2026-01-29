# -*- coding: utf-8 -*-
"""
GEA-MVA + ViT-B/16 EEG
EEG (17-ch scalogram) + sMRI (3D) gender classification
- EEG: ViT-B/16 pretrained on ImageNet-1K, adapted to 17ch
- MRI: r3d_18 on K=3 center-biased 128^3 crops + attention pooling
- Fusion: Gated fusion head
- Loss: CE + lambda_align*(1-cos) + lambda_contrast*InfoNCE

Logging/outputs (now in output_pretrained/):
- Train/Val history CSV + curves
- Test predictions CSV
- Confusion matrix heatmap
- ROC curves
- t-SNE of fused test embeddings
- Light TTA for evaluation
"""

import os, random
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torchvision.models.video as vmodels
from scipy.ndimage import zoom, rotate

from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             auc)
from sklearn.manifold import TSNE
from tqdm import tqdm
import pandas as pd

# ====================== Config & Device ======================
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------- EDIT THESE ---------
EEG_DIR = r"C:\research\EEG_Domain\eye_openscalo"    # .npy, shape (224,224,17)
MRI_DIR = r"C:/research/MRI/structural_MRI"          # .nii/.nii.gz
CSV     = r"C:/research/MRI/participants_LSD_andLEMON.csv"  # participant_id, gender
# ------------------------------

OUT_DIR = "output_pretrained"
CKPT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Dataloader knobs (Windows-safe)
NUM_WORKERS = 0
PIN_MEMORY  = True

# Training knobs
BATCH = 4
EPOCHS_STAGE_A = 25
EPOCHS_STAGE_B = 30
LR_HEAD      = 2e-4
LR_BACKBONE  = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_ALIGN = 0.10
LAMBDA_CONTR_A = 0.05
LAMBDA_CONTR_B = 0.10
TAU_CONTRAST   = 0.2
K_MRI_CROPS    = 3

# ====================== Utils ======================

def extract_pid(path_or_name: str) -> str:
    return os.path.basename(path_or_name).split("_")[0]

def zscore(x, eps: float = 1e-6):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def specaugment_like(x: torch.Tensor, time_mask_p=0.1, freq_mask_p=0.1):
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

def mri_augment(vol: np.ndarray):
    if random.random() < 0.5: vol = np.flip(vol, 0).copy()
    if random.random() < 0.5: vol = np.flip(vol, 1).copy()
    if random.random() < 0.5: vol = np.flip(vol, 2).copy()
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        vol = rotate(vol, angle, axes=(1, 2), reshape=False, order=1, mode='nearest')
    return vol

# ====================== Data ======================

gender_df = pd.read_csv(CSV)
gender_map = {row['participant_id']: (0 if str(row['gender']).strip().upper()=='M' else 1)
              for _, row in gender_df.iterrows()}

class EEGDataset(Dataset):
    def __init__(self, eeg_dir, labels, train=True):
        self.samples = []
        for fn in os.listdir(eeg_dir):
            if not fn.endswith(".npy"): continue
            pid = extract_pid(fn)
            if pid in labels:
                self.samples.append((os.path.join(eeg_dir, fn), int(labels[pid]), pid))
        self.train = train

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        arr = np.load(path).astype(np.float32)      # (224,224,17)
        for c in range(arr.shape[-1]):
            arr[..., c] = zscore(arr[..., c])
        x = torch.from_numpy(arr).permute(2,0,1)    # [17,224,224]
        if self.train:
            x = specaugment_like(x, 0.08, 0.08)
        return x, torch.tensor(y).long(), pid

class MRIDataset(Dataset):
    """
    Returns K center-biased 128^3 crops for MVA.
    Output shape: [K, 3, 128,128,128]
    """
    def __init__(self, mri_dir, labels, train=True, k_crops=3):
        self.samples = []
        for fn in os.listdir(mri_dir):
            if not fn.endswith((".nii",".nii.gz")): continue
            pid = extract_pid(fn)
            if pid in labels:
                self.samples.append((os.path.join(mri_dir, fn), int(labels[pid]), pid))
        self.train = train
        self.k_crops = k_crops

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        vol = nib.load(path).get_fdata().astype(np.float32)
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)

        target_big = 160
        zf = (target_big/vol.shape[0], target_big/vol.shape[1], target_big/vol.shape[2])
        vol = zoom(vol, zf, order=1)  # [160,160,160]

        crops = []
        for _ in range(self.k_crops):
            if self.train:
                c = target_big // 2; delta = 8
                z0 = np.clip(c-64 + np.random.randint(-delta, delta+1), 0, target_big-128)
                y0 = np.clip(c-64 + np.random.randint(-delta, delta+1), 0, target_big-128)
                x0 = np.clip(c-64 + np.random.randint(-delta, delta+1), 0, target_big-128)
            else:
                z0 = y0 = x0 = (target_big - 128)//2

            v = vol[z0:z0+128, y0:y0+128, x0:x0+128]
            if self.train:
                v = mri_augment(v)

            v = torch.from_numpy(v).unsqueeze(0).repeat(3,1,1,1)  # [3,128,128,128]
            crops.append(v)

        crops = torch.stack(crops, dim=0)  # [K,3,128,128,128]
        return crops, torch.tensor(y).long(), pid

class FusionDataset(Dataset):
    def __init__(self, eeg_ds: EEGDataset, mri_ds: MRIDataset):
        self.eeg_map = {pid: i for i, (_,_,pid) in enumerate(eeg_ds.samples)}
        self.mri_map = {pid: i for i, (_,_,pid) in enumerate(mri_ds.samples)}
        self.ids = sorted(list(set(self.eeg_map).intersection(self.mri_map)))
        self.eeg_ds = eeg_ds
        self.mri_ds = mri_ds

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        xe, y, _  = self.eeg_ds[self.eeg_map[pid]]
        xm, y2, _ = self.mri_ds[self.mri_map[pid]]
        assert y.item() == y2.item()
        return xe, xm, y, pid

def _labels_for_sampler(ds):
    if isinstance(ds, FusionDataset):
        return [int(selfy) for _, selfy, _ in [ds.eeg_ds.samples[ds.eeg_map[pid]] for pid in ds.ids]]
    else:
        return [int(l) for (_, l, _) in ds.samples]

def make_loader(ds, batch_size):
    labels = _labels_for_sampler(ds)
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sample_w = [float(weights[l]) for l in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ====================== Models ======================

class EEGEncoderViT_B16(nn.Module):
    """
    ViT-B/16 for 17-ch scalogram:
    - Adapt 3->17 patch embed via weight-mean init
    - Use cls-token output (768-d), project to 512-d
    """
    def __init__(self, out_dim=512, freeze_until='encoder.blocks.9'):
        super().__init__()
        base = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # patch embedding 3 -> 17
        patch = base.conv_proj  # Conv2d(3, 768, kernel=16, stride=16)
        new_patch = nn.Conv2d(17, patch.out_channels, kernel_size=patch.kernel_size,
                              stride=patch.stride, padding=patch.padding, bias=False)
        with torch.no_grad():
            new_patch.weight.copy_(patch.weight.mean(1, keepdim=True).repeat(1, 17, 1, 1))
        base.conv_proj = new_patch

        self.vit = base
        self.vit.heads = nn.Identity()  # return 768-d cls-token
        self.proj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(768, out_dim))

        # freeze all, then unfreeze tail blocks + layer norms + proj
        for p in self.parameters(): p.requires_grad = False
        for name, p in self.vit.named_parameters():
            if (freeze_until is None) or name.startswith(freeze_until) or ('encoder.ln' in name):
                p.requires_grad = True
        for p in self.proj.parameters(): p.requires_grad = True

    def forward(self, x):  # [B,17,224,224]
        z = self.vit(x)      # [B, 768]
        z = self.proj(z)     # [B, 512]
        return F.normalize(z, dim=-1)

class MRIEncoder3D(nn.Module):
    """
    r3d_18 backbone with attention pooling over K crops.
    forward(xK): xK [B,K,3,128,128,128] -> [B,512]
    """
    def __init__(self, out_dim=512, unfreeze_layers=('layer4',)):
        super().__init__()
        base = vmodels.r3d_18(weights=vmodels.R3D_18_Weights.DEFAULT)
        self.backbone = base
        in_feat = base.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(in_feat, in_feat), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(in_feat, out_dim))
        # attention over crops
        self.att_q = nn.Linear(out_dim, out_dim, bias=False)
        self.att_k = nn.Linear(out_dim, out_dim, bias=False)

        self._freeze_all(); self._unfreeze_layers(unfreeze_layers)

    def _freeze_all(self):
        for p in self.parameters(): p.requires_grad=False

    def _unfreeze_layers(self, names=('layer4',)):
        for n, m in self.backbone.named_children():
            if n in names:
                for p in m.parameters(): p.requires_grad=True
        for p in self.proj.parameters(): p.requires_grad=True
        for p in self.att_q.parameters(): p.requires_grad=True
        for p in self.att_k.parameters(): p.requires_grad=True

    def encode_one(self, x):  # [B,3,128,128,128]
        h = self.backbone(x)          # [B,C]
        z = self.proj(h)              # [B,D]
        return F.normalize(z, dim=-1)

    def forward(self, xK):  # [B,K,3,128,128,128]
        B, K = xK.shape[:2]
        x = xK.view(B*K, 3, 128,128,128)
        z = self.encode_one(x).view(B, K, -1)   # [B,K,D]
        q = self.att_q(z); k = self.att_k(z)
        att = torch.softmax((q*k).sum(-1), dim=1).unsqueeze(-1)  # [B,K,1]
        z_pool = (att * z).sum(1)               # [B,D]
        return F.normalize(z_pool, dim=-1)

class GatedFusionHead(nn.Module):
    def __init__(self, dim=512, n_classes=2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 1), nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Linear(3*dim, 2*dim), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(2*dim, n_classes)
        )

    def forward(self, e, m, return_fused=False):
        x = torch.cat([e, m], dim=-1)          # [B,1024]
        g = self.gate(x)                       # [B,1]
        fused_vec = g*e + (1-g)*m              # [B,512]
        fused = torch.cat([fused_vec, x], -1)  # [B,1536]
        logits = self.cls(fused)               # [B,2]
        if return_fused:
            return logits, g.squeeze(-1), fused_vec
        return logits, g.squeeze(-1)

# ====================== Losses & Eval ======================

def cosine_align_loss(e, m, eps=1e-8):
    return (1 - F.cosine_similarity(e, m, dim=-1).clamp(min=-1+eps, max=1-eps)).mean()

def contrastive_loss(e, m, tau=0.2):
    e = F.normalize(e, dim=-1); m = F.normalize(m, dim=-1)
    sim = e @ m.t() / tau
    pos = torch.arange(e.size(0), device=e.device)
    loss_em = F.cross_entropy(sim, pos)
    loss_me = F.cross_entropy(sim.t(), pos)
    return 0.5*(loss_em + loss_me)

@torch.no_grad()
def eval_epoch(eeg_enc, mri_enc, head, loader, split_name="test", tta=False):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    preds, gts, probs, gates = [], [], [], []
    total_ce, n_ce = 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        eeg_x, mri_xK, y, _ = batch
        eeg_x  = eeg_x.to(device, non_blocking=True)
        mri_xK = mri_xK.to(device, non_blocking=True)
        y      = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if not tta:
                e = eeg_enc(eeg_x); m = mri_enc(mri_xK)
                logits, g = head(e, m)
            else:
                logits_all, gates_all = [], []
                e0 = eeg_enc(eeg_x); m0 = mri_enc(mri_xK)
                l0, g0 = head(e0, m0); logits_all.append(l0); gates_all.append(g0)
                e1 = eeg_enc(eeg_x.flip(-1)); l1, g1 = head(e1, m0); logits_all.append(l1); gates_all.append(g1)
                e2 = eeg_enc(eeg_x.flip(-2)); l2, g2 = head(e2, m0); logits_all.append(l2); gates_all.append(g2)
                m3 = mri_enc(mri_xK.flip(-4)); l3, g3 = head(e0, m3); logits_all.append(l3); gates_all.append(g3)
                logits = torch.stack(logits_all, 0).mean(0)
                g      = torch.stack(gates_all, 0).mean(0)

        prob = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        total_ce += float(ce(logits, y).item()); n_ce += y.size(0)
        preds.append(pred.cpu()); gts.append(y.cpu()); probs.append(prob.cpu()); gates.append(g.cpu())

    if not preds:
        print(f"[{split_name}] No samples.")
        return {"acc": 0.0, "loss_ce": 0.0}

    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    probs = torch.cat(probs).numpy()
    gates = torch.cat(gates).numpy()

    acc = (preds == gts).mean()
    val_loss = total_ce / max(1, n_ce)

    print(f"[{split_name}] CE loss: {val_loss:.4f} | Acc: {acc:.4f}")
    print(confusion_matrix(gts, preds, labels=[0,1]))
    print(classification_report(gts, preds, labels=[0,1], target_names=["M","F"], digits=4, zero_division=0))
    return {"acc": float(acc), "loss_ce": float(val_loss), "preds": preds, "gts": gts, "probs": probs, "gates": gates}

# ====================== Plotting / Reporting ======================

def plot_curves(history, out_png_base):
    ep = np.arange(1, len(history)+1)
    tr_loss = [h["train_loss"] for h in history]
    tr_acc  = [h["train_acc"]  for h in history]
    va_loss = [h["val_loss"]   for h in history]
    va_acc  = [h["val_acc"]    for h in history]

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_loss, label="Train Loss")
    plt.plot(ep, va_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, out_png_base + "_loss.png"), dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_acc, label="Train Acc")
    plt.plot(ep, va_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, out_png_base + "_acc.png"), dpi=200); plt.close()

def save_history_csv(history, name="history_metrics.csv"):
    df = pd.DataFrame(history)
    df.index = np.arange(1, len(df)+1)
    df.index.name = "epoch"
    df.to_csv(os.path.join(OUT_DIR, name))

def plot_confusion_and_roc(gts, probs, out_cm="confusion_matrix.png", out_roc="roc_curve.png"):
    preds = probs.argmax(1)
    cm = confusion_matrix(gts, preds, labels=[0,1])

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["M","F"])
    plt.yticks(tick_marks, ["M","F"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_cm), dpi=200); plt.close()

    y_true_bin = np.eye(2)[gts]
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(6,5))
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"Class {i} AUC={roc_auc[i]:.3f}")
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], lw=2.0, linestyle="--", label=f"Micro AUC={roc_auc['micro']:.3f}")
    plt.plot([0,1], [0,1], linestyle=":", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend()
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, out_roc), dpi=200); plt.close()

def plot_tsne(fused_feats, gts, out_png="tsne_fused_test.png",
              perplexity=30, random_state=SEED):
    fused_feats = np.asarray(fused_feats); gts = np.asarray(gts)
    path = os.path.join(OUT_DIR, out_png)

    if fused_feats.ndim != 2 or fused_feats.shape[0] < 3:
        plt.figure(figsize=(6,4)); plt.axis('off')
        msg = f"t-SNE skipped: shape={fused_feats.shape}" if fused_feats.ndim != 2 else f"t-SNE skipped: n={fused_feats.shape[0]} < 3"
        plt.text(0.5, 0.5, msg, ha='center', va='center')
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        print("[t-SNE]", msg); return

    valid = np.isfinite(fused_feats).all(axis=1)
    fused_feats = fused_feats[valid]; gts = gts[valid]
    n = fused_feats.shape[0]
    p = max(2, min(perplexity, n-1))
    print(f"[t-SNE] n={n}, perplexity={p}")

    tsne = TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto", random_state=random_state)
    z = tsne.fit_transform(fused_feats)

    plt.figure(figsize=(6,5))
    for lab, name, marker in [(0,"M","o"), (1,"F","^")]:
        idx = (gts == lab)
        if idx.any():
            plt.scatter(z[idx,0], z[idx,1], s=18, marker=marker, label=name, alpha=0.8)
    plt.legend(); plt.title("t-SNE of Fused Test Embeddings")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

# ====================== Training ======================

def train_fusion(eeg_tr, eeg_te, mri_tr, mri_te):
    fus_train = FusionDataset(eeg_tr, mri_tr)
    fus_test  = FusionDataset(eeg_te, mri_te)

    tr_loader = make_loader(fus_train, BATCH)
    te_loader = DataLoader(fus_test, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    eeg_enc = EEGEncoderViT_B16(out_dim=512, freeze_until='encoder.blocks.9').to(device)
    mri_enc = MRIEncoder3D(out_dim=512, unfreeze_layers=('layer4',)).to(device)
    head    = GatedFusionHead(dim=512, n_classes=2).to(device)

    params = [
        {'params': [p for p in eeg_enc.parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': [p for p in mri_enc.parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': head.parameters(), 'lr': LR_HEAD},
    ]
    opt = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE_A + EPOCHS_STAGE_B)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    ce = nn.CrossEntropyLoss()
    history = []

    def one_phase(num_epochs, lambda_contrast=0.0, unfreeze=None):
        nonlocal history
        if unfreeze is not None:
            if 'eeg' in unfreeze:
                for n, p in eeg_enc.vit.named_parameters():
                    if any(n.startswith(prefix) for prefix in unfreeze['eeg']):
                        p.requires_grad = True
            if 'mri' in unfreeze:
                mri_enc._unfreeze_layers(unfreeze['mri'])

        best = 0.0
        for ep in range(1, num_epochs+1):
            eeg_enc.train(); mri_enc.train(); head.train()
            tot, corr, n = 0.0, 0, 0
            for eeg_x, mri_xK, y, _ in tqdm(tr_loader, desc=f"Epoch {ep}"):
                eeg_x  = eeg_x.to(device, non_blocking=True)
                mri_xK = mri_xK.to(device, non_blocking=True)
                y      = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    e = eeg_enc(eeg_x)
                    m = mri_enc(mri_xK)
                    logits, _ = head(e, m)
                    loss_cls   = ce(logits, y)
                    loss_align = cosine_align_loss(e, m)
                    loss_ctr   = contrastive_loss(e, m, tau=TAU_CONTRAST) if lambda_contrast>0 else 0.0
                    loss = loss_cls + LAMBDA_ALIGN*loss_align + lambda_contrast*loss_ctr
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                scaler.step(opt); scaler.update()

                tot += float(loss.item()) * y.size(0)
                pred = logits.argmax(1)
                corr += int((pred==y).sum().item()); n += y.size(0)

            sched.step()
            tr_loss = tot/max(1,n); tr_acc = corr/max(1,n)
            val_stats = eval_epoch(eeg_enc, mri_enc, head, te_loader, split_name="val/test", tta=False)
            te_acc = val_stats["acc"]; te_loss = val_stats["loss_ce"]
            best = max(best, te_acc)
            print(f"[Stage] ep {ep} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {te_loss:.4f} acc {te_acc:.4f} | best {best:.4f}")

            history.append({
                "stage_contrast": float(lambda_contrast),
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(te_loss),
                "val_acc": float(te_acc),
            })
        return best

    print("\n=== Stage A: warm-up (small contrast) ===")
    bestA = one_phase(EPOCHS_STAGE_A, lambda_contrast=LAMBDA_CONTR_A, unfreeze=None)

    print("\n=== Stage B: finetune (higher contrast, deeper unfreeze) ===")
    bestB = one_phase(EPOCHS_STAGE_B, lambda_contrast=LAMBDA_CONTR_B,
                      unfreeze={'eeg': ['encoder.blocks.8', 'encoder.blocks.9', 'encoder.blocks.10', 'encoder.blocks.11'],
                                'mri': ('layer3','layer4')})

    # Save curves & history
    save_history_csv(history, name="history_metrics.csv")
    plot_curves(history, out_png_base="curves_train_val")

    print(f"Done. Best Acc StageA={bestA:.4f}, StageB={bestB:.4f}")
    print("Evaluating with light TTA & collecting outputs...")

    test_stats, fused_feats, pids = collect_test_outputs(eeg_enc, mri_enc, head, te_loader, use_tta=True)
    gts   = test_stats["gts"]
    probs = test_stats["probs"]
    preds = probs.argmax(1)

    # Save per-sample predictions
    df_pred = pd.DataFrame({
        "pid": pids,
        "y_true": gts,
        "y_pred": preds,
        "prob_M": probs[:,0],
        "prob_F": probs[:,1],
        "gate":  test_stats["gates"]
    })
    df_pred.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # Confusion matrix & ROC
    plot_confusion_and_roc(gts, probs,
                           out_cm="confusion_matrix.png",
                           out_roc="roc_curve.png")

    # t-SNE
    plot_tsne(fused_feats, gts, out_png="tsne_fused_test.png", perplexity=30)

    print(f"Outputs saved under ./{OUT_DIR}/")
    acc_tta = (preds == gts).mean()
    print(f"TTA Acc (computed from saved probs): {acc_tta:.4f}")

    # Save checkpoints
    torch.save({
        'eeg_enc': eeg_enc.state_dict(),
        'mri_enc': mri_enc.state_dict(),
        'head': head.state_dict(),
        'config': {
            'K_MRI_CROPS': K_MRI_CROPS,
            'LAMBDA_ALIGN': LAMBDA_ALIGN,
            'LAMBDA_CONTR_A': LAMBDA_CONTR_A,
            'LAMBDA_CONTR_B': LAMBDA_CONTR_B,
            'TAU_CONTRAST': TAU_CONTRAST,
            'eeg_encoder': 'ViT-B/16',
        }
    }, os.path.join(CKPT_DIR, "fusion_GEA_MVA_CMCR_ViT.pth"))
    print(f"✅ Saved: {os.path.join(CKPT_DIR, 'fusion_GEA_MVA_CMCR_ViT.pth')}")

    return eeg_enc, mri_enc, head

@torch.no_grad()
def collect_test_outputs(eeg_enc, mri_enc, head, loader, use_tta=True):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    all_probs, all_gts, all_gates, all_pids = [], [], [], []
    fused_list = []

    for eeg_x, mri_xK, y, pids in loader:
        eeg_x  = eeg_x.to(device, non_blocking=True)
        mri_xK = mri_xK.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if not use_tta:
                e0 = eeg_enc(eeg_x); m0 = mri_enc(mri_xK)
                logits, g, fused = head(e0, m0, return_fused=True)
            else:
                logits_all, gates_all, fused_all = [], [], []
                e0 = eeg_enc(eeg_x); m0 = mri_enc(mri_xK)
                l0, g0, f0 = head(e0, m0, return_fused=True)
                logits_all.append(l0); gates_all.append(g0); fused_all.append(f0)
                e1 = eeg_enc(eeg_x.flip(-1)); l1, g1, f1 = head(e1, m0, return_fused=True)
                logits_all.append(l1); gates_all.append(g1); fused_all.append(f1)
                e2 = eeg_enc(eeg_x.flip(-2)); l2, g2, f2 = head(e2, m0, return_fused=True)
                logits_all.append(l2); gates_all.append(g2); fused_all.append(f2)
                m3 = mri_enc(mri_xK.flip(-4)); l3, g3, f3 = head(e0, m3, return_fused=True)
                logits = torch.stack(logits_all, 0).mean(0)
                g      = torch.stack(gates_all, 0).mean(0)
                fused  = torch.stack(fused_all, 0).mean(0)

        prob = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(prob)
        all_gts.append(y.numpy())
        all_gates.append(g.cpu().numpy())
        all_pids.extend(list(pids))
        fused_list.append(fused.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,2))
    gts   = np.concatenate(all_gts, axis=0)   if all_gts else np.array([])
    gates = np.concatenate(all_gates, axis=0) if all_gates else np.array([])
    fused = np.concatenate(fused_list, axis=0) if fused_list else np.zeros((0,512))
    return {"probs": probs, "gts": gts, "gates": gates}, fused, all_pids

# ====================== Split helpers & Build ======================

def count_ids(ids, gmap):
    m = sum(1 for pid in ids if gmap.get(pid,1)==0)
    f = sum(1 for pid in ids if gmap.get(pid,1)==1)
    return m,f

def print_split_stats(train_ids, test_ids, gmap):
    m_tr, f_tr = count_ids(train_ids, gmap)
    m_te, f_te = count_ids(test_ids,  gmap)
    print(f"[Subjects] Train: M={m_tr} F={f_tr} (Total={len(train_ids)})")
    print(f"[Subjects] Test : M={m_te} F={f_te} (Total={len(test_ids)})")

def build_subject_lists(eeg_dir, mri_dir, gmap):
    eeg_full = EEGDataset(eeg_dir, gmap, train=True)
    mri_full = MRIDataset(mri_dir, gmap, train=True, k_crops=K_MRI_CROPS)
    eeg_ids_all = [extract_pid(p) for (p,_,_) in eeg_full.samples]
    mri_ids_all = [extract_pid(p) for (p,_,_) in mri_full.samples]
    common_ids = sorted(list(set(eeg_ids_all).intersection(mri_ids_all)))
    return eeg_full, mri_full, common_ids

def build_test_split_m2f1(common_ids, gmap, *, test_frac=0.20, seed=SEED):
    rng = random.Random(seed)
    males   = [pid for pid in common_ids if gmap.get(pid,1)==0]
    females = [pid for pid in common_ids if gmap.get(pid,1)==1]
    rng.shuffle(males); rng.shuffle(females)
    N  = len(common_ids)
    T0 = max(1, int(round(test_frac*N)))
    k  = min(T0//3, len(males)//2, len(females))
    if k==0 and len(males)>=2 and len(females)>=1: k=1
    test_ids = set(males[:2*k] + females[:k])
    train_ids = set(pid for pid in common_ids if pid not in test_ids)
    print(f"[Split] Total={N} | Target≈{T0} | Test={len(test_ids)}")
    return train_ids, test_ids

def subset_dataset(ds, keep_ids, is_train):
    keep = []
    for (p,l,pid) in ds.samples:
        if pid in keep_ids:
            keep.append((p,l,pid))
    if isinstance(ds, EEGDataset):
        new = EEGDataset(EEG_DIR, gender_map, train=is_train)
    else:
        new = MRIDataset(MRI_DIR, gender_map, train=is_train, k_crops=K_MRI_CROPS)
    new.samples = keep
    return new

# ====================== Main ======================
if __name__ == "__main__":
    eeg_full, mri_full, common_ids = build_subject_lists(EEG_DIR, MRI_DIR, gender_map)
    train_ids, test_ids = build_test_split_m2f1(common_ids, gender_map, test_frac=0.20, seed=SEED)
    print_split_stats(train_ids, test_ids, gender_map)

    eeg_tr = subset_dataset(eeg_full, train_ids, True)
    eeg_te = subset_dataset(eeg_full, test_ids,  False)
    mri_tr = subset_dataset(mri_full, train_ids, True)
    mri_te = subset_dataset(mri_full, test_ids,  False)

    eeg_enc, mri_enc, head = train_fusion(eeg_tr, eeg_te, mri_tr, mri_te)

    print(f"✅ All done. Check '{OUT_DIR}/' for curves, ROC, CM, t-SNE, predictions; '{CKPT_DIR}/' for weights.")
