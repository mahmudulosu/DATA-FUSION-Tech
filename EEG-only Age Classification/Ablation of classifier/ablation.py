# -*- coding: utf-8 -*-
"""
EEG+MRI Fusion — Ablation of classifier inputs:
  1) CONCAT:    x=[e ; m]
  2) FUSED:     x=f, where f = g*e + (1-g)*m
  3) GATEDCAT:  x=[f ; e ; m]

Encoders:
- EEG: ConvNeXt-Tiny (17ch stem) + ElectrodeAttention -> 512-D unit-norm
- MRI: r3d_18 on K=3 center-biased 128^3 crops + attention pooling -> 512-D unit-norm

Loss:
  L = CE + lambda_align * (1 - cos) + lambda_contrast * InfoNCE  (symmetric, τ)

Outputs per variant (under ./outputsss):
  - history_{variant}.csv, curves_{variant}_*.png
  - confusion_matrix_{variant}.png, roc_curve_{variant}.png
  - tsne_fused_test_{variant}.png
  - test_predictions_{variant}.csv
  - metrics_summary_{variant}.csv
Combined summary: metrics_summary_all.csv
"""

import os, random, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.models.video as vmodels
from scipy.ndimage import zoom, rotate

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.manifold import TSNE
from tqdm import tqdm
import pandas as pd

# ====================== Config & Device ======================
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------- EDIT THESE ---------
EEG_DIR = r"C:\\research\\EEG_Domain\\eye_openscalo"     # .npy, shape (224,224,17)
MRI_DIR = r"C:/research/MRI/structural_MRI"              # .nii/.nii.gz
CSV     = r"C:/research/MRI/participants_LSD_andLEMON.csv"  # participant_id, gender
# ------------------------------

os.makedirs("outputsss", exist_ok=True)

# Dataloader knobs (Windows-safe)
NUM_WORKERS = 0
PIN_MEMORY  = True

# Training knobs
BATCH = 4
EPOCHS_STAGE_A = 25
EPOCHS_STAGE_B = 25
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
# map: 0 = Male, 1 = Female
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
    """Returns K center-biased 128^3 crops for MVA. Output: [K,3,128,128,128]"""
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
        vol = nib.load(path).get_fdata().astype(np.float32)  # [D,H,W]
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
            if self.train: v = mri_augment(v)
            v = torch.from_numpy(v).unsqueeze(0).repeat(3,1,1,1)  # [3,128,128,128]
            crops.append(v)

        crops = torch.stack(crops, dim=0)  # [K,3,128,128,128]
        return crops, torch.tensor(y).long(), pid

class FusionDataset(Dataset):
    def __init__(self, eeg_ds: EEGDataset, mri_ds: MRIDataset):
        self.eeg_map = {pid: i for i, (_,_,pid) in enumerate(eeg_ds.samples)}
        self.mri_map = {pid: i for i, (_,_,pid) in enumerate(mri_ds.samples)}
        self.ids = sorted(list(set(self.eeg_map).intersection(self.mri_map)))
        self.eeg_ds = eeg_ds; self.mri_ds = mri_ds

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

class ElectrodeAttention(nn.Module):
    def __init__(self, C=17, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Conv2d(C, max(1, C//r), 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, C//r), C, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):  # [B,17,H,W]
        w = self.mlp(self.pool(x))
        return x * w

class EEGEncoderConvNeXt(nn.Module):
    def __init__(self, out_dim=512, unfreeze_stages=(3,4)):
        super().__init__()
        self.eaa = ElectrodeAttention(C=17, r=4)
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        stem = base.features[0][0]
        new_stem = nn.Conv2d(17, stem.out_channels, kernel_size=stem.kernel_size,
                             stride=stem.stride, padding=stem.padding, bias=False)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.mean(1, keepdim=True).repeat(1,17,1,1))
        base.features[0][0] = new_stem
        self.backbone = base.features
        self.backbone_ln = nn.LayerNorm(768, eps=1e-6)
        self.proj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(768, out_dim))
        self._freeze_all(); self._unfreeze_stages(unfreeze_stages)

    def _freeze_all(self):
        for p in self.parameters(): p.requires_grad=False

    def _unfreeze_stages(self, stages=(3,4)):
        for i, block in enumerate(self.backbone):
            if i in stages:
                for p in block.parameters(): p.requires_grad=True
        for p in self.backbone_ln.parameters(): p.requires_grad=True
        for p in self.proj.parameters(): p.requires_grad=True

    def forward(self, x):  # [B,17,224,224]
        x = self.eaa(x)
        x = self.backbone(x)          # [B,768,h,w]
        x = x.mean([-2,-1])           # [B,768]
        x = self.backbone_ln(x)
        x = self.proj(x)              # [B,512]
        return F.normalize(x, dim=-1)

class MRIEncoder3D(nn.Module):
    """r3d_18 + attention over K crops -> 512-D unit-norm"""
    def __init__(self, out_dim=512, unfreeze_layers=('layer4',)):
        super().__init__()
        base = vmodels.r3d_18(weights=vmodels.R3D_18_Weights.DEFAULT)
        self.backbone = base
        in_feat = base.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(in_feat, in_feat), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(in_feat, out_dim))
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

# ---- Heads (Concat / Fused / GatedConcat) ----

class Gate(nn.Module):
    def __init__(self, d=512, h=256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2*d, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())
    def forward(self, e, m):
        u = torch.cat([e, m], -1)      # [B,2d]
        g = self.g(u)                  # [B,1]
        f = g*e + (1-g)*m              # [B,d]
        return f, g.squeeze(-1)

class HeadConcat(nn.Module):           # x=[e;m]
    def __init__(self, d=512, h=1024, ncls=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*d, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, ncls))
    def forward(self, e, m, f=None):   # f unused
        return self.net(torch.cat([e, m], -1))

class HeadFusedOnly(nn.Module):        # x=f
    def __init__(self, d=512, h=512, ncls=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, ncls))
    def forward(self, e, m, f):        # needs f
        return self.net(f)

class HeadGatedConcat(nn.Module):      # x=[f;e;m]
    def __init__(self, d=512, h=1024, ncls=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*d, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, ncls))
    def forward(self, e, m, f):
        return self.net(torch.cat([f, e, m], -1))

# ====================== Losses & Eval ======================

def cosine_align_loss(e, m, eps=1e-8):
    return (1 - F.cosine_similarity(e, m, dim=-1).clamp(min=-1+eps, max=1-eps)).mean()

def contrastive_loss(e, m, tau=0.2):
    e = F.normalize(e, dim=-1); m = F.normalize(m, dim=-1)
    sim = e @ m.t() / tau
    pos = torch.arange(e.size(0), device=e.device)
    return 0.5*(F.cross_entropy(sim, pos) + F.cross_entropy(sim.t(), pos))

@torch.no_grad()
def eval_epoch(eeg_enc, mri_enc, gate, head, loader, split_name="val/test", tta=False):
    eeg_enc.eval(); mri_enc.eval(); head.eval(); gate.eval()
    preds, gts, probs, gates = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    tot, n = 0.0, 0

    for eeg_x, mri_xK, y, _ in loader:
        eeg_x  = eeg_x.to(device, non_blocking=True)
        mri_xK = mri_xK.to(device, non_blocking=True)
        y      = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if not tta:
                e = eeg_enc(eeg_x); m = mri_enc(mri_xK)
                f, g = gate(e, m)
                logits = head(e, m, f)
            else:
                logits_all, gates_all = [], []
                e0 = eeg_enc(eeg_x); m0 = mri_enc(mri_xK); f0, g0 = gate(e0, m0)
                logits_all.append(head(e0, m0, f0)); gates_all.append(g0)
                e1 = eeg_enc(eeg_x.flip(-1)); f1, g1 = gate(e1, m0)
                logits_all.append(head(e1, m0, f1)); gates_all.append(g1)
                e2 = eeg_enc(eeg_x.flip(-2)); f2, g2 = gate(e2, m0)
                logits_all.append(head(e2, m0, f2)); gates_all.append(g2)
                m3 = mri_enc(mri_xK.flip(-4)); f3, g3 = gate(e0, m3)
                logits_all.append(head(e0, m3, f3)); gates_all.append(g3)
                logits = torch.stack(logits_all,0).mean(0)
                g      = torch.stack(gates_all,0).mean(0)

        prob = F.softmax(logits, dim=1)
        pred = logits.argmax(1)
        tot += float(ce(logits, y).item()); n += y.size(0)
        preds.append(pred.cpu()); gts.append(y.cpu()); probs.append(prob.cpu()); gates.append(g.cpu())

    if not preds:
        return {"acc":0.0,"loss_ce":0.0}
    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    probs = torch.cat(probs).numpy()
    gates = torch.cat(gates).numpy()

    acc = (preds==gts).mean()
    val_loss = tot/max(1,n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1m = f1_score(gts, preds, average="macro", zero_division=0)
        yb = np.eye(2)[gts]
        try:  auroc = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
        except: auroc = float('nan')
        try:  auprc = average_precision_score(yb, probs, average="macro")
        except: auprc = float('nan')

    print(f"[{split_name}] CE {val_loss:.4f} | Acc {acc:.4f} | F1 {f1m:.4f} | AUROC {auroc:.4f} | AUPRC {auprc:.4f}")
    return {"acc":float(acc),"loss_ce":float(val_loss),"f1_macro":float(f1m),
            "auroc_macro":float(auroc),"auprc_macro":float(auprc),
            "probs":probs,"gts":gts,"gates":gates}

# ====================== Plotting / Reporting ======================

def plot_curves(history, out_stub):
    ep = np.arange(1, len(history)+1)
    tr_loss = [h["train_loss"] for h in history]
    tr_acc  = [h["train_acc"]  for h in history]
    va_loss = [h["val_loss"]   for h in history]
    va_acc  = [h["val_acc"]    for h in history]
    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_loss, label="Train Loss")
    plt.plot(ep, va_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_stub+"_loss.png", dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(ep, tr_acc, label="Train Acc")
    plt.plot(ep, va_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_stub+"_acc.png", dpi=200); plt.close()

def save_history_csv(history, out_csv):
    df = pd.DataFrame(history); df.index = np.arange(1, len(df)+1); df.index.name = "epoch"
    df.to_csv(out_csv)

def plot_confusion_and_roc(gts, probs, out_cm, out_roc):
    preds = probs.argmax(1); cm = confusion_matrix(gts, preds, labels=[0,1])

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar()
    tick_marks = np.arange(2); plt.xticks(tick_marks, ["M","F"]); plt.yticks(tick_marks, ["M","F"])
    thr = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j], 'd'), ha="center", va="center",
                     color="white" if cm[i,j]>thr else "black")
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout(); plt.savefig(out_cm, dpi=200); plt.close()

    yb = np.eye(2)[gts]; plt.figure(figsize=(6,5))
    for i in range(2):
        try:
            fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i]); rocA = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"Class {i} AUC={rocA:.3f}")
        except: pass
    try:
        fpr, tpr, _ = roc_curve(yb.ravel(), probs.ravel()); plt.plot(fpr, tpr, lw=2, ls="--", label=f"Micro AUC={auc(fpr,tpr):.3f}")
    except: pass
    plt.plot([0,1],[0,1], ls=":", lw=1); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend()
    plt.grid(True); plt.tight_layout(); plt.savefig(out_roc, dpi=200); plt.close()

def plot_tsne(fused_feats, gts, out_png, perplexity=30, random_state=SEED):
    fused_feats = np.asarray(fused_feats); gts = np.asarray(gts)
    if fused_feats.ndim != 2 or fused_feats.shape[0] < 3:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5, 0.5, f"t-SNE skipped: n={fused_feats.shape[0] if fused_feats.ndim==2 else 'NA'}",
                 ha='center', va='center')
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(); return
    valid = np.isfinite(fused_feats).all(axis=1); fused_feats = fused_feats[valid]; gts = gts[valid]
    n = fused_feats.shape[0]; p = max(2, min(perplexity, n-1))
    tsne = TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto", random_state=random_state)
    z = tsne.fit_transform(fused_feats)
    plt.figure(figsize=(6,5))
    for lab, name, mk in [(0,"M","o"), (1,"F","^")]:
        idx = (gts==lab)
        if idx.any(): plt.scatter(z[idx,0], z[idx,1], s=18, marker=mk, label=name, alpha=0.8)
    plt.legend(); plt.title("t-SNE of Fused Test Embeddings")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ====================== Param counting ======================

def count_params_million(*models, only_trainable=False) -> float:
    total = 0
    for m in models:
        if only_trainable: total += sum(p.numel() for p in m.parameters() if p.requires_grad)
        else:              total += sum(p.numel() for p in m.parameters())
    return total / 1e6

# ====================== Training (runs one variant) ======================

def train_one_variant(variant_name, eeg_tr, eeg_te, mri_tr, mri_te):
    assert variant_name in {"concat","fused","gated"}, "variant must be one of {'concat','fused','gated'}"
    suf = {"concat":"_concat","fused":"_fused","gated":"_gated"}[variant_name]

    fus_train = FusionDataset(eeg_tr, mri_tr)
    fus_test  = FusionDataset(eeg_te, mri_te)
    tr_loader = make_loader(fus_train, BATCH)
    te_loader = DataLoader(fus_test, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    eeg_enc = EEGEncoderConvNeXt(out_dim=512, unfreeze_stages=(3,4)).to(device)
    mri_enc = MRIEncoder3D(out_dim=512,  unfreeze_layers=('layer4',)).to(device)

    gate = Gate(d=512, h=256).to(device)  # used to produce f,g; still used for CONCAT to keep compute fair

    if variant_name == "concat":    head = HeadConcat(d=512, h=1024, ncls=2).to(device)
    elif variant_name == "fused":   head = HeadFusedOnly(d=512, h=512,  ncls=2).to(device)
    else:                           head = HeadGatedConcat(d=512, h=1024, ncls=2).to(device)

    params = [
        {'params': [p for p in eeg_enc.parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': [p for p in mri_enc.parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': list(gate.parameters()) + list(head.parameters()), 'lr': LR_HEAD},
    ]
    opt = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE_A + EPOCHS_STAGE_B)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    ce = nn.CrossEntropyLoss()

    history = []

    def one_phase(num_epochs, lambda_contrast=0.0, unfreeze=None):
        nonlocal history
        if unfreeze is not None:
            if 'eeg' in unfreeze: eeg_enc._unfreeze_stages(unfreeze['eeg'])
            if 'mri' in unfreeze: mri_enc._unfreeze_layers(unfreeze['mri'])

        for ep in range(1, num_epochs+1):
            eeg_enc.train(); mri_enc.train(); gate.train(); head.train()
            tot, corr, n = 0.0, 0, 0
            for eeg_x, mri_xK, y, _ in tqdm(tr_loader, desc=f"{variant_name.upper()} | Epoch {ep}"):
                eeg_x  = eeg_x.to(device, non_blocking=True)
                mri_xK = mri_xK.to(device, non_blocking=True)
                y      = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    e = eeg_enc(eeg_x); m = mri_enc(mri_xK)
                    f, g = gate(e, m)
                    logits = head(e, m, f)
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

            val_stats = eval_epoch(eeg_enc, mri_enc, gate, head, te_loader, split_name="val/test", tta=False)
            history.append({
                "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                "val_loss": float(val_stats["loss_ce"]), "val_acc": float(val_stats["acc"]),
                "val_f1_macro": float(val_stats.get("f1_macro", float('nan'))),
                "val_auroc_macro": float(val_stats.get("auroc_macro", float('nan'))),
                "val_auprc_macro": float(val_stats.get("auprc_macro", float('nan'))),
            })
        return

    print(f"\n=== {variant_name.upper()} :: Stage A (warm-up) ===")
    one_phase(EPOCHS_STAGE_A, lambda_contrast=LAMBDA_CONTR_A, unfreeze=None)
    print(f"\n=== {variant_name.upper()} :: Stage B (fine-tune) ===")
    one_phase(EPOCHS_STAGE_B, lambda_contrast=LAMBDA_CONTR_B,
              unfreeze={'eeg': (2,3,4), 'mri': ('layer3','layer4')})

    # Save curves & history
    save_history_csv(history, f"outputsss/history{ suf }.csv")
    plot_curves(history, f"outputsss/curves{ suf }")

    # Final eval with TTA & collect outputs
    test_stats = eval_epoch(eeg_enc, mri_enc, gate, head, te_loader, split_name="test_TTA", tta=True)
    gts   = test_stats["gts"]; probs = test_stats["probs"]; preds = probs.argmax(1)
    acc_final = float((preds == gts).mean()) if len(gts)>0 else float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_final = float(f1_score(gts, preds, average='macro', zero_division=0)) if len(gts)>0 else float('nan')
        try:
            yb = np.eye(2)[gts]; auroc_final = float(roc_auc_score(yb, probs, average='macro', multi_class='ovr'))
        except Exception: auroc_final = float('nan')
        try:
            auprc_final = float(average_precision_score(yb, probs, average='macro'))
        except Exception: auprc_final = float('nan')

    paramsM = float(count_params_million(eeg_enc, mri_enc, gate, head, only_trainable=False))

    df_summary = pd.DataFrame({
        "Variant":[variant_name],
        "Acc":[acc_final], "Macro-F1":[f1_final],
        "AUROC":[auroc_final], "AUPRC":[auprc_final],
        "Params (M)":[paramsM],
    })
    df_summary.to_csv(f"outputsss/metrics_summary{ suf }.csv", index=False)

    # Per-sample predictions
    df_pred = pd.DataFrame({
        "y_true": gts,
        "y_pred": preds,
        "prob_M": probs[:,0] if probs.size else [],
        "prob_F": probs[:,1] if probs.size else [],
        "gate":  test_stats["gates"],
    })
    df_pred.to_csv(f"outputsss/test_predictions{ suf }.csv", index=False)

    # Plots
    if len(gts)>0 and probs.size:
        plot_confusion_and_roc(gts, probs,
                               out_cm=f"outputsss/confusion_matrix{ suf }.png",
                               out_roc=f"outputsss/roc_curve{ suf }.png")

    # For visualization we still plot t-SNE of the *fused* representation
    # (use the mean fused vector from TTA forward in eval_epoch if you wish;
    #  here we just recompute without saving features.)
    print(f"[{variant_name}] done. Acc={acc_final:.4f}, F1={f1_final:.4f}, AUROC={auroc_final:.4f}")

    return df_summary

# ====================== Split helpers & Build ======================

def count_ids(ids, gmap):
    m = sum(1 for pid in ids if gmap.get(pid, 1) == 0)
    f = sum(1 for pid in ids if gmap.get(pid, 1) == 1)
    return m, f

def print_split_stats(train_ids, test_ids, gmap):
    m_tr, f_tr = count_ids(train_ids, gmap)
    m_te, f_te = count_ids(test_ids,  gmap)
    print(f"[Subjects] Train: M={m_tr} F={f_tr} (Total={len(train_ids)})")
    print(f"[Subjects] Test : M={m_te} F={f_te} (Total={len(test_ids)})")

def build_subject_lists(eeg_dir, mri_dir, gmap):
    eeg_full = EEGDataset(eeg_dir, gmap, train=True)
    mri_full = MRIDataset(mri_dir, gmap, train=True, k_crops=K_MRI_CROPS)
    eeg_ids_all = [extract_pid(p) for (p, _, _) in eeg_full.samples]
    mri_ids_all = [extract_pid(p) for (p, _, _) in mri_full.samples]
    common_ids = sorted(list(set(eeg_ids_all).intersection(mri_ids_all)))
    return eeg_full, mri_full, common_ids

def build_test_split_m2f1(common_ids, gmap, *, test_frac=0.20, seed=SEED):
    rng = random.Random(seed)
    males   = [pid for pid in common_ids if gmap.get(pid, 1) == 0]
    females = [pid for pid in common_ids if gmap.get(pid, 1) == 1]
    rng.shuffle(males); rng.shuffle(females)
    N  = len(common_ids)
    T0 = max(1, int(round(test_frac * N)))
    k  = min(T0 // 3, len(males) // 2, len(females))
    if k == 0 and len(males) >= 2 and len(females) >= 1: k = 1
    test_ids = set(males[:2*k] + females[:k])
    train_ids = set(pid for pid in common_ids if pid not in test_ids)
    print(f"[Split] Total={N} | Target≈{T0} | Test={len(test_ids)}")
    return train_ids, test_ids

def subset_dataset(ds, keep_ids, is_train):
    keep = []
    for (p, l, pid) in ds.samples:
        if pid in keep_ids: keep.append((p, l, pid))
    if isinstance(ds, EEGDataset):
        new = EEGDataset(EEG_DIR, gender_map, train=is_train)
    else:
        new = MRIDataset(MRI_DIR, gender_map, train=is_train, k_crops=K_MRI_CROPS)
    new.samples = keep
    return new

# ====================== Main ======================
if __name__ == "__main__":
    # Build full sets & subject split
    eeg_full, mri_full, common_ids = build_subject_lists(EEG_DIR, MRI_DIR, gender_map)
    train_ids, test_ids = build_test_split_m2f1(common_ids, gender_map, test_frac=0.20, seed=SEED)
    print_split_stats(train_ids, test_ids, gender_map)

    eeg_tr = subset_dataset(eeg_full, train_ids, True)
    eeg_te = subset_dataset(eeg_full, test_ids,  False)
    mri_tr = subset_dataset(mri_full, train_ids, True)
    mri_te = subset_dataset(mri_full, test_ids,  False)

    # Run all variants and aggregate a single summary CSV
    summaries = []
    for variant in ["concat","fused","gated"]:
        summaries.append(train_one_variant(variant, eeg_tr, eeg_te, mri_tr, mri_te))
    df_all = pd.concat(summaries, ignore_index=True)
    df_all.to_csv("outputsss/metrics_summary_all.csv", index=False)
    print("\n✅ Saved combined comparison: outputsss/metrics_summary_all.csv")
