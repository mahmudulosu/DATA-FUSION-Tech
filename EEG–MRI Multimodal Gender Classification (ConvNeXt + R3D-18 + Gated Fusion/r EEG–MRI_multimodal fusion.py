# -*- coding: utf-8 -*-
import os, random
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.models.video as vmodels
from scipy.ndimage import zoom, rotate

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ================= Repro & device =================
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================= Paths (edit) =================
EEG_DIR = r"C:\research\EEG_Domain\eye_openscalo"   # .npy (224,224,17)
MRI_DIR = r"C:/research/MRI/structural_MRI"                  # .nii/.nii.gz
CSV     = r"C:/research/MRI/participants_LSD_andLEMON.csv"  # columns: participant_id, gender

# ================= Utils =================
def zscore(x, eps=1e-6):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def specaugment_like(x, time_mask_p=0.1, freq_mask_p=0.1):
    # x: [C,H,W] where H=freq, W=time (scalogram). Apply rectangle dropouts on (H,W).
    C, H, W = x.shape
    x = x.clone()
    if random.random() < 0.8:
        # time mask
        tw = int(W * time_mask_p)
        if tw > 0:
            t0 = random.randint(0, max(0, W - tw))
            x[:, :, t0:t0+tw] = 0
        # freq mask
        fh = int(H * freq_mask_p)
        if fh > 0:
            f0 = random.randint(0, max(0, H - fh))
            x[:, f0:f0+fh, :] = 0
    return x

def mri_augment(vol):
    # vol: [D,H,W], float32 in [0,1]
    if random.random() < 0.5: vol = np.flip(vol, 0).copy()
    if random.random() < 0.5: vol = np.flip(vol, 1).copy()
    if random.random() < 0.5: vol = np.flip(vol, 2).copy()
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        vol = rotate(vol, angle, axes=(1,2), reshape=False, order=1, mode='nearest')
    return vol

def extract_pid(path_or_name):
    # assumes leading token before first underscore is the participant id
    return os.path.basename(path_or_name).split("_")[0]

# ================= Data =================
import pandas as pd
gender_df = pd.read_csv(CSV)
# Map: 0 = Male (M), 1 = Female (F)
gender_map = {row['participant_id']: (0 if str(row['gender']).strip().upper() == 'M' else 1)
              for _, row in gender_df.iterrows()}

class EEGDataset(Dataset):
    def __init__(self, eeg_dir, labels, train=True):
        self.samples = []
        for fn in os.listdir(eeg_dir):
            if not fn.endswith(".npy"): continue
            pid = extract_pid(fn)
            if pid in labels:
                self.samples.append((os.path.join(eeg_dir, fn), labels[pid], pid))
        self.train = train

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        arr = np.load(path).astype(np.float32)    # (224,224,17)
        # per-electrode zscore
        for c in range(arr.shape[-1]):
            arr[..., c] = zscore(arr[..., c])
        x = torch.from_numpy(arr).permute(2,0,1)  # [17,224,224]
        if self.train:
            x = specaugment_like(x, 0.08, 0.08)
        return x, torch.tensor(label, dtype=torch.long), pid

class MRIDataset(Dataset):
    def __init__(self, mri_dir, labels, train=True, target_size=128, target_depth=96):
        self.samples = []
        for fn in os.listdir(mri_dir):
            if not fn.endswith((".nii",".nii.gz")): continue
            pid = extract_pid(fn)
            if pid in labels:
                self.samples.append((os.path.join(mri_dir, fn), labels[pid], pid))
        self.train = train
        self.target_size = target_size
        self.target_depth = target_depth

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        vol = nib.load(path).get_fdata().astype(np.float32)  # [D,H,W]
        # robust normalize then resample
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol = (vol - lo) / (hi - lo + 1e-6)
        vol = np.clip(vol, 0, 1)
        # resample to cube target_size^3
        zf = (self.target_size/vol.shape[0], self.target_size/vol.shape[1], self.target_size/vol.shape[2])
        vol = zoom(vol, zf, order=1)
        # central slab of target_depth along z
        zc = vol.shape[0]//2
        hd = self.target_depth//2
        z0, z1 = zc-hd, zc+hd
        vol = vol[max(0,z0):min(vol.shape[0],z1), :, :]
        # pad/crop to exact depth
        if vol.shape[0] < self.target_depth:
            pad0 = (self.target_depth - vol.shape[0])//2
            pad1 = self.target_depth - vol.shape[0] - pad0
            vol = np.pad(vol, ((pad0,pad1),(0,0),(0,0)), mode='edge')
        elif vol.shape[0] > self.target_depth:
            start = (vol.shape[0]-self.target_depth)//2
            vol = vol[start:start+self.target_depth, :, :]

        if self.train:
            vol = mri_augment(vol)

        # r3d_18 expects [C=3, T=depth, H, W]; replicate channels
        vol = torch.from_numpy(vol)               # [T,H,W]
        vol = vol.unsqueeze(0).repeat(3,1,1,1)    # [3,T,H,W]
        return vol, torch.tensor(label, dtype=torch.long), pid

class FusionDataset(Dataset):
    def __init__(self, eeg_ds: EEGDataset, mri_ds: MRIDataset):
        self.eeg_map = {pid: i for i, (_, _, pid) in enumerate(eeg_ds.samples)}
        self.mri_map = {pid: i for i, (_, _, pid) in enumerate(mri_ds.samples)}
        self.ids = sorted(list(set(self.eeg_map).intersection(self.mri_map)))
        self.eeg_ds = eeg_ds
        self.mri_ds = mri_ds

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        eeg_x, y, _  = self.eeg_ds[self.eeg_map[pid]]
        mri_x, y2, _ = self.mri_ds[self.mri_map[pid]]
        assert y.item()==y2.item()
        return eeg_x, mri_x, y, pid

def _labels_for_sampler(ds):
    if isinstance(ds, FusionDataset):
        lbls = []
        for pid in ds.ids:
            _, l, _ = ds.eeg_ds.samples[ds.eeg_map[pid]]
            lbls.append(int(l))
        return lbls
    else:
        return [int(l) for (_, l, _) in ds.samples]

def make_loader(ds, batch_size):
    # class-balanced sampler
    labels = _labels_for_sampler(ds)
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sample_w = [weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

# ================= Models =================
class EEGEncoderConvNeXt(nn.Module):
    def __init__(self, out_dim=512, unfreeze_stages=(3,4)):
        super().__init__()
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # change input conv to 17 channels
        stem = base.features[0][0]  # Conv2d(3->96, k=4, s=4)
        new_stem = nn.Conv2d(17, stem.out_channels, kernel_size=stem.kernel_size,
                             stride=stem.stride, padding=stem.padding, bias=False)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.mean(1, keepdim=True).repeat(1,17,1,1))
        base.features[0][0] = new_stem
        self.backbone = base.features
        self.backbone_ln = nn.LayerNorm(768, eps=1e-6)  # convnext tiny final dim=768
        self.proj = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(768, out_dim)
        )
        self._freeze_all()
        self._unfreeze_stages(unfreeze_stages)

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def _unfreeze_stages(self, stages=(3,4)):
        for i, block in enumerate(self.backbone):
            if i in stages:
                for p in block.parameters(): p.requires_grad = True
        for p in self.backbone_ln.parameters(): p.requires_grad = True
        for p in self.proj.parameters(): p.requires_grad = True

    def forward(self, x):  # [B,17,224,224]
        x = self.backbone(x)          # [B,768,H/32,W/32]
        x = x.mean([-2,-1])           # GAP -> [B,768]
        x = self.backbone_ln(x)
        x = self.proj(x)              # [B,out_dim]
        x = F.normalize(x, dim=-1)
        return x

class MRIEncoder3D(nn.Module):
    def __init__(self, out_dim=512, unfreeze_layers=('layer4',)):
        super().__init__()
        base = vmodels.r3d_18(weights="DEFAULT")
        self.backbone = base
        in_feat = base.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_feat, in_feat), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(in_feat, out_dim)
        )
        self._freeze_all()
        self._unfreeze_layers(unfreeze_layers)

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def _unfreeze_layers(self, names=('layer4',)):
        for n, m in self.backbone.named_children():
            if n in names:
                for p in m.parameters(): p.requires_grad = True
        for p in self.proj.parameters(): p.requires_grad = True

    def forward(self, x):  # x: [B,3,T,H,W]
        x = self.backbone(x)          # [B, C]
        x = self.proj(x)              # [B, out_dim]
        x = F.normalize(x, dim=-1)
        return x

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

    def forward(self, e, m):
        x = torch.cat([e, m], dim=-1)          # [B, 1024]
        g = self.gate(x)                       # [B, 1] in [0,1]
        fused_vec = g*e + (1-g)*m              # [B, 512]
        fused = torch.cat([fused_vec, x], -1)  # [B, 1536]
        logits = self.cls(fused)               # [B, n_classes]
        return logits, g.squeeze(-1)

# ================= Losses & Eval =================
def cosine_align_loss(e, m, eps=1e-8):
    return (1 - F.cosine_similarity(e, m, dim=-1).clamp(min=-1+eps, max=1-eps)).mean()

@torch.no_grad()
def eval_epoch(eeg_enc, mri_enc, head, loader, split_name="test"):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    preds, gts = [], []
    for batch in loader:
        if len(batch) != 4:
            continue
        eeg_x, mri_x, y, _ = batch
        eeg_x = eeg_x.to(device, non_blocking=True)
        mri_x = mri_x.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type=='cuda')):
            e = eeg_enc(eeg_x)
            m = mri_enc(mri_x)
            logits, _ = head(e, m)
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu())
        gts.append(y.cpu())
    if not preds:
        print(f"[{split_name}] No samples to evaluate.")
        return 0.0, None, None
    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    acc = (preds==gts).mean()

    # Fix: force both labels so shapes are consistent even if one class absent
    cm = confusion_matrix(gts, preds, labels=[0,1])
    print(cm)
    print(classification_report(gts, preds, labels=[0,1],
                                target_names=["M","F"], digits=4, zero_division=0))
    return acc, preds, gts

# ================= Train =================
def train_fusion(eeg_ds_train, eeg_ds_test, mri_ds_train, mri_ds_test,
                 batch_size=4, epochs_stageA=15, epochs_stageB=10,
                 lr_head=2e-4, lr_backbone=1e-4, wd=1e-4, lambda_align=0.15):
    # build fusion set on overlap only
    fus_train = FusionDataset(eeg_ds_train, mri_ds_train)
    fus_test  = FusionDataset(eeg_ds_test,  mri_ds_test)

    tr_loader = make_loader(fus_train, batch_size)
    te_loader = DataLoader(fus_test, batch_size=batch_size, shuffle=False, num_workers=0)

    eeg_enc = EEGEncoderConvNeXt(out_dim=512, unfreeze_stages=(3,4)).to(device)
    mri_enc = MRIEncoder3D(out_dim=512, unfreeze_layers=('layer4',)).to(device)
    head    = GatedFusionHead(dim=512, n_classes=2).to(device)

    params_head = list(head.parameters())
    params_eeg  = [p for p in eeg_enc.parameters() if p.requires_grad]
    params_mri  = [p for p in mri_enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': params_head, 'lr': lr_head},
        {'params': params_eeg,  'lr': lr_backbone},
        {'params': params_mri,  'lr': lr_backbone},
    ], weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stageA+epochs_stageB)

    # Fix: new GradScaler API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    ce = nn.CrossEntropyLoss()

    def run_epochs(num_epochs, unfreeze=None):
        if unfreeze is not None:
            if 'eeg' in unfreeze: eeg_enc._unfreeze_stages(unfreeze['eeg'])
            if 'mri' in unfreeze: mri_enc._unfreeze_layers(unfreeze['mri'])
        best = 0.0
        for ep in range(1, num_epochs+1):
            eeg_enc.train(); mri_enc.train(); head.train()
            tot, correct, total = 0.0, 0, 0
            for eeg_x, mri_x, y, _ in tqdm(tr_loader, desc=f"Epoch {ep}"):
                eeg_x = eeg_x.to(device, non_blocking=True)
                mri_x = mri_x.to(device, non_blocking=True)
                y     = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type=='cuda')):
                    e = eeg_enc(eeg_x)
                    m = mri_enc(mri_x)
                    logits, g = head(e, m)
                    loss_cls = ce(logits, y)
                    loss_align = cosine_align_loss(e, m)
                    loss = loss_cls + lambda_align * loss_align
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()

                tot += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

            sched.step()
            train_loss = tot/total; train_acc = correct/total
            test_acc, _, _ = eval_epoch(eeg_enc, mri_enc, head, te_loader, split_name="test")
            best = max(best, test_acc if test_acc is not None else 0.0)
            print(f"[Stage] ep {ep} | train loss {train_loss:.4f} acc {train_acc:.4f} | test acc {test_acc:.4f} | best {best:.4f}")
        return best

    print("\n=== Stage A: heads + top blocks ===")
    bestA = run_epochs(epochs_stageA, unfreeze=None)

    print("\n=== Stage B: unfreeze one more block on each side (optional) ===")
    bestB = run_epochs(epochs_stageB, unfreeze={'eeg': (2,3,4), 'mri': ('layer3','layer4')})

    print(f"Done. Best test acc StageA={bestA:.4f}, StageB={bestB:.4f}")
    return eeg_enc, mri_enc, head

# ================= Split helpers =================
def count_ids(ids, gender_map):
    m = sum(1 for pid in ids if gender_map.get(pid, 1) == 0)
    f = sum(1 for pid in ids if gender_map.get(pid, 1) == 1)
    return m, f

def print_split_stats(train_ids, test_ids, gender_map):
    m_tr, f_tr = count_ids(train_ids, gender_map)
    m_te, f_te = count_ids(test_ids,  gender_map)
    print(f"[Subjects] Train: M={m_tr}  F={f_tr}  (Total={len(train_ids)})")
    print(f"[Subjects] Test : M={m_te}  F={f_te}  (Total={len(test_ids)})")
    if f_te > 0:
        print(f"[Subjects] Test ratio M:F = {m_te}:{f_te} = {m_te/f_te:.2f}:1")
    else:
        print("[Subjects] Test ratio M:F = INF:0 (no females)")

def build_test_split_m2f1(common_ids, gender_map, *,
                          test_frac=0.20,
                          male_label=0,
                          female_label=1,
                          seed=SEED):
    """
    Create a subject-wise TEST split with ratio M:F = 2:1 (exact if possible),
    using only subjects that exist in BOTH modalities.

    Strategy:
      - Let T = round(test_frac * N). We want T = 3k with M=2k, F=k.
      - Choose k = min(floor(T/3), floor(#M/2), #F).
      - If k==0 but at least 2M & 1F exist, use k=1 to force ratio.
      - Pick first 2k males and k females (shuffled), rest go to train.
    """
    rng = random.Random(seed)
    males   = [pid for pid in common_ids if gender_map.get(pid, female_label) == male_label]
    females = [pid for pid in common_ids if gender_map.get(pid, female_label) == female_label]
    rng.shuffle(males); rng.shuffle(females)

    N  = len(common_ids)
    T0 = max(1, int(round(test_frac * N)))
    k  = min(T0 // 3, len(males) // 2, len(females))

    # If rounded size prevents any test but we have at least 2M & 1F, force k=1
    if k == 0 and len(males) >= 2 and len(females) >= 1:
        k = 1

    test_m = males[:2*k]
    test_f = females[:k]
    test_ids = set(test_m + test_f)
    train_ids = set(pid for pid in common_ids if pid not in test_ids)

    print(f"[Split] Total subjects={N} | Desired test size≈{T0} | Achieved test size={len(test_ids)}")
    print(f"[Split] Available: M={len(males)} F={len(females)} | Picked for TEST: M={len(test_m)} F={len(test_f)}")
    return train_ids, test_ids

def subset_dataset(ds, keep_ids, is_train):
    keep = []
    for (p,l,pid) in ds.samples:
        if pid in keep_ids:
            keep.append((p,l,pid))
    new = type(ds)(EEG_DIR if isinstance(ds, EEGDataset) else MRI_DIR,
                   gender_map, train=is_train)
    new.samples = keep
    return new

# ================= Build full datasets =================
eeg_full = EEGDataset(EEG_DIR, gender_map, train=True)
mri_full = MRIDataset(MRI_DIR, gender_map, train=True)

# Subjects present in both modalities
eeg_ids_all = [extract_pid(p) for (p,_,_) in eeg_full.samples]
mri_ids_all = [extract_pid(p) for (p,_,_) in mri_full.samples]
common_ids = sorted(list(set(eeg_ids_all).intersection(mri_ids_all)))

# ====== Make TEST split with M:F = 2:1 (exact if possible) ======
TEST_FRAC = 0.20  # adjust as you like
train_ids, test_ids = build_test_split_m2f1(common_ids, gender_map, test_frac=TEST_FRAC, seed=SEED)
print_split_stats(train_ids, test_ids, gender_map)

# Build train/test datasets for each modality, with correct transform flags
eeg_tr = subset_dataset(eeg_full, train_ids, True)
eeg_te = subset_dataset(eeg_full, test_ids,  False)
mri_tr = subset_dataset(mri_full, train_ids, True)
mri_te = subset_dataset(mri_full, test_ids,  False)

# ================= Train & Evaluate =================
eeg_enc, mri_enc, head = train_fusion(
    eeg_tr, eeg_te, mri_tr, mri_te,
    batch_size=4, epochs_stageA=15, epochs_stageB=10,
    lr_head=2e-4, lr_backbone=1e-4, wd=1e-4, lambda_align=0.15
)

# ================= Save =================
torch.save({
    'eeg_enc': eeg_enc.state_dict(),
    'mri_enc': mri_enc.state_dict(),
    'head': head.state_dict(),
}, "fusion_convnext_r3d18_gated_align.pth")
print("✅ Saved: fusion_convnext_r3d18_gated_align.pth")
