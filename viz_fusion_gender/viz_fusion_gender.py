# viz_fusion_gender.py
# Visualization for EEG+MRI fusion gender model (TEST-only, no leakage).
# - t-SNE + LDA on fused embeddings
# - 3D Grad-CAM on MRI branch (subject & group)
# - EEG input-gradient saliency (subject & group)

import os, random, numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.models.video as vmodels
from scipy.ndimage import zoom

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm

# ---------- EDIT THESE ----------
EEG_DIR = r"C:\research\EEG_Domain\eye_openscalo"   # .npy (224,224,17)
MRI_DIR = r"C:/research/MRI/structural_MRI"          # .nii/.nii.gz
CSV     = r"C:/research/MRI/participants_LSD_andLEMON.csv"  # participant_id, gender (M/F)
CKPT    = r"checkpoints\fusion_GEA_MVA_CMCR.pth"
OUT_DIR = "fusion_gender_viz"
# --------------------------------

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

K_MRI_CROPS = 3
CLASS_NAMES = ["M","F"]  # 0: M, 1: F

# ----------- Labels / split utils (match training) -----------
def extract_pid(path_or_name: str) -> str:
    return os.path.basename(path_or_name).split("_")[0]

def build_gender_map(csv_path: str):
    df = pd.read_csv(csv_path)
    return {row['participant_id']: (0 if str(row['gender']).strip().upper()=='M' else 1)
            for _, row in df.iterrows()}

gender_map = build_gender_map(CSV)

def build_test_split_m2f1(common_ids, gmap, *, test_frac=0.20, seed=SEED):
    rng = random.Random(seed)
    males   = [pid for pid in common_ids if gmap.get(pid,1)==0]
    females = [pid for pid in common_ids if gmap.get(pid,1)==1]
    rng.shuffle(males); rng.shuffle(females)
    N  = len(common_ids); T0 = max(1, int(round(test_frac*N)))
    k  = min(T0//3, len(males)//2, len(females))
    if k==0 and len(males)>=2 and len(females)>=1: k=1
    test_ids = set(males[:2*k] + females[:k])
    train_ids = set(pid for pid in common_ids if pid not in test_ids)
    return train_ids, test_ids

def zscore(x, eps=1e-6):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

# ---------------- Datasets (test-only) ----------------
class EEGDataset(Dataset):
    def __init__(self, eeg_dir, labels):
        self.samples=[]
        for fn in os.listdir(eeg_dir):
            if fn.endswith(".npy"):
                pid = extract_pid(fn)
                if pid in labels:
                    self.samples.append((os.path.join(eeg_dir,fn), int(labels[pid]), pid))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        arr = np.load(path).astype(np.float32)      # (224,224,17)
        for c in range(arr.shape[-1]): arr[...,c] = zscore(arr[...,c])
        x = torch.from_numpy(arr).permute(2,0,1)    # [17,224,224]
        return x, torch.tensor(y).long(), pid

class MRIDataset(Dataset):
    """Returns K center-biased 128^3 crops AND 160^3 full volume for overlays."""
    def __init__(self, mri_dir, labels, k_crops=3):
        self.samples=[]
        for fn in os.listdir(mri_dir):
            if fn.endswith((".nii",".nii.gz")):
                pid = extract_pid(fn)
                if pid in labels:
                    self.samples.append((os.path.join(mri_dir,fn), int(labels[pid]), pid))
        self.k_crops=k_crops
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        vol = nib.load(path).get_fdata().astype(np.float32)
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
        target_big=160
        zf = (target_big/vol.shape[0], target_big/vol.shape[1], target_big/vol.shape[2])
        vol = zoom(vol, zf, order=1)  # [160,160,160]
        z0=y0=x0=(target_big-128)//2
        crops=[]
        for _ in range(self.k_crops):
            v=vol[z0:z0+128, y0:y0+128, x0:x0+128]
            v=torch.from_numpy(v).unsqueeze(0).repeat(3,1,1,1) # [3,128,128,128]
            crops.append(v)
        crops=torch.stack(crops,0)  # [K,3,128,128,128]
        return crops, torch.tensor(y).long(), pid, vol

class FusionDataset(Dataset):
    def __init__(self, eeg_ds: EEGDataset, mri_ds: MRIDataset):
        self.eeg_map={pid:i for i,(_,_,pid) in enumerate(eeg_ds.samples)}
        self.mri_map={pid:i for i,(_,_,pid) in enumerate(mri_ds.samples)}
        self.ids=sorted(list(set(self.eeg_map).intersection(self.mri_map)))
        self.eeg_ds=eeg_ds; self.mri_ds=mri_ds
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        pid=self.ids[idx]
        xe, y, _        = self.eeg_ds[self.eeg_map[pid]]
        xm, y2, _, vol  = self.mri_ds[self.mri_map[pid]]
        assert y.item()==y2.item()
        return xe, xm, y, pid, vol

def make_loader(ds, batch=1):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- Models (same as training; MRI returns att for viz) ----------------
class ElectrodeAttention(nn.Module):
    def __init__(self, C=17, r=4):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.mlp=nn.Sequential(
            nn.Conv2d(C, max(1,C//r), 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(max(1,C//r), C, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x): return x * self.mlp(self.pool(x))

class EEGEncoderConvNeXt(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.eaa=ElectrodeAttention(C=17, r=4)
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        stem = base.features[0][0]
        new_stem = nn.Conv2d(17, stem.out_channels, kernel_size=stem.kernel_size,
                             stride=stem.stride, padding=stem.padding, bias=False)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.mean(1, keepdim=True).repeat(1,17,1,1))
        base.features[0][0] = new_stem
        self.backbone = base.features
        self.backbone_ln = nn.LayerNorm(768, eps=1e-6)
        self.proj = nn.Sequential(nn.Linear(768,768), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(768,512))
    def forward(self, x):
        x=self.eaa(x); x=self.backbone(x); x=x.mean([-2,-1]); x=self.backbone_ln(x)
        x=self.proj(x); return F.normalize(x,dim=-1)

class MRIEncoder3D(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        base = vmodels.r3d_18(weights=vmodels.R3D_18_Weights.DEFAULT)
        self.backbone=base
        in_feat=base.fc.in_features
        self.backbone.fc=nn.Identity()
        self.proj=nn.Sequential(nn.Linear(in_feat,in_feat), nn.ReLU(inplace=True),
                                nn.Dropout(0.2), nn.Linear(in_feat,out_dim))
        self.att_q=nn.Linear(out_dim,out_dim,bias=False)
        self.att_k=nn.Linear(out_dim,out_dim,bias=False)
    def encode_one(self,x):
        h=self.backbone(x); z=self.proj(h); return F.normalize(z,dim=-1)
    def forward(self, xK):
        B,K=xK.shape[:2]
        x=xK.view(B*K,3,128,128,128)
        z=self.encode_one(x).view(B,K,-1)
        q=self.att_q(z); k=self.att_k(z)
        att=torch.softmax((q*k).sum(-1), dim=1).unsqueeze(-1)  # [B,K,1]
        z_pool=(att*z).sum(1)
        return F.normalize(z_pool,dim=-1), att.squeeze(-1)

class GatedFusionHead(nn.Module):
    def __init__(self, dim=512, n_classes=2):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(2*dim,dim), nn.ReLU(inplace=True),
                                nn.Linear(dim,1), nn.Sigmoid())
        self.cls =nn.Sequential(nn.Linear(3*dim,2*dim), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                nn.Linear(2*dim,n_classes))
    def forward(self, e,m,return_fused=False):
        x=torch.cat([e,m],-1); g=self.gate(x)
        fused_vec=g*e+(1-g)*m
        fused=torch.cat([fused_vec,x],-1)
        logits=self.cls(fused)
        if return_fused: return logits, g.squeeze(-1), fused_vec
        return logits, g.squeeze(-1)

# ---------------- Build TEST set (match training’s split logic) ----------------
def build_subject_lists(eeg_dir, mri_dir, gmap):
    eeg_full=EEGDataset(eeg_dir,gmap)
    mri_full=MRIDataset(mri_dir,gmap,k_crops=K_MRI_CROPS)
    eeg_ids=[extract_pid(p) for (p,_,p) in eeg_full.samples]
    mri_ids=[extract_pid(p) for (p,_,p) in mri_full.samples]
    common=sorted(list(set(eeg_ids).intersection(mri_ids)))
    return eeg_full, mri_full, common

def subset_ids(ds, ids):
    keep=[]
    for (p,l,pid) in ds.samples:
        if pid in ids: keep.append((p,l,pid))
    new = ds.__class__(EEG_DIR if isinstance(ds, EEGDataset) else MRI_DIR, gender_map,
                       **({} if isinstance(ds,EEGDataset) else {"k_crops":K_MRI_CROPS}))
    new.samples=keep; return new

# ---------------- t-SNE & LDA helpers ----------------
def plot_tsne_clean(feats, gts, out_png, title, seed=SEED):
    feats=np.asarray(feats); gts=np.asarray(gts).astype(int)
    if feats.ndim!=2 or feats.shape[0]<5:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5,0.5,"t-SNE skipped (not enough samples)",ha='center',va='center')
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(); return
    good=np.isfinite(feats).all(axis=1); X=feats[good]; y=gts[good]
    X=(X-X.mean(0))/(X.std(0)+1e-8)
    d=min(50,X.shape[1],X.shape[0]-1)
    Xp=PCA(n_components=d, random_state=seed).fit_transform(X)
    n=Xp.shape[0]; p=max(5,min(10,(n-1)//3))
    Z=TSNE(n_components=2, perplexity=p, init="pca",
           learning_rate="auto", metric="euclidean",
           random_state=seed, n_iter=1500).fit_transform(Xp)
    plt.figure(figsize=(6.5,5.5))
    for i,name in enumerate(CLASS_NAMES):
        idx=(y==i)
        if idx.any():
            plt.scatter(Z[idx,0],Z[idx,1],s=40,marker=("o" if i==0 else "^"),
                        edgecolors="k",linewidths=0.6,alpha=0.9,label=name)
    plt.legend(frameon=True); plt.title(f"{title} (perplexity={p}, n={n})")
    plt.xticks([]); plt.yticks([]); plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

def plot_lda_2d(feats, gts, out_png, title):
    feats=np.asarray(feats); gts=np.asarray(gts).astype(int)
    if feats.ndim!=2 or feats.shape[0]<3 or len(np.unique(gts))<2:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5,0.5,"LDA skipped (not enough classes/samples)",ha='center',va='center')
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(); return
    good=np.isfinite(feats).all(axis=1); X=feats[good]; y=gts[good]
    X=(X-X.mean(0))/(X.std(0)+1e-8)
    z1=LDA(n_components=1).fit_transform(X,y).ravel()
    Z=np.c_[z1, np.zeros_like(z1)]
    plt.figure(figsize=(6.5,2.7))
    for i,name in enumerate(CLASS_NAMES):
        idx=(y==i)
        if idx.any():
            plt.scatter(Z[idx,0],Z[idx,1],s=50,marker=("o" if i==0 else "^"),
                        edgecolors="k",linewidths=0.6,alpha=0.9,label=name)
    plt.axvline(0,linestyle="--",linewidth=1); plt.yticks([])
    plt.xlabel("LDA axis"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

# ---------------- Grad-CAM + EEG saliency ----------------
class CAMHook:
    def __init__(self, module):
        self.fmap=None; self.grad=None
        module.register_forward_hook(self._fwd)
        module.register_full_backward_hook(self._bwd)
    def _fwd(self, m, i, o): self.fmap=o.detach()
    def _bwd(self, m, gin, gout): self.grad=gout[0].detach()

def gradcam_3d_for_crop(mri_enc, head, crop_tensor, e_feat, class_idx=None):
    """
    Fresh graph per crop. e_feat is detached to avoid graph reuse.
    crop_tensor: [1,3,128,128,128]; e_feat: [1,512]
    """
    mri_enc.eval(); head.eval()
    e_fixed = e_feat.detach()
    target = mri_enc.backbone.layer4
    hook = CAMHook(target)
    with torch.enable_grad():
        z = mri_enc.encode_one(crop_tensor)                    # [1,512]
        logits, _, _ = head(e_fixed, z, return_fused=True)
        if class_idx is None: class_idx=int(logits.argmax(1).item())
        loss = logits[0, class_idx]
        mri_enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        loss.backward()
        fmap=hook.fmap[0]; grad=hook.grad[0]
        w=grad.mean(dim=(1,2,3), keepdim=True)
        cam=(w*fmap).sum(0).unsqueeze(0).unsqueeze(0)
        cam=torch.relu(cam)
        cam_up = F.interpolate(cam, size=(128,128,128), mode="trilinear", align_corners=False)[0,0]
        cam_up = (cam_up - cam_up.min())/(cam_up.max()-cam_up.min()+1e-8)
        return cam_up.detach().cpu().numpy()

def overlay_panels(volume_160, cam_128, out_png, title=""):
    big=np.zeros_like(volume_160); z0=y0=x0=(160-128)//2
    big[z0:z0+128, y0:y0+128, x0:x0+128]=cam_128
    zc=yc=xc=80
    fig,axs=plt.subplots(1,3,figsize=(11,3.8))
    axs[0].imshow(volume_160[:,yc,:],cmap="gray"); axs[0].imshow(big[:,yc,:],cmap="jet",alpha=0.45); axs[0].set_title("Sagittal"); axs[0].axis("off")
    axs[1].imshow(volume_160[zc,:,:],cmap="gray"); axs[1].imshow(big[zc,:,:],cmap="jet",alpha=0.45); axs[1].set_title("Coronal");  axs[1].axis("off")
    axs[2].imshow(volume_160[:,:,xc],cmap="gray"); axs[2].imshow(big[:,:,xc],cmap="jet",alpha=0.45); axs[2].set_title("Axial");    axs[2].axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

def eeg_input_saliency(eeg_enc, head, eeg_img, m_emb, class_idx=None):
    """Fresh graph; saliency wrt input EEG. eeg_img: [1,17,224,224]; m_emb: [1,512]"""
    eeg_enc.eval(); head.eval()
    with torch.enable_grad():
        eeg = eeg_img.clone().detach().requires_grad_(True)
        m_fixed = m_emb.detach()
        e_emb = eeg_enc(eeg)
        logits, _, _ = head(e_emb, m_fixed, return_fused=True)
        if class_idx is None: class_idx=int(logits.argmax(1).item())
        loss = logits[0, class_idx]
        eeg_enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        loss.backward()
        grad = eeg.grad.detach()                        # [1,17,H,W]
        sal  = grad.abs().mean(1)[0]                   # [H,W]
        sal  = (sal - sal.min())/(sal.max()-sal.min()+1e-8)
        return sal.cpu().numpy()

# ---------------- Collect TEST fused embeddings ----------------
@torch.no_grad()
def collect_test_embeddings(eeg_enc, mri_enc, head, loader):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    fused_list=[]; gts=[]; pids=[]
    for xe,xm,y,ids,_ in loader:
        xe=xe.to(device); xm=xm.to(device)
        e=eeg_enc(xe); m, _=mri_enc(xm)
        logits, _, fused_vec = head(e,m,return_fused=True)
        fused_list.append(F.normalize(fused_vec,dim=-1).cpu().numpy())
        gts.extend(list(y.numpy())); pids.extend(list(ids))
    fused = np.concatenate(fused_list,0) if fused_list else np.zeros((0,512))
    return fused, np.asarray(gts), pids

def cams_and_saliency_groups(eeg_enc, mri_enc, head, loader, k_crops=3):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    cam_sum={0:None,1:None}; cam_n={0:0,1:0}
    eeg_sum={0:None,1:None}; eeg_n={0:0,1:0}

    for xe,xm,y,ids,vol in tqdm(loader, desc="Grad-CAM & EEG saliency (test)"):
        xe=xe.to(device); xm=xm.to(device)
        vol=vol.numpy()[0]; yint=int(y.item())

        # targets and attention (no grad)
        with torch.no_grad():
            e_tmp = eeg_enc(xe)           # [1,512]
            m_tmp, att = mri_enc(xm)      # [1,512], [1,K]
            logits, _ = head(e_tmp, m_tmp)
            class_idx = int(logits.argmax(1).item())

        # MRI per-crop CAM (fresh graph per crop) -> attention-weighted subject CAM
        cams=[]
        for k in range(k_crops):
            crop = xm[:,k]                 # [1,3,128,128,128]
            cam_k = gradcam_3d_for_crop(mri_enc, head, crop, e_tmp, class_idx=class_idx)
            cams.append(cam_k)
        cams=np.stack(cams,0)
        att_w=att.squeeze(0).cpu().numpy()
        att_w=att_w/(att_w.sum()+1e-8)
        cam_subj=(att_w[:,None,None,None]*cams).sum(0)
        cam_subj=(cam_subj-cam_subj.min())/(cam_subj.max()-cam_subj.min()+1e-8)

        # accumulate by TRUE class
        if cam_sum[yint] is None: cam_sum[yint]=cam_subj.copy()
        else: cam_sum[yint]+=cam_subj
        cam_n[yint]+=1

        # EEG saliency
        sal = eeg_input_saliency(eeg_enc, head, xe, m_tmp, class_idx=class_idx)
        if eeg_sum[yint] is None: eeg_sum[yint]=sal.copy()
        else: eeg_sum[yint]+=sal
        eeg_n[yint]+=1

        # optional per-subject panels
        overlay_panels(vol, cam_subj, out_png=os.path.join(OUT_DIR, f"camMRI_{ids[0]}_panels.png"),
                       title=f"MRI Grad-CAM ({CLASS_NAMES[class_idx]})")
        plt.figure(figsize=(4,4)); plt.imshow(sal, cmap="inferno"); plt.axis("off")
        plt.title(f"EEG saliency ({CLASS_NAMES[class_idx]})")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"salEEG_{ids[0]}.png"), dpi=220); plt.close()

    cam_avg={c:(None if cam_n[c]==0 else (cam_sum[c]/cam_n[c])) for c in [0,1]}
    eeg_avg={c:(None if eeg_n[c]==0 else (eeg_sum[c]/eeg_n[c])) for c in [0,1]}
    return cam_avg, eeg_avg, cam_n, eeg_n

def save_group_cam_nifti(avg_maps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for c, cam in avg_maps.items():
        if cam is None: continue
        big=np.zeros((160,160,160),dtype=np.float32)
        z0=y0=x0=(160-128)//2
        big[z0:z0+128, y0:y0+128, x0:x0+128]=cam
        nib.save(nib.Nifti1Image(big, affine=np.eye(4)),
                 os.path.join(out_dir, f"group_CAM_MRI_{CLASS_NAMES[c]}.nii.gz"))
        overlay_panels(np.zeros_like(big), cam,
                       out_png=os.path.join(out_dir, f"group_CAM_MRI_{CLASS_NAMES[c]}_panels.png"),
                       title=f"Group MRI Grad-CAM ({CLASS_NAMES[c]})")

def save_group_eeg(avg_maps, out_dir):
    for c, sal in avg_maps.items():
        if sal is None: continue
        plt.figure(figsize=(4.2,4)); plt.imshow(sal, cmap="inferno"); plt.axis("off")
        plt.title(f"Group EEG saliency ({CLASS_NAMES[c]})")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"group_SALIENCY_EEG_{CLASS_NAMES[c]}.png"), dpi=240); plt.close()

# --------------------------- Main ---------------------------
if __name__=="__main__":
    # Build TEST set to avoid leakage (same split logic as training)
    eeg_full, mri_full, common_ids = build_subject_lists(EEG_DIR, MRI_DIR, gender_map)
    _, test_ids = build_test_split_m2f1(common_ids, gender_map, test_frac=0.20, seed=SEED)
    eeg_te = subset_ids(eeg_full, test_ids)
    mri_te = subset_ids(mri_full, test_ids)
    test_ds = FusionDataset(eeg_te, mri_te)
    te_loader = make_loader(test_ds, batch=1)

    print(f"[TEST] subjects: {len(test_ds)} | M={sum(gender_map[i]==0 for i in test_ds.ids)} "
          f"| F={sum(gender_map[i]==1 for i in test_ds.ids)}")

    # Load models
    eeg_enc = EEGEncoderConvNeXt(out_dim=512).to(device)
    mri_enc = MRIEncoder3D(out_dim=512).to(device)
    head    = GatedFusionHead(dim=512, n_classes=2).to(device)

    ckpt=torch.load(CKPT, map_location=device)
    eeg_enc.load_state_dict(ckpt['eeg_enc'], strict=True)
    mri_enc.load_state_dict(ckpt['mri_enc'], strict=True)
    head.load_state_dict(ckpt['head'], strict=True)
    eeg_enc.eval(); mri_enc.eval(); head.eval()

    # 1) Collect fused TEST embeddings -> t-SNE & LDA
    print("Collecting fused TEST embeddings...")
    fused, gts, pids = collect_test_embeddings(eeg_enc, mri_enc, head, te_loader)

    preds=[]
    with torch.no_grad():
        for xe,xm,y,ids,_ in te_loader:
            xe=xe.to(device); xm=xm.to(device)
            e=eeg_enc(xe); m,_=mri_enc(xm)
            logits,_=head(e,m); preds.append(int(logits.argmax(1)))
    preds=np.array(preds)
    print(confusion_matrix(gts, preds, labels=[0,1]))
    print(classification_report(gts, preds, labels=[0,1], target_names=CLASS_NAMES, digits=4, zero_division=0))

    plot_tsne_clean(fused, gts, out_png=os.path.join(OUT_DIR,"tsne_fused_test_clean.png"),
                    title="t-SNE of Fused Test Embeddings")
    plot_lda_2d(fused, gts, out_png=os.path.join(OUT_DIR,"lda_fused_test.png"),
                title="LDA (2-class) Projection of Fused Embeddings")

    # 2) MRI Grad-CAM + EEG saliency (TEST ONLY; no leakage)
    print("Computing MRI Grad-CAM and EEG saliency (group averages)...")
    cam_avg, eeg_avg, n_cam, n_eeg = cams_and_saliency_groups(eeg_enc, mri_enc, head, te_loader, k_crops=K_MRI_CROPS)
    print(f"Subjects per class (test): CAM M={n_cam[0]}, F={n_cam[1]} | EEG M={n_eeg[0]}, F={n_eeg[1]}")

    # 3) Save groups
    save_group_cam_nifti(cam_avg, OUT_DIR)
    save_group_eeg(eeg_avg, OUT_DIR)

    print(f"✅ Saved: t-SNE, LDA, per-subject MRI CAM panels, EEG saliency, and group NIfTIs under ./{OUT_DIR}/")
