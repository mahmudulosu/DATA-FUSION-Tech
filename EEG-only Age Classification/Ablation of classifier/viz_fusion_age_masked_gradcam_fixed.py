# viz_fusion_age_masked_gradcam_fixed.py
# EEG+MRI age (young/old) visualization with *brain-masked* 3D Grad-CAM (robust hooks).
# - Same TEST split logic as training (subject-level, no leakage)
# - CAM normalized *inside brain* to avoid background artifacts
# - Per-subject panels show: pred label (+prob) AND true label
# - Classwise group-average CAMs saved as NIfTI + PNG
# - Also saves fused TEST embeddings t-SNE/LDA (unchanged)

import os, random, numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.models.video as vmodels
from scipy.ndimage import zoom, binary_opening, binary_closing, binary_fill_holes

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm

# ----------------- EDIT THESE -----------------
EEG_DIR = r"C:\research\EEG_Domain\eye_openscalo"  # .npy, shape (224,224,17)
MRI_DIR = r"C:\research\MRI\structural_MRI"         # .nii/.nii.gz
CSV     = r"C:\research\MRI\participants_LSD_andLEMON.csv"
CKPT    = r"checkpoints\fusion_GEA_MVA_CMCR_AGE.pth"
OUT_DIR = "fusion_age_viz_masked_gradcam"
# ----------------------------------------------

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

K_MRI_CROPS = 3
CLASS_NAMES = ["young", "old"]  # 0,1

# ---------------- Labels ----------------
def build_age_label_map(csv_path: str):
    df = pd.read_csv(csv_path)
    valid_ages = ["20-25","25-30","60-65","65-70","70-75"]
    age_group = {"20-25":"young","25-30":"young","60-65":"old","65-70":"old","70-75":"old"}
    df = df[df["age"].isin(valid_ages)].copy()
    lab = {"young":0,"old":1}
    return {str(row["participant_id"]).strip(): lab[age_group[row["age"]]] for _,row in df.iterrows()}

age_map = build_age_label_map(CSV)

def extract_pid(path_or_name: str) -> str:
    return os.path.basename(path_or_name).split("_")[0]

def zscore(x, eps=1e-6):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

# ---------------- Datasets ----------------
class EEGDataset(Dataset):
    def __init__(self, eeg_dir, labels, train=False):
        self.samples=[]
        for fn in os.listdir(eeg_dir):
            if fn.endswith(".npy"):
                pid = extract_pid(fn)
                if pid in labels:
                    self.samples.append((os.path.join(eeg_dir,fn), int(labels[pid]), pid))
        self.train=train
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, y, pid = self.samples[idx]
        arr = np.load(path).astype(np.float32)      # (H,W,17)
        for c in range(arr.shape[-1]): arr[...,c] = zscore(arr[...,c])
        x = torch.from_numpy(arr).permute(2,0,1)    # [17,224,224]
        return x, torch.tensor(y).long(), pid

class MRIDataset(Dataset):
    """Returns K center 128^3 crops AND the 160^3 full volume for overlays."""
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

        # center crop (K identical center crops to match training signature)
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
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

# ---------------- Models (same as training) ----------------
class ElectrodeAttention(nn.Module):
    def __init__(self, C=17, r=4):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.mlp=nn.Sequential(
            nn.Conv2d(C, max(1,C//r), 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(max(1,C//r), C, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.mlp(self.pool(x))

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
                                  nn.Linear(768,out_dim))
    def forward(self, x):
        x = self.eaa(x)
        x = self.backbone(x)      # [B,768,h,w]
        x = x.mean([-2,-1])       # [B,768]
        x = self.backbone_ln(x)
        x = self.proj(x)          # [B,512]
        return F.normalize(x, dim=-1)

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
        h=self.backbone(x)          # [B,C]
        z=self.proj(h)              # [B,512]
        return F.normalize(z,dim=-1)
    def forward(self, xK):
        B,K=xK.shape[:2]
        x=xK.view(B*K,3,128,128,128)
        z=self.encode_one(x).view(B, K, -1)
        q=self.att_q(z); k=self.att_k(z)
        att=torch.softmax((q*k).sum(-1),dim=1).unsqueeze(-1)  # [B,K,1]
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
        x=torch.cat([e,m],-1)
        g=self.gate(x)
        fused_vec=g*e+(1-g)*m
        fused=torch.cat([fused_vec,x],-1)
        logits=self.cls(fused)
        if return_fused: return logits, g.squeeze(-1), fused_vec
        return logits, g.squeeze(-1)

# ---------------- Split helpers (subject-level) ----------------
def build_subject_lists(eeg_dir, mri_dir, id2label):
    eeg_full = EEGDataset(eeg_dir, id2label, train=False)
    mri_full = MRIDataset(mri_dir, id2label, k_crops=K_MRI_CROPS)
    eeg_ids=[extract_pid(p) for (p,_,p) in eeg_full.samples]
    mri_ids=[extract_pid(p) for (p,_,p) in mri_full.samples]
    common=sorted(list(set(eeg_ids).intersection(mri_ids)))
    return eeg_full, mri_full, common

def stratified_subject_split(common_ids, id2label, test_frac=0.20, seed=SEED):
    rng=random.Random(seed)
    by_cls={0:[],1:[]}
    for pid in common_ids:
        if pid in id2label: by_cls[id2label[pid]].append(pid)
    for k in by_cls: rng.shuffle(by_cls[k])
    test=set()
    for k,lst in by_cls.items():
        n=len(lst); t=max(1,int(round(test_frac*n))) if n>0 else 0
        test.update(lst[:t])
    train=set(pid for pid in common_ids if pid not in test)
    return train,test

def subset(ds, keep_ids):
    keep=[]
    for (p,l,pid) in ds.samples:
        if pid in keep_ids: keep.append((p,l,pid))
    new = ds.__class__(EEG_DIR if isinstance(ds, EEGDataset) else MRI_DIR, age_map,
                       **({} if isinstance(ds,EEGDataset) else {"k_crops":K_MRI_CROPS}))
    new.samples=keep
    return new

# ---------------- Viz helpers: t-SNE & LDA ----------------
def plot_tsne_clean(feats, gts, out_png, title, seed=SEED):
    feats=np.asarray(feats); gts=np.asarray(gts).astype(int)
    if feats.ndim!=2 or feats.shape[0]<5:
        plt.figure(figsize=(6,4)); plt.axis('off')
        plt.text(0.5,0.5,"t-SNE skipped (not enough samples)",ha='center',va='center')
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close(); return
    good=np.isfinite(feats).all(axis=1)
    X=feats[good]; y=gts[good]
    X=(X-X.mean(0))/(X.std(0)+1e-8)
    d=min(50,X.shape[1],X.shape[0]-1)
    Xp=PCA(n_components=d, random_state=seed).fit_transform(X)
    n=Xp.shape[0]; p=max(5,min(10,(n-1)//3))
    Z=TSNE(n_components=2, perplexity=p, init="pca",
           learning_rate="auto", random_state=seed,
           n_iter=1500).fit_transform(Xp)
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

# ---------------- Brain mask + overlay utils ----------------
def make_brain_mask_160(vol160, thr=0.10):
    m = (vol160 > thr).astype(np.uint8)
    m = binary_opening(m, structure=np.ones((3,3,3))).astype(np.uint8)
    m = binary_closing(m, structure=np.ones((3,3,3))).astype(np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    return m  # [160,160,160]

def put_128_into_160(x128):
    big = np.zeros((160,160,160), dtype=np.float32)
    z0 = y0 = x0 = (160-128)//2
    big[z0:z0+128, y0:y0+128, x0:x0+128] = x128
    return big

def normalize_cam_inside_mask(cam128, brain_mask_160):
    cam160 = put_128_into_160(cam128)
    m = brain_mask_160.astype(bool)
    if m.sum() == 0:
        return cam160
    v = cam160[m]
    vmin, vmax = np.percentile(v, [1.0, 99.5])
    v = np.clip((v - vmin) / (vmax - vmin + 1e-8), 0, 1)
    out = np.zeros_like(cam160, dtype=np.float32)
    out[m] = v
    return out

def overlay_panels_masked(volume_160, cam_128, out_png, title=""):
    brain_mask = make_brain_mask_160(volume_160)
    cam160 = normalize_cam_inside_mask(cam_128, brain_mask)
    zc = yc = xc = 80
    fig, axs = plt.subplots(1,3,figsize=(11,3.8))
    axs[0].imshow(volume_160[:,yc,:],cmap="gray"); axs[0].imshow(cam160[:,yc,:],cmap="jet",alpha=0.45,vmin=0,vmax=1); axs[0].set_title("Sagittal"); axs[0].axis("off")
    axs[1].imshow(volume_160[zc,:,:],cmap="gray"); axs[1].imshow(cam160[zc,:,:],cmap="jet",alpha=0.45,vmin=0,vmax=1); axs[1].set_title("Coronal");  axs[1].axis("off")
    axs[2].imshow(volume_160[:,:,xc],cmap="gray"); axs[2].imshow(cam160[:,:,xc],cmap="jet",alpha=0.45,vmin=0,vmax=1); axs[2].set_title("Axial");    axs[2].axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

# ---------------- Robust 3D Grad-CAM (forward hook + retain_grad) ----------------
def get_cam_target_module(mri_enc, layer_name="layer3"):
    target = getattr(mri_enc.backbone, layer_name, None)
    if target is None:
        raise ValueError(f"Backbone has no layer '{layer_name}'. "
                         f"Available: {list(dict(mri_enc.backbone.named_children()).keys())}")
    return target

def gradcam_3d_for_crop(mri_enc, head, crop_tensor, e_feat, class_idx=None, cam_layer="layer3"):
    """
    crop_tensor: [1,3,128,128,128]; e_feat: [1,512] (fixed EEG embedding)
    Returns CAM volume [128,128,128] scaled to [0,1].
    """
    mri_enc.eval(); head.eval()
    e_fixed = e_feat.detach()

    target = get_cam_target_module(mri_enc, cam_layer)
    activations = {}
    def fwd_hook(_m, _inp, out):
        activations["act"] = out
        out.retain_grad()
    handle = target.register_forward_hook(fwd_hook)

    try:
        with torch.enable_grad():
            z = mri_enc.encode_one(crop_tensor)                  # [1,512]
            logits, _, _ = head(e_fixed, z, return_fused=True)   # fusion head
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            loss = logits[0, class_idx]
            mri_enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            loss.backward()

            act   = activations["act"]        # [1,C,D',H',W']
            grads = act.grad                  # [1,C,D',H',W']
            if grads is None:
                raise RuntimeError("Activation.grad is None; ensure no global no_grad/inference_mode.")

            weights = grads.mean(dim=(2,3,4), keepdim=True)      # [1,C,1,1,1]
            cam = (weights * act).sum(dim=1, keepdim=True)       # [1,1,D',H',W']
            cam = torch.relu(cam)
            cam_up = F.interpolate(cam, size=(128,128,128), mode="trilinear", align_corners=False)[0,0]
            cam_up = (cam_up - cam_up.min())/(cam_up.max()-cam_up.min()+1e-8)
            return cam_up.detach().cpu().numpy()
    finally:
        handle.remove()

# ---------------- Collect TEST feats & saliency/CAM ----------------
@torch.no_grad()
def collect_test_embeddings(eeg_enc, mri_enc, head, loader):
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    fused_list=[]; gts=[]; pids=[]
    for xe,xm,y,ids,_ in loader:
        xe=xe.to(device); xm=xm.to(device)
        e = eeg_enc(xe)
        m, _ = mri_enc(xm)
        logits, _, fused_vec = head(e,m,return_fused=True)
        fused_list.append(F.normalize(fused_vec,dim=-1).cpu().numpy())
        gts.extend(list(y.numpy())); pids.extend(list(ids))
    fused=np.concatenate(fused_list,0) if fused_list else np.zeros((0,512))
    return fused, np.asarray(gts), pids

def cams_group(eeg_enc, mri_enc, head, loader, k_crops=3, cam_layer="layer3", use_pred_class=True):
    """
    Computes per-subject CAMs and classwise averages.
    If use_pred_class=True, CAMs highlight the predicted class; else use the true class.
    Also saves test_predictions.csv with probs and labels.
    """
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    cam_sum={0:None,1:None}; cam_n={0:0,1:0}
    rows=[]

    for xe,xm,y,ids,vol in tqdm(loader, desc="MRI Grad-CAM (masked)"):
        xe=xe.to(device); xm=xm.to(device)
        vol=vol.numpy()[0]; yint=int(y.item())

        # get e-feature, attention + prediction once (no grad)
        with torch.no_grad():
            e_tmp = eeg_enc(xe)
            m_tmp, att = mri_enc(xm)
            logits, _ = head(e_tmp, m_tmp)
            probs = F.softmax(logits,1)
            pred_idx = int(probs.argmax(1).item())
            pred_p   = float(probs[0, pred_idx].item())

        class_idx = pred_idx if use_pred_class else yint

        # per-crop CAMs with gradients enabled inside helper
        cams=[]
        for k in range(k_crops):
            crop = xm[:,k]                   # [1,3,128,128,128]
            cam_k = gradcam_3d_for_crop(mri_enc, head, crop, e_tmp, class_idx=class_idx, cam_layer=cam_layer)
            cams.append(cam_k)
        cams=np.stack(cams,0)                # [K,128,128,128]

        # attention-weighted fusion across crops
        att_w=att.squeeze(0).cpu().numpy()
        att_w=att_w/(att_w.sum()+1e-8)
        cam_subj=(att_w[:,None,None,None]*cams).sum(0)
        cam_subj=(cam_subj-cam_subj.min())/(cam_subj.max()-cam_subj.min()+1e-8)

        # accumulate by TRUE class
        if cam_sum[yint] is None: cam_sum[yint]=cam_subj.copy()
        else: cam_sum[yint]+=cam_subj
        cam_n[yint]+=1

        # save per-subject overlay with pred + true in title
        title=f"{ids[0]} — pred:{CLASS_NAMES[pred_idx]} ({pred_p:.2f}) | true:{CLASS_NAMES[yint]}"
        overlay_panels_masked(vol, cam_subj,
                              out_png=os.path.join(OUT_DIR, f"camMRI_{ids[0]}_panels.png"),
                              title=title)

        rows.append({
            "pid": ids[0],
            "y_true": yint,
            "y_pred": pred_idx,
            f"prob_{CLASS_NAMES[0]}": float(probs[0,0].item()),
            f"prob_{CLASS_NAMES[1]}": float(probs[0,1].item()),
        })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"test_predictions_segments.csv"), index=False)

    cam_avg={c:(None if cam_n[c]==0 else (cam_sum[c]/cam_n[c])) for c in [0,1]}
    return cam_avg, cam_n

def save_group_cam_nifti(avg_maps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for c, cam in avg_maps.items():
        if cam is None: continue
        big=np.zeros((160,160,160),dtype=np.float32)
        z0=y0=x0=(160-128)//2
        big[z0:z0+128, y0:y0+128, x0:x0+128]=cam
        nib.save(nib.Nifti1Image(big, affine=np.eye(4)),
                 os.path.join(out_dir, f"group_CAM_MRI_{CLASS_NAMES[c]}.nii.gz"))
        overlay_panels_masked(np.ones_like(big), cam,
                              out_png=os.path.join(out_dir, f"group_CAM_MRI_{CLASS_NAMES[c]}_panels.png"),
                              title=f"Group MRI Grad-CAM ({CLASS_NAMES[c]})")

# --------------------------- Main ---------------------------
if __name__=="__main__":
    # Build TEST split (subject-level; identical logic to training; we only use TEST here)
    eeg_full, mri_full, common_ids = build_subject_lists(EEG_DIR, MRI_DIR, age_map)
    train_ids, test_ids = stratified_subject_split(common_ids, age_map, test_frac=0.20, seed=SEED)

    def subset_ids(ds, ids): 
        keep=[]
        for (p,l,pid) in ds.samples:
            if pid in ids: keep.append((p,l,pid))
        new = ds.__class__(EEG_DIR if isinstance(ds, EEGDataset) else MRI_DIR, age_map,
                           **({} if isinstance(ds,EEGDataset) else {"k_crops":K_MRI_CROPS}))
        new.samples=keep; return new

    eeg_te = subset_ids(eeg_full, test_ids)
    mri_te = subset_ids(mri_full, test_ids)
    test_ds = FusionDataset(eeg_te, mri_te)
    te_loader = make_loader(test_ds, batch=1)

    print(f"[TEST] subjects: {len(test_ds)} | young={sum(age_map[i]==0 for i in test_ds.ids)} "
          f"| old={sum(age_map[i]==1 for i in test_ds.ids)}")

    # Load models
    eeg_enc = EEGEncoderConvNeXt(out_dim=512).to(device)
    mri_enc = MRIEncoder3D(out_dim=512).to(device)
    head    = GatedFusionHead(dim=512, n_classes=2).to(device)

    ckpt=torch.load(CKPT, map_location=device)
    eeg_enc.load_state_dict(ckpt['eeg_enc'], strict=True)
    mri_enc.load_state_dict(ckpt['mri_enc'], strict=True)
    head.load_state_dict(ckpt['head'], strict=True)
    eeg_enc.eval(); mri_enc.eval(); head.eval()
    print("✓ Fusion weights loaded.")

    # 1) Fused embeddings -> t-SNE & LDA (TEST ONLY)
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

    # 2) MRI Grad-CAM (brain-masked)
    print("Computing brain-masked MRI Grad-CAM (group averages)…")
    cam_avg, n_cam = cams_group(eeg_enc, mri_enc, head, te_loader,
                                k_crops=K_MRI_CROPS, cam_layer="layer3", use_pred_class=True)
    print(f"Subjects per class (true labels): young={n_cam[0]}, old={n_cam[1]}")

    # 3) Save group maps
    save_group_cam_nifti(cam_avg, OUT_DIR)

    print(f"✅ Saved t-SNE, LDA, per-subject MRI CAM panels (masked), and group NIfTIs under ./{OUT_DIR}/")
