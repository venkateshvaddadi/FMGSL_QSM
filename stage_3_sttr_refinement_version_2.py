#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:30:00 2026

@author: venkatesh
"""

import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from math import exp
from scipy.signal import convolve as convn
from monai.networks.nets import SwinUNETR

# Import model architecture directly from your verified repository code
from QSMnet import QSMnet

print('=' * 85)
print('LAUNCHING STAGE-III: SEMANTIC TEST-TIME REFINEMENT (STTR) ENGINE')
print('=' * 85)

# =====================================================================
# 1. SPECIALIZED QSM EVALUATION METRICS REFERENCE
# =====================================================================
def compute_psnr(chi_recon, chi_true):
    img1 = np.asarray(chi_recon).copy()
    img2 = np.asarray(chi_true).copy()
    min_img = min(img1.min(), img2.min())
    img1[img1 != 0] -= min_img
    img2[img2 != 0] -= min_img
    max_img = max(img1.max(), img2.max())
    if max_img != 0:
        img1 = 255 * img1 / max_img
        img2 = 255 * img2 / max_img
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_rmse(chi_recon, chi_true):
    chi_recon = np.asarray(chi_recon)
    chi_true = np.asarray(chi_true)
    numerator = np.linalg.norm(chi_recon.ravel() - chi_true.ravel())
    denominator = np.linalg.norm(chi_true.ravel())
    if denominator == 0: return float('inf')
    return 100 * numerator / denominator

def compute_hfen(img1, img2):
    img1 = np.squeeze(img1).astype(np.float64)
    img2 = np.squeeze(img2).astype(np.float64)
    if img1.ndim != 3: raise ValueError(f"Expected 3D volumes, got shape {img1.shape}")
    filt_siz = np.array([15, 15, 15])
    sig = np.array([1.5, 1.5, 1.5])
    siz = (filt_siz - 1) / 2
    x_range = np.arange(-siz[0], siz[0] + 1)
    y_range = np.arange(-siz[1], siz[1] + 1)
    z_range = np.arange(-siz[2], siz[2] + 1)
    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    h = np.exp(-(x**2 / (2 * sig[0]**2) + y**2 / (2 * sig[1]**2) + z**2 / (2 * sig[2]**2)))
    h = h / np.sum(h)
    arg = (x**2 / sig[0]**4 + y**2 / sig[1]**4 + z**2 / sig[2]**4 - (1/sig[0]**2 + 1/sig[1]**2 + 1/sig[2]**2))
    H = arg * h
    H = H - (np.sum(H) / np.prod(filt_siz))
    img1_log = convn(img1, H, mode='same')
    img2_log = convn(img2, H, mode='same')
    return compute_rmse(img1_log, img2_log)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    return Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())

def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.conv3d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    if img1.is_cuda: window = window.cuda(img1.get_device())
    return _ssim_3D(img1, img2, window.type_as(img1), window_size, channel, size_average)

def compute_ssim_numpy(img1_np, img2_np, window_size=11, size_average=True):
    img1_t = torch.from_numpy(img1_np).float()
    img2_t = torch.from_numpy(img2_np).float()
    if img1_t.ndimension() == 3: img1_t = img1_t.unsqueeze(0).unsqueeze(0)
    elif img1_t.ndimension() == 4: img1_t = img1_t.unsqueeze(1)
    if torch.cuda.is_available(): img1_t, img2_t = img1_t.cuda(), img2_t.cuda()
    with torch.no_grad(): return ssim3D(img1_t, img2_t, window_size=window_size, size_average=size_average).item()

# =====================================================================
# 2. GLOBAL FULL-RESOLUTION INVERSION OBJECTIVES
# =====================================================================
class DataConsistencyLoss(nn.Module):
    def __init__(self, matrix_size=(176, 176, 160)):
        super().__init__()
        kx = torch.fft.fftfreq(matrix_size[0])
        ky = torch.fft.fftfreq(matrix_size[1])
        kz = torch.fft.fftfreq(matrix_size[2])
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX**2 + KY**2 + KZ**2
        k2 = torch.where(k2 == 0, torch.ones_like(k2) * 1e-12, k2)
        kernel = (1.0 / 3.0) - (KZ**2 / k2)
        kernel[0, 0, 0] = 0.0
        self.register_buffer("D", kernel)
        self.mse = nn.MSELoss()

    def forward(self, pred_qsm, input_local_field, mask):
        pred_fft = torch.fft.fftn(pred_qsm.squeeze(1), dim=(-3, -2, -1))
        sim_field = torch.fft.ifftn(self.D * pred_fft, dim=(-3, -2, -1)).real.unsqueeze(1)
        return self.mse(sim_field * mask, input_local_field * mask)

class FrozenSwinUNETRTeacher(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.swin_unetr = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_checkpoint=False)
        self.gap = nn.AdaptiveAvgPool3d(1)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        s_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else (checkpoint["model"] if "model" in checkpoint else checkpoint)
        encoder_dict = {k[11:] if k.startswith("swin_unetr.") else (k[7:] if k.startswith("module.") else k): v for k, v in s_dict.items()}
        self.swin_unetr.load_state_dict(encoder_dict, strict=False)
        for param in self.parameters(): param.requires_grad = False

    def forward(self, x):
        hidden_states_out = self.swin_unetr.swinViT(x)
        pooled = self.gap(hidden_states_out[4])
        return pooled.view(pooled.size(0), -1)

# =====================================================================
# 3. PATHS & PARAMETER CONFIGURATIONS
# =====================================================================
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

STTR_ITERS = 40          
STTR_LR    = 1e-3        

lambda_dc    = 1.0         
lambda_prior = 0.05      
lambda_sem   = 0.01      

raw_data_path = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
data_path     = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/'
stats_path    = os.path.join(data_path, 'csv_files/tr-stats.mat')
teacher_path  = "pretrained/ssl_pretrained_weights.pth"
database_path = "pretrained/7T_embedding_database.pt"
ref_7t_path   = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches/"

stage2_model_checkpoint = "savedModels_stage2/qsmnet_stage2_epoch_7.pt"
patients_list = [7, 8, 9, 10, 11, 12]
results = []

outdir = "savedModels_stage2/STTR_Inference_Outputs/"
os.makedirs(outdir, exist_ok=True)
os.makedirs("pretrained", exist_ok=True)

# Initialize teacher network
teacher_net = FrozenSwinUNETRTeacher(checkpoint_path=teacher_path).to(device).eval()

# -----------------------------------------------------------------
# 🔍 AUTOMATIC MANIFOLD DATABASE COMPILER
# -----------------------------------------------------------------
# if not os.path.exists(database_path):
#     print("[*] 7T Embedding Database missing. Compiling 7T Manifold directly...")
#     manifold_database = []
#     # Fetch all patches from your LPCNN directory
#     patch_files = [f for f in os.listdir(ref_7t_path) if f.endswith('.mat')]
    
#     if len(patch_files) == 0:
#         print(f"❌ Error: No .mat files found in 7T data path: {ref_7t_path}")
#         exit()
        
#     for idx, file in enumerate(patch_files):
#         mat = sio.loadmat(os.path.join(ref_7t_path, file))
#         # Safely locate the correct susceptibility dictionary variable name
#         k = 'qsm_7t' if 'qsm_7t' in mat else [key for key in mat.keys() if not key.startswith('__')][0]
#         tensor = torch.from_numpy(mat[k].astype(float)).unsqueeze(0).unsqueeze(0).float().to(device)
        
#         with torch.no_grad():
#             z = teacher_net(tensor)
#             manifold_database.append(z.cpu())
            
#     manifold_tensor = torch.cat(manifold_database, dim=0)
#     torch.save(manifold_tensor, database_path)
#     print(f"[✓] Compiled {manifold_tensor.shape[0]} patches into {database_path}\n")







# # -----------------------------------------------------------------
# # 🔍 AUTOMATIC MANIFOLD DATABASE COMPILER (Updated with Exact Key 'susc')
# # -----------------------------------------------------------------
# if not os.path.exists(database_path):
#     print("[*] 7T Embedding Database missing. Compiling 7T Manifold directly...")
#     manifold_database = []
#     # Fetch all patches from your LPCNN directory
#     patch_files = [f for f in os.listdir(ref_7t_path) if f.endswith('.mat')]
    
#     if len(patch_files) == 0:
#         print(f"❌ Error: No .mat files found in 7T data path: {ref_7t_path}")
#         exit()
        
#     for idx, file in enumerate(patch_files):
#         mat = sio.loadmat(os.path.join(ref_7t_path, file))
        
#         # 🟢 FIX: Directly target 'susc' if present, otherwise fallback safely
#         if 'susc' in mat:
#             k = 'susc'
#         elif 'qsm_7t' in mat:
#             k = 'qsm_7t'
#         else:
#             k = [key for key in mat.keys() if not key.startswith('__')][0]
            
#         tensor = torch.from_numpy(mat[k].astype(float)).unsqueeze(0).unsqueeze(0).float().to(device)
        
#         with torch.no_grad():
#             z = teacher_net(tensor)
#             manifold_database.append(z.cpu())
            
#     manifold_tensor = torch.cat(manifold_database, dim=0)
#     torch.save(manifold_tensor, database_path)
#     print(f"[✓] Compiled {manifold_tensor.shape[0]} patches into {database_path}\n")




import tqdm

# -----------------------------------------------------------------
# 🔍 AUTOMATIC MANIFOLD DATABASE COMPILER (Updated with Exact Key 'susc' & tqdm)
# -----------------------------------------------------------------
if not os.path.exists(database_path):
    print("[*] 7T Embedding Database missing. Compiling 7T Manifold directly...")
    manifold_database = []
    # Fetch all patches from your LPCNN directory
    patch_files = [f for f in os.listdir(ref_7t_path) if f.endswith('.mat')]
    
    if len(patch_files) == 0:
        print(f"❌ Error: No .mat files found in 7T data path: {ref_7t_path}")
        exit()
        
    # 🟢 Wrapped patch files with a descriptive tqdm progress bar
    pbar_comp = tqdm.tqdm(patch_files, desc="Compiling 7T Manifold", leave=True)
        
    for file in pbar_comp:
        mat = sio.loadmat(os.path.join(ref_7t_path, file))
        
        # Directly target 'susc' if present, otherwise fallback safely
        if 'susc' in mat:
            k = 'susc'
        elif 'qsm_7t' in mat:
            k = 'qsm_7t'
        else:
            k = [key for key in mat.keys() if not key.startswith('__')][0]
            
        tensor = torch.from_numpy(mat[k].astype(float)).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            z = teacher_net(tensor)
            manifold_database.append(z.cpu())
            
    manifold_tensor = torch.cat(manifold_database, dim=0)
    torch.save(manifold_tensor, database_path)
    print(f"[✓] Compiled {manifold_tensor.shape[0]} patches into {database_path}\n")




#%%








# Load the compiled database
manifold_db = torch.load(database_path, map_location=device)  # Shape: [N, 768]

# Load frozen reconstruction backbone network
recon_net = QSMnet().to(device)
if os.path.exists(stage2_model_checkpoint):
    recon_net.load_state_dict(torch.load(stage2_model_checkpoint, map_location=device))
    print(f"[✓] Warm-start network weights loaded from: {stage2_model_checkpoint}")
else:
    print(f"❌ Error: Stage 2 weight file missing at {stage2_model_checkpoint}")
    exit()
    
recon_net.eval()
for param in recon_net.parameters(): param.requires_grad = False  

# Load normalization statistics
stats = sio.loadmat(stats_path)
phs_mean = torch.tensor(stats['inp_mean']).float().to(device)
phs_std  = torch.tensor(stats['inp_std']).float().to(device)
y_mean   = torch.tensor(stats['out_mean']).float().to(device)
y_std    = torch.tensor(stats['out_std']).float().to(device)

criterion_dc = DataConsistencyLoss(matrix_size=(176, 176, 160)).to(device)

# =====================================================================
# 4. ACTIVE VOXEL-WISE SEMANTIC REFINEMENT LOOP (STTR)
# =====================================================================
print(f"\n[*] Starting Semantic Test-Time Refinement Loop. Optimizing map voxels...")
print("-" * 95)

for i in patients_list:
    for j in range(1, 6):
        try:
            phs_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/phs{j}.mat"))['phs']
            sus     = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/cos{j}.mat"))['cos']
            msk_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/msk{j}.mat"))['msk']
        except FileNotFoundError:
            continue

        phs = torch.from_numpy(phs_raw).float().unsqueeze(0).unsqueeze(0).to(device)
        msk = torch.from_numpy(msk_raw).float().unsqueeze(0).unsqueeze(0).to(device)
        phs = phs * msk
        phs_norm = (phs - phs_mean) / phs_std

        # --- STEP 1: COMPUTE INITIAL ESTIMATE MAP (χ₀) ---
        with torch.no_grad():
            chi_norm_0 = recon_net(phs_norm)
            chi_0 = (chi_norm_0 * y_std) + y_mean
            chi_0 = chi_0 * msk

        # --- STEP 2: DECLARE VOXELS AS THE ONLY PARAMETERS TO OPTIMIZE ---
        chi = nn.Parameter(chi_0.clone())
        sttr_optimizer = torch.optim.Adam([chi], lr=STTR_LR)

        # --- STEP 3: REFINE SUSCEPTIBILITY INTEGRITIES OVER ITERATIONS ---
        for step in range(STTR_ITERS):
            sttr_optimizer.zero_grad()
            chi_masked = chi * msk
            
            # Constraints paths
            loss_dc = criterion_dc(chi_masked, phs, msk)
            loss_p  = torch.mean(torch.abs(chi_masked - chi_0))
            
            # Crop native 64x64x64 central sub-patch
            chi_patch = chi_masked[:, :, 56:120, 56:120, 48:112]
            z_pred = teacher_net(chi_patch)  
            
            # Nearest Neighbor Manifold Search
            with torch.no_grad():
                distances = torch.sum((z_pred - manifold_db) ** 2, dim=1)
                nearest_idx = torch.argmin(distances)
                z_nearest = manifold_db[nearest_idx]
                
            loss_semantic = torch.mean((z_pred - z_nearest) ** 2)
            total_loss = (lambda_dc * loss_dc) + (lambda_prior * loss_p) + (lambda_sem * loss_semantic)
            
            total_loss.backward()
            sttr_optimizer.step()

        # Save prediction array back to CPU
        x_k_cpu = (chi * msk).squeeze().detach().cpu().numpy()
        sio.savemat(os.path.join(outdir, f"susc_cal_sttr_{i}_{j}.mat"), {"sus_cal": x_k_cpu})

        # Calculations stats
        psnr_val = compute_psnr(x_k_cpu, sus)
        rmse_val = compute_rmse(x_k_cpu, sus)
        hfen_val = compute_hfen(x_k_cpu, sus)
        # ssim_val = compute_ssim_numpy(x_k_cpu, sus)

        sus_copy = np.copy(sus)            
        sus_copy = np.expand_dims(sus_copy, axis=0)  # Shape becomes (1, depth, height, width)
        sus_copy = np.expand_dims(sus_copy, axis=0)  # Shape becomes (1, 1, depth, height, width)

        # [cite_start]Calculates 3D SSIM via requested module library parameters [cite: 1001]
        ssim_val = compute_ssim_numpy(x_k_cpu, sus_copy)


        print(f"   ↳ Patient_{i} Orient_{j} Optimized | "
              f"SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB | "
              f"RMSE: {rmse_val:.4f}% | HFEN: {hfen_val:.4f}")

        results.append({
            'patient': i, 'orientation': j, 'ssim': ssim_val, 'psnr': psnr_val, 'rmse': rmse_val, 'hfen': hfen_val
        })

# =====================================================================
# 5. DATASET CONSOLIDATION AND VISUALIZATION SUMMARY
# =====================================================================
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(outdir, "sttr_evaluation_results.csv")
df_results.to_csv(csv_save_path, index=False)

means = df_results[['ssim', 'psnr', 'rmse', 'hfen']].mean()
stds  = df_results[['ssim', 'psnr', 'rmse', 'hfen']].std()

print("\n" + "="*75)
print("   FINAL STAGE-III SEMANTIC TEST-TIME REFINEMENT SUMMARY (STTR)")
print("="*75)
print(f"  • Overall 3D SSIM : {means['ssim']:.4f} ± {stds['ssim']:.4f}")
print(f"  • Overall PSNR    : {means['psnr']:.2f} ± {stds['psnr']:.2f} dB")
print(f"  • Overall RMSE (%) : {means['rmse']:.4f} ± {stds['rmse']:.4f}%")
print(f"  • Overall HFEN    : {means['hfen']:.4f} ± {stds['hfen']:.4f}")
print("="*75)

patient_summary = df_results.groupby('patient')[['ssim', 'psnr', 'rmse', 'hfen']].mean()
print(patient_summary.to_string())
print("="*75)

summary_txt_path = os.path.join(outdir, "summary_metrics.txt")
with open(summary_txt_path, "w") as f:
    f.write("=== STAGE-III STTR REFINEMENT SUMMARY AGGREGATES ===\n")
    f.write(f"SSIM: {means['ssim']:.4f} ± {stds['ssim']:.4f}\n")
    f.write(f"PSNR: {means['psnr']:.2f} ± {stds['psnr']:.2f} dB\n")
    f.write(f"RMSE: {means['rmse']:.4f} ± {stds['rmse']:.4f}%\n")
    f.write(f"HFEN: {means['hfen']:.4f} ± {stds['hfen']:.4f}\n\n")
    f.write("=== PER PATIENT AVERAGES ===\n")
    f.write(patient_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(x_k_cpu[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[0].set_title('Stage-III STTR Refined Result (Slice 80)')
axes[0].axis('off')
axes[1].imshow(sus[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[1].set_title('Ground Truth COSMOS')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "sttr_verification_slice.png"), dpi=150)
plt.close()
print(f"✅ STTR verification complete. Outputs saved inside: {outdir}")