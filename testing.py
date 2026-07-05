#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:35:00 2026

@author: venkatesh
"""

import os
import time
import random
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

print('=' * 70)
print('LAUNCHING UNIFIED METRICS EVALUATION ENGINE')
print('=' * 70)

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
    if mse == 0:
        return float('inf')

    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_rmse(chi_recon, chi_true):
    chi_recon = np.asarray(chi_recon)
    chi_true = np.asarray(chi_true)
    numerator = np.linalg.norm(chi_recon.ravel() - chi_true.ravel())
    denominator = np.linalg.norm(chi_true.ravel())
    if denominator == 0:
        return float('inf')
    return 100 * numerator / denominator

def compute_hfen(img1, img2):
    img1 = np.squeeze(img1).astype(np.float64)
    img2 = np.squeeze(img2).astype(np.float64)
    
    if img1.ndim != 3:
        raise ValueError(f"Expected 3D volumes after squeezing, but got shape {img1.shape}")

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

# --- 3D SSIM ENGINE CORE ---
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
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    return _ssim_3D(img1, img2, window.type_as(img1), window_size, channel, size_average)

def compute_ssim_numpy(img1_np, img2_np, window_size=11, size_average=True):
    img1_t = torch.from_numpy(img1_np).float()
    img2_t = torch.from_numpy(img2_np).float()
    
    if img1_t.ndimension() == 3:
        img1_t = img1_t.unsqueeze(0).unsqueeze(0)
        img2_t = img2_t.unsqueeze(0).unsqueeze(0)
    elif img1_t.ndimension() == 4:
        img1_t = img1_t.unsqueeze(1)
        img2_t = img2_t.unsqueeze(1)

    if torch.cuda.is_available():
        img1_t, img2_t = img1_t.cuda(), img2_t.cuda()
    with torch.no_grad():
        return ssim3D(img1_t, img2_t, window_size=window_size, size_average=size_average).item()

# =====================================================================
# 2. ARCHITECTURE IMPORT (QSMnet Backbone)
# =====================================================================
from QSMnet import QSMnet

# =====================================================================
# 3. GLOBAL CONFIGURATION & RUNNER SCOPE
# =====================================================================
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

# 🔴 TOGGLE CONFIGURATION SWITCH: Set to 1 for Stage-1 Baseline, or 2 for Stage-2 Enhanced model
TEST_STAGE = 1

raw_data_path = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
data_path     = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/'
stats_path    = os.path.join(data_path, 'csv_files/tr-stats.mat')

patients_list = [7, 8, 9, 10, 11, 12]
results = []  # Metric tracking warehouse array

if TEST_STAGE == 1:
    print("[*] Evaluation Target: STAGE-1 Supervised Baseline Model")
    model_path = "savedModels/QSMnet_Stage1/QSMnet_50_model.pth"
    model_path = "savedModels/QSMnet_Stage1/QSMnet_50_model.pth"
    model_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_Other_Experiments/QSMnet/savedModels/QSMnet/27Dec_0244pm_model/QSMnet_5_model.pth"
    outdir = "savedModels/QSMnet_Stage1/predictions_epoch_50/"
else:
    print("[*] Evaluation Target: STAGE-2 Semantic Fine-Tuned Model")
    model_path = "savedModels_stage2/QSMnet_Stage2_Epoch_50.pth"
    outdir = "savedModels_stage2/predictions_epoch_50/"

os.makedirs(outdir, exist_ok=True)

# Load model layouts onto system memory
dw = QSMnet().to(device)
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    dw.load_state_dict(state_dict)
    print(f"[✓] Checkpoint weight states successfully initialized from: {model_path}")
else:
    print(f"❌ Error: Model weights file missing at {model_path}")
    exit()

dw.eval()

# Load scaling stats
stats = sio.loadmat(stats_path)
phs_mean = torch.tensor(stats['inp_mean']).float().to(device)
phs_std  = torch.tensor(stats['inp_std']).float().to(device)
y_mean   = torch.tensor(stats['out_mean']).float().to(device)
y_std    = torch.tensor(stats['out_std']).float().to(device)

# =====================================================================
# 4. VOLUMETRIC METRICS EVALUATION PIPELINE
# =====================================================================
print(f"\n[*] Launching tracking loop. Logging metric telemetry into destination file outputs...")
print("-" * 80)

criterion_l1 = nn.L1Loss()

with torch.no_grad():
    for i in patients_list:
        print(f"\n>>>> Iterating over Target Volume Element: Patient_{i}")
        
        for j in range(1, 6):
            try:
                phs_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/phs{j}.mat"))['phs']
                sus     = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/cos{j}.mat"))['cos']
                msk_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{i}/msk{j}.mat"))['msk']
            except FileNotFoundError:
                continue

            # Shape into 5D spatial map parameters
            phs = torch.from_numpy(phs_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            msk = torch.from_numpy(msk_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            y_gt = torch.from_numpy(sus).float().unsqueeze(0).unsqueeze(0).to(device)

            # Apply background maps and normalizations
            phs = phs * msk
            phs_norm = (phs - phs_mean) / phs_std
            y_norm   = (y_gt - y_mean) / y_std

            # Execution pass
            chi_norm_pred = dw(phs_norm)
            
            # Compute evaluation L1 loss item
            loss = criterion_l1(chi_norm_pred, y_norm)

            # Map back to un-normalized absolute scales (ppm)
            sus_cal = (chi_norm_pred * y_std) + y_mean
            sus_cal = sus_cal * msk
            
            x_k_cpu = sus_cal.squeeze().cpu().numpy()

            # Save the processed prediction matrix cleanly
            sio.savemat(os.path.join(outdir, f"susc_cal_{i}_{j}.mat"), {"sus_cal": x_k_cpu})

            # =========================================================
            # INTEGRATED EVALUATION METRICS OPERATIONS
            # =========================================================
            psnr_val = compute_psnr(x_k_cpu, sus)
            rmse_val = compute_rmse(x_k_cpu, sus)
            hfen_val = compute_hfen(x_k_cpu, sus)
            ssim_val = compute_ssim_numpy(x_k_cpu, sus)

            # Print telemetry outputs matching progress terminal logs
            print({
                'patient': i,
                'orientation': j,
                'SSIM': f'{ssim_val:.4f}',
                'Loss': f'{loss.item():.2e}',
                'PSNR': f'{psnr_val:.2f}',
                'RMSE': f'{rmse_val:.4f}',
                'HFEN': f'{hfen_val:.4f}',
            })

            # Append structured results maps cleanly
            result_dict = {
                'patient': i,
                'orientation': j,
                'loss': loss.item(),
                'ssim': ssim_val,
                'psnr': psnr_val,
                'rmse': rmse_val,
                'hfen': hfen_val
            }
            results.append(result_dict)

# =====================================================================
# 5. DATASET CONSOLIDATION AND VISUALIZATION SUMMARY
# =====================================================================
# Convert results metrics dictionary stack to structured pandas dataframes
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(outdir, "evaluation_results.csv")
df_results.to_csv(csv_save_path, index=False)
print(f"\n[✓] Telemetry complete! Structured evaluation tracking spreadsheet stored at: {csv_save_path}")

# Render cross-checking midplane validation slice tracking maps
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(x_k_cpu[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[0].set_title(f'Stage {TEST_STAGE} Output prediction (Slice 80)')
axes[0].axis('off')

axes[1].imshow(sus[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[1].set_title('Ground Truth COSMOS')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "metric_verification_slice.png"), dpi=150)
plt.close()
print(f"✅ Metric validation complete. Benchmark comparison slice generated successfully inside: {outdir}")


#%%

# =====================================================================
# 5. DATASET CONSOLIDATION AND VISUALIZATION SUMMARY
# =====================================================================
# Convert results metrics dictionary stack to structured pandas dataframes
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(outdir, "evaluation_results.csv")
df_results.to_csv(csv_save_path, index=False)
print(f"\n[✓] Telemetry complete! Structured evaluation tracking spreadsheet stored at: {csv_save_path}")

# --- NEW: COMPUTE AND DISPLAY METRIC AVERAGES ---
print("\n" + "="*60)
print(f"   FINAL AGGREGATED PERFORMANCE METRICS SUMMARY (STAGE {TEST_STAGE})")
print("="*60)

# Calculate global mean and standard deviation for formal paper/report recording
means = df_results[['loss', 'ssim', 'psnr', 'rmse', 'hfen']].mean()
stds  = df_results[['loss', 'ssim', 'psnr', 'rmse', 'hfen']].std()

print(f"  • Overall 3D SSIM : {means['ssim']:.4f} ± {stds['ssim']:.4f}")
print(f"  • Overall PSNR    : {means['psnr']:.2f} ± {stds['psnr']:.2f} dB")
print(f"  • Overall RMSE (%) : {means['rmse']:.4f} ± {stds['rmse']:.4f}%")
print(f"  • Overall HFEN    : {means['hfen']:.4f} ± {stds['hfen']:.4f}")
print(f"  • Model Eval Loss : {means['loss']:.2e} ± {stds['loss']:.2e}")
print("="*60)

# Granular Breakdown: Average per patient to profile individual subject variations
print("\n[Granular Breakdown] Mean Metrics Grouped Per Patient:")
patient_summary = df_results.groupby('patient')[['ssim', 'psnr', 'rmse', 'hfen']].mean()
print(patient_summary.to_string())
print("="*60)

# Save summary stats to a text file for easy reference
summary_txt_path = os.path.join(outdir, "summary_metrics.txt")
with open(summary_txt_path, "w") as f:
    f.write(f"=== STAGE {TEST_STAGE} AGGREGATED METRICS ===\n")
    f.write(f"SSIM: {means['ssim']:.4f} ± {stds['ssim']:.4f}\n")
    f.write(f"PSNR: {means['psnr']:.2f} ± {stds['psnr']:.2f} dB\n")
    f.write(f"RMSE: {means['rmse']:.4f} ± {stds['rmse']:.4f}%\n")
    f.write(f"HFEN: {means['hfen']:.4f} ± {stds['hfen']:.4f}\n\n")
    f.write("=== PER PATIENT AVERAGES ===\n")
    f.write(patient_summary.to_string())

print(f"[✓] Summary metric text logs stored at: {summary_txt_path}")

# Render cross-checking midplane validation slice tracking maps
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(x_k_cpu[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[0].set_title(f'Stage {TEST_STAGE} Output prediction (Slice 80)')
axes[0].axis('off')

axes[1].imshow(sus[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[1].set_title('Ground Truth COSMOS')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "metric_verification_slice.png"), dpi=150)
plt.close()
print(f"✅ Metric validation complete. Benchmark comparison slice generated successfully inside: {outdir}")