#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:31:11 2026

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:30:00 2026

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
import matplotlib.pyplot as plt
import tqdm

# Import model architecture directly from your verified repository code
from QSMnet import QSMnet

print('=' * 60)
print('LAUNCHING STAGE-II PIPELINE FULL-VOLUME EVALUATION')
print('=' * 60)

# -----------------------------------------------------------------
# 1. Environment and Path Configurations
# -----------------------------------------------------------------
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

raw_data_path = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
data_path     = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/'
stats_path    = os.path.join(data_path, 'csv_files/tr-stats.mat')

# Define test evaluation tracking list
patients_list = [7, 8, 9, 10, 11, 12]

# Target your saved Stage 2 checkpoint files seamlessly
experiments_folder = "savedModels_stage2"
model_path = os.path.join(experiments_folder, "QSMnet_Stage2_Epoch_50.pth")

# Create output directories cleanly
outdir = os.path.join(experiments_folder, "predictions_epoch_50/")
os.makedirs(outdir, exist_ok=True)

# -----------------------------------------------------------------
# 2. Model Initialization & Weights Injection
# -----------------------------------------------------------------
dw = QSMnet().to(device)

if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    # Automatically strip away DataParallel 'module.' wraps if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    dw.load_state_dict(state_dict)
    print(f"[✓] Model successfully loaded into hardware from: {model_path}")
else:
    print(f"⚠️ Warning: Target model weight file not found at {model_path}. Running with random state configuration.")

dw.eval()

# -----------------------------------------------------------------
# 3. Calibration Stats Tracking (Normalization Set up)
# -----------------------------------------------------------------
# FIX: Hard-coded to True to ensure preprocessed data properties map correctly
is_data_normalized = True 

if is_data_normalized and os.path.exists(stats_path):
    stats = sio.loadmat(stats_path)
    phs_mean = torch.tensor(stats['inp_mean']).float().to(device)
    phs_std  = torch.tensor(stats['inp_std']).float().to(device)
    sus_mean = torch.tensor(stats['out_mean']).float().to(device)
    sus_std  = torch.tensor(stats['out_std']).float().to(device)
    print("[*] Calibration mean/std variables successfully initialized.")
else:
    phs_mean, phs_std = torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
    sus_mean, sus_std = torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
    print("[!] Running inference without normalization mappings.")

# -----------------------------------------------------------------
# 4. Fallback Metric Calculations Definitions
# -----------------------------------------------------------------
def inline_psnr(gt, pred, mask):
    mse = np.mean(((gt - pred) * mask) ** 2)
    if mse == 0: return float('inf')
    max_val = np.max(gt * mask) - np.min(gt * mask)
    return 20 * np.log10(max_val / np.sqrt(mse))

def inline_rmse(gt, pred, mask):
    return np.sqrt(np.mean(((gt - pred) * mask) ** 2)) / (np.max(gt * mask) + 1e-8) * 100

# -----------------------------------------------------------------
# 5. Volumetric Testing Execution Loop
# -----------------------------------------------------------------
print("\n[*] Starting testing iteration pipeline across volumes...")
print("-" * 60)

with torch.no_grad():
    for patient_idx in patients_list:
        print(f"\n>>>> Processing Target Volume Element: Patient_{patient_idx}")
        
        for orientation_idx in range(1, 6):
            # Safe checking pathways to avoid broken iteration traces
            try:
                phs_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{patient_idx}/phs{orientation_idx}.mat"))['phs']
                sus_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{patient_idx}/cos{orientation_idx}.mat"))['cos']
                msk_raw = sio.loadmat(os.path.join(raw_data_path, f"patient_{patient_idx}/msk{orientation_idx}.mat"))['msk']
            except FileNotFoundError as e:
                print(f"   ↳ Skipping Orientation {orientation_idx}: File not found.")
                continue

            # Convert numpy structures to PyTorch 5D spatial tensors [B, C, H, W, D]
            phs = torch.from_numpy(phs_raw).float().unsqueeze(0).unsqueeze(0).to(device)
            msk = torch.from_numpy(msk_raw).float().unsqueeze(0).unsqueeze(0).to(device)

            # Apply scanner field background boundaries
            phs = phs * msk
            
            # Apply exact scaling configurations
            phs_norm = (phs - phs_mean) / phs_std

            start_tic = time.time()
            
            # Forward reconstruction map passes
            sus_cal_norm = dw(phs_norm)
            
            # Denormalize output values back to physical ppm scaling limits
            sus_cal = (sus_cal_norm * sus_std) + sus_mean
            
            # Apply the tissue mask to clean background values
            sus_cal = sus_cal * msk
            elapsed_time = time.time() - start_tic

            # Extract arrays safely to host memory CPU formats
            sus_cal_cpu = sus_cal.squeeze().cpu().numpy()
            
            # Save predictions cleanly into target Matlab formats
            sio.savemat(os.path.join(outdir, f"susc_cal_{patient_idx}_{orientation_idx}.mat"), {"sus_cal": sus_cal_cpu})

            # Calculate metrics safely matching precise 3D data dimension scopes
            psnr_score = inline_psnr(sus_raw, sus_cal_cpu, msk_raw)
            rmse_score = inline_rmse(sus_raw, sus_cal_cpu, msk_raw)

            print(f"   Orientation {orientation_idx} | Time: {elapsed_time:.3f}s | "
                  f"PSNR: {psnr_score:.2f} dB | RMSE: {rmse_score:.4f}%")

# -----------------------------------------------------------------
# 6. Quality Validation Slice Visualization Mapping
# -----------------------------------------------------------------
print("\n[*] Rendering comparative validation slice...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Grab the final processed volume slice at midplane index 80
axes[0].imshow(sus_cal_cpu[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[0].set_title('Stage 2 Enhanced Prediction')
axes[0].axis('off')

axes[1].imshow(sus_raw[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[1].set_title('Ground Truth COSMOS')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "comparison_slice.png"), dpi=150)
plt.show()
print(f"✅ Full validation complete. Slice visualization plot stored at: {outdir}comparison_slice.png")