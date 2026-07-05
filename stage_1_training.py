#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:56:30 2026

@author: venkatesh
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

print('*******************************************************')
print('LAUNCHING STAGE-I SUPERVISED BASELINE TRAINING')
print('*******************************************************')
#%%
# =====================================================================
# 1. YOUR EXACT ARCHITECTURE (QSMnet Backbone)
# =====================================================================

from QSMnet import QSMnet

#%%

# =====================================================================
# 2. DATA UTILITIES AND LOSS LOGIC (Modernized Physics API)
# =====================================================================
class mydataloader(Dataset):
    def __init__(self, csv_file, root_dir):
        self.names = pd.read_csv(csv_file)  # Now resolves perfectly
        self.root_dir = root_dir  
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir, self.names['FileName'][idx])             
        data = sio.loadmat(file_name)
        phs  = torch.tensor(data['phs']).unsqueeze(dim=0).float()
        msk  = torch.tensor(data['msk']).unsqueeze(dim=0).float()
        sus  = torch.tensor(data['susc']).unsqueeze(dim=0).float()
        return phs, msk, sus






#%%

def dipole_kernel(matrix_size, voxel_size, B0_dir=[0, 0, 1]):
    [Y, X, Z] = np.meshgrid(
        np.linspace(-int(matrix_size[1]/2), int(matrix_size[1]/2)-1, matrix_size[1]),
        np.linspace(-int(matrix_size[0]/2), int(matrix_size[0]/2)-1, matrix_size[0]),
        np.linspace(-int(matrix_size[2]/2), int(matrix_size[2]/2)-1, matrix_size[2])
    )
    X = X / (matrix_size[0]) * voxel_size[0]
    Y = Y / (matrix_size[1]) * voxel_size[1]
    Z = Z / (matrix_size[2]) * voxel_size[2]
    
    numerator = np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2])
    denominator = np.square(X) + np.square(Y) + np.square(Z) + np.finfo(float).eps
    D = 1/3 - np.divide(numerator, denominator)
    D = np.where(np.isnan(D), 0, D)

    D = np.roll(D, int(np.floor(matrix_size[0]/2)), axis=0)
    D = np.roll(D, int(np.floor(matrix_size[1]/2)), axis=1)
    D = np.roll(D, int(np.floor(matrix_size[2]/2)), axis=2)
    return torch.tensor(np.float32(D)).unsqueeze(dim=0)

def sobel_kernel():
    s = [[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]]
    s = torch.FloatTensor(s)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    return torch.stack([sx, sy, sz]).unsqueeze(1)

def modernized_total_loss(chi, y, b, d, m, b_mean, b_std, y_mean, y_std, sobel):    
    # 1. Denormalize tensors back to absolute values for physical consistency
    chi_denorm = chi * y_std + y_mean
    b_denorm   = b   * b_std + b_mean

    # 2. Modern Complex FFT-based Dipole forward model mapping 
    chi_fft = torch.fft.fftn(chi_denorm.squeeze(1), dim=(-3, -2, -1))
    # Pointwise frequency domain multiplication with dipole grid
    b_hat = torch.fft.ifftn(d * chi_fft, dim=(-3, -2, -1)).real.unsqueeze(1)

    # Apply scanner tissue boundaries
    b_masked = b_denorm * m
    b_hat_masked = b_hat * m    

    # 3. Component Losses calculations
    loss_model = torch.mean(torch.abs(b_masked - b_hat_masked))
    loss_l1    = torch.mean(torch.abs(chi - y))
    
    difference = F.conv3d(y - chi, sobel, padding=1)
    loss_grad  = torch.mean(torch.abs(difference))
    
    # Blended optimization penalty weight vectors
    return (1.0 * loss_l1) + (0.5 * loss_model) + (0.1 * loss_grad)

#%%
# =====================================================================
# 3. MASTER RUNNER INTERACTION SCOPE
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 50

save_dir = 'savedModels/QSMnet_Stage1/'
os.makedirs(save_dir, exist_ok=True)

# Initialize model backbone structures 
net = QSMnet().to(device)
par = torch.nn.DataParallel(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Precompute physical dipole and sobel matrices
dk = dipole_kernel([64, 64, 64], [1, 1, 1], B0_dir=[0, 0, 1]).to(device)
ss = sobel_kernel().to(device)

# Load calibration stats files
stats_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat"
if os.path.exists(stats_path):
    stats = sio.loadmat(stats_path)
    b_mean = torch.tensor(stats['inp_mean']).float().to(device)
    b_std  = torch.tensor(stats['inp_std']).float().to(device)
    y_mean = torch.tensor(stats['out_mean']).float().to(device)
    y_std  = torch.tensor(stats['out_std']).float().to(device)
else:
    # Fallback initialization vectors
    b_mean, b_std, y_mean, y_std = torch.tensor(0.0).to(device), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)


csv_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv"
dir_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches"


# Setup standard Data loaders
import pandas as pd # import fallback constraint validation
traindata   = mydataloader('/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv', '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches')
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

valdata     = mydataloader('/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/val.csv', '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches')
valloader   = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


#%%
train_loss_history = []
val_loss_history = []

# Execution epochs engine
for epoch in range(1, epochs + 1):
    # --- TRAINING PASS ---
    net.train()
    runtrainloss = 0.0
    train_bar = tqdm.tqdm(trainloader, desc=f"Stage 1 Train Epoch [{epoch}/{epochs}]", leave=True)

    
    for data in train_bar:

        b, m, y = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # Apply standard calibration scalar normalizations
        b_norm = (b - b_mean) / b_std
        y_norm = (y - y_mean) / y_std
        
        optimizer.zero_grad()
        chi = par(b_norm)
        
        loss = modernized_total_loss(chi, y_norm, b_norm, dk, m, b_mean, b_std, y_mean, y_std, ss)
        loss.backward()
        optimizer.step()
        
        runtrainloss += loss.item()
        train_bar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
        
    avg_train_loss = runtrainloss / len(trainloader)
    train_loss_history.append(avg_train_loss)

    # --- VALIDATION PASS ---
    net.eval()
    runvalloss = 0.0
    val_bar = tqdm.tqdm(valloader, desc=f"Stage 1 Val   Epoch [{epoch}/{epochs}]", leave=True)
    
    with torch.no_grad():
        for data in val_bar:
            b, m, y = data[0].to(device), data[1].to(device), data[2].to(device)
            b_norm = (b - b_mean) / b_std
            y_norm = (y - y_mean) / y_std
            
            chi = par(b_norm)
            loss = modernized_total_loss(chi, y_norm, b_norm, dk, m, b_mean, b_std, y_mean, y_std, ss)
            runvalloss += loss.item()
            val_bar.set_postfix({"Val Loss": f"{loss.item():.6f}"})
            
    avg_val_loss = runvalloss / len(valloader)
    val_loss_history.append(avg_val_loss)

    print(f" -> Summary Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
    
    scheduler.step()
    
    # Save checkpoints after every epoch
    torch.save(net.state_dict(), os.path.join(save_dir, f"QSMnet_{epoch}_model.pth"))

# Save training graphs stats matrices
np.save(os.path.join(save_dir, 'trnloss.npy'), np.array(train_loss_history))
np.save(os.path.join(save_dir, 'valloss.npy'), np.array(val_loss_history))
print("✅ Stage-1 baseline training completed successfully! Checkpoints stored safely.")