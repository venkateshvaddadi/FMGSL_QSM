#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:22:18 2026

@author: venkatesh
"""

import os
import random
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR

# =====================================================================
# 1. INTEGRATED RECONSTRUCTION NETWORK (QSMNet Architecture Backbone)
# =====================================================================



# =====================================================================
# 2. UNPAIRED PIPELINE COMPONENTS & OBJECTIVES
# =====================================================================

class QSMLPCNNUnpairedDataset(Dataset):
    """Loads 3T data via train.csv and picks random 7T instances from lpcnn_patches.csv."""
    def __init__(self, csv_3t, csv_7t, data_dir_3t, data_dir_7t):
        self.data_dir_3t = data_dir_3t
        self.data_dir_7t = data_dir_7t
        self.df_3t = pd.read_csv(csv_3t)
        self.df_7t = pd.read_csv(csv_7t)
        self.files_3t = self.df_3t.iloc[:, 0].tolist()
        self.files_7t = self.df_7t.iloc[:, 0].tolist()
        print(f"[Dataset] Parsed -> 3T count: {len(self.files_3t)} | 7T count: {len(self.files_7t)}")

    def __len__(self):
        return len(self.files_3t)

    def __getitem__(self, idx):
        filename_3t = self.files_3t[idx]
        mat_3t = sio.loadmat(os.path.join(self.data_dir_3t, filename_3t))
        
        filename_7t = random.choice(self.files_7t)
        mat_7t = sio.loadmat(os.path.join(self.data_dir_7t, filename_7t))
        
        k_field_3t = 'field_3t' if 'field_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][0]
        k_qsm_3t = 'qsm_3t' if 'qsm_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][1]
        k_qsm_7t = 'qsm_7t' if 'qsm_7t' in mat_7t else [k for k in mat_7t.keys() if not k.startswith('__')][0]
        
        f_3t = torch.from_numpy(mat_3t[k_field_3t].astype(float)).unsqueeze(0).float()
        q_3t = torch.from_numpy(mat_3t[k_qsm_3t].astype(float)).unsqueeze(0).float()
        q_7t = torch.from_numpy(mat_7t[k_qsm_7t].astype(float)).unsqueeze(0).float()
        return {"input_field_3t": f_3t, "target_qsm_3t": q_3t, "reference_qsm_7t": q_7t}

class FrozenSwinUNETRTeacher(nn.Module):
    """Loads foundation SSL parameters and freezes parameters from receiving optimization metrics."""
    def __init__(self, checkpoint_path):
        super().__init__()
        self.swin_unetr = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_checkpoint=False)
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        print(f"[*] Parsing frozen master weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        s_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else (checkpoint["model"] if "model" in checkpoint else checkpoint)
        
        encoder_dict = {}
        for k, v in s_dict.items():
            if k.startswith("module."): k = k[7:]
            if k.startswith("swin_unetr."): encoder_dict[k[11:]] = v
            else: encoder_dict[k] = v
        self.swin_unetr.load_state_dict(encoder_dict, strict=False)
        
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, x):
        hidden_states_out = self.swin_unetr.swinViT(x)
        bottleneck = hidden_states_out[4] # Sized [B, 768, 4, 4, 4]
        pooled = self.gap(bottleneck)
        return pooled.view(pooled.size(0), -1)

class DomainClusterContrastiveLoss(nn.Module):
    """Enforces intra-domain clustering compactness and drives cross-field separation."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_3t, z_7t):
        batch_size = z_3t.size(0)
        device = z_3t.device
        
        z_3t_norm = F.normalize(z_3t, p=2, dim=-1)
        z_7t_norm = F.normalize(z_7t, p=2, dim=-1)
        embeddings = torch.cat([z_3t_norm, z_7t_norm], dim=0)
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        domain_labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).to(device).unsqueeze(0)
        
        same_domain_mask = torch.eq(domain_labels.T, domain_labels)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        positive_mask = same_domain_mask & ~self_mask
        
        max_logits, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - max_logits.detach()
        log_prob = logits - torch.log((torch.exp(logits) * (~self_mask)).sum(dim=1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        return -mean_log_prob_pos.mean()

class DataConsistencyLoss(nn.Module):
    """Anchors inversion inside physical constraints using a Fourier dipole kernel."""
    def __init__(self, matrix_size=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0)):
        super().__init__()
        Nx, Ny, Nz = matrix_size
        dx, dy, dz = voxel_size
        kx = torch.fft.fftfreq(Nx, d=dx)
        ky = torch.fft.fftfreq(Ny, d=dy)
        kz = torch.fft.fftfreq(Nz, d=dz)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        
        k2 = KX**2 + KY**2 + KZ**2
        k2 = torch.where(k2 == 0, torch.ones_like(k2) * 1e-12, k2)
        kernel = (1.0 / 3.0) - (KZ**2 / k2)
        kernel[0, 0, 0] = 0.0
        
        self.register_buffer("D", kernel)
        self.mse = nn.MSELoss()

    def forward(self, pred_qsm, input_local_field):
        pred_fft = torch.fft.fftn(pred_qsm.squeeze(1), dim=(-3, -2, -1))
        sim_field_fft = self.D * pred_fft
        sim_field = torch.fft.ifftn(sim_field_fft, dim=(-3, -2, -1)).real
        return self.mse(sim_field, input_local_field.squeeze(1))

# =====================================================================
# 3. GLOBAL CONTINUOUS RUNNER PIPELINE
# =====================================================================

if __name__ == "__main__":
    # Absolute Path Directories Verification Targets
    csv_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv"
    csv_7t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/lpcnn_patches.csv"
    dir_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches"
    dir_7t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches"
    
    teacher_checkpoint = "pretrained/ssl_pretrained_weights.pth"
    stage1_checkpoint = "pretrained/qsmnet_stage1_baseline.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 50
    lr = 1e-4
    
    # Objective trade-off tuning constants
    lambda_dc = 1.0
    lambda_rec = 1.0
    lambda_sem = 0.05

    # 1. Pipeline Model Loading Mechanics
    recon_net = QSMNet().to(device)
    if os.path.exists(stage1_checkpoint):
        recon_net.load_state_dict(torch.load(stage1_checkpoint, map_location=device))
        print(f"[*] Warm-up checkpoints matched: {stage1_checkpoint}")
    recon_net.train()
    
    teacher_net = FrozenSwinUNETRTeacher(checkpoint_path=teacher_checkpoint).to(device)
    teacher_net.eval()
    
    # 2. Initialize Loss Engines
    criterion_dc = DataConsistencyLoss(matrix_size=(64, 64, 64)).to(device)
    criterion_rec = torch.nn.L1Loss()
    criterion_sem = DomainClusterContrastiveLoss(temperature=0.07).to(device)
    
    optimizer = torch.optim.Adam(recon_net.parameters(), lr=lr)

    # 3. Loader Instantiation
    dataset = QSMLPCNNUnpairedDataset(csv_3t=csv_3t, csv_7t=csv_7t, data_dir_3t=dir_3t, data_dir_7t=dir_7t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    print("\n" + "="*50)
    print("BEGINNING INTEGRATED STAGE 2 CLUSTERING ITERATIONS")
    print("="*50)

    # 4. Master Optimization Execution
    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_dc, epoch_rec, epoch_sem = 0.0, 0.0, 0.0, 0.0
        
        for batch in dataloader:
            b_3t = batch["input_field_3t"].to(device)
            chi_3t_gt = batch["target_qsm_3t"].to(device)
            chi_7t_ref = batch["reference_qsm_7t"].to(device)
            
            optimizer.zero_grad()
            
            # Reconstruction pass maps field arrays to predicted susceptibilities
            chi_3t_pred = recon_net(b_3t)
            
            # Latent extraction path handles representation generation
            with torch.no_grad():
                z_3t_pred = teacher_net(chi_3t_pred)
                z_7t_ref = teacher_net(chi_7t_ref)
                
            # Compute independent tracking errors
            loss_dc = criterion_dc(chi_3t_pred, b_3t)
            loss_rec = criterion_rec(chi_3t_pred, chi_3t_gt)
            loss_sem = criterion_sem(z_3t_pred, z_7t_ref)
            
            loss_total = (lambda_dc * loss_dc) + (lambda_rec * loss_rec) + (lambda_sem * loss_sem)
            
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            epoch_dc += loss_dc.item()
            epoch_rec += loss_rec.item()
            epoch_sem += loss_sem.item()
            
        num_batches = len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] -> "
              f"Total Loss: {epoch_loss/num_batches:.4f} | "
              f"DC: {epoch_dc/num_batches:.4f} | "
              f"Rec: {epoch_rec/num_batches:.4f} | "
              f"Semantic (Clusters): {epoch_sem/num_batches:.4f}")

                
        os.makedirs("pretrained", exist_ok=True)
        torch.save(recon_net.state_dict(), f"pretrained/qsmnet_stage2_epoch_{epoch}.pt")
        print(f" -> Snapshot saved cleanly.")