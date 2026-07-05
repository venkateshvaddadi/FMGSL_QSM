#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:01:05 2026

@author: venkatesh
"""

import torch
import torch.nn as nn

def compute_dipole_kernel(matrix_size, voxel_size, b0_dir=(0, 0, 1), device="cuda"):
    """
    Computes the 3D Fourier-domain dipole kernel operator.
    
    Args:
        matrix_size (tuple/list): Spatial dimensions of the volume or patch, e.g., (64, 64, 64).
        voxel_size (tuple/list): Spatial resolution of voxels in mm, e.g., (1.0, 1.0, 1.0).
        b0_dir (tuple): Direction vector of the main magnetic field B0 (default is along Z-axis).
        device (str/torch.device): Device where the tensor will be allocated.
        
    Returns:
        torch.Tensor: Precomputed complex or real frequency-domain dipole kernel grid of shape [*matrix_size].
    """
    Nx, Ny, Nz = matrix_size
    dx, dy, dz = voxel_size
    
    # 1. Generate k-space frequency coordinates using Fourier-shifted frequencies
    kx = torch.fft.fftfreq(Nx, d=dx, device=device)
    ky = torch.fft.fftfreq(Ny, d=dy, device=device)
    kz = torch.fft.fftfreq(Nz, d=dz, device=device)
    
    # Create 3D grid coordinate configurations
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    
    # 2. Compute the square of the frequency magnitude vector (k^2)
    k2 = KX**2 + KY**2 + KZ**2
    
    # Handle the singularity coordinate at the DC center (k=0) to prevent DivisionByZero
    k2 = torch.where(k2 == 0, torch.ones_like(k2) * 1e-12, k2)
    
    # 3. Calculate projection along the B0 field vector direction
    # Standard formula assumes alignment along Z-axis (b0_dir = (0,0,1)), meaning k_z^2 / k^2
    b0_x, b0_y, b0_z = b0_dir
    k_b0 = KX * b0_x + KY * b0_y + KZ * b0_z
    
    # 4. Apply the mathematical physical dipole expression: D = 1/3 - (k_b0^2 / k^2)
    kernel = (1.0 / 3.0) - (k_b0**2 / k2)
    
    # Explicitly force the exact DC center frequency index to be exactly 0
    kernel[0, 0, 0] = 0.0
    
    return kernel


class DataConsistencyLoss(nn.Module):
    """
    Computes the physical Data Consistency Loss (L_dc) using a precomputed dipole kernel.
    Ensures that the forward field simulation of the predicted susceptibility volume matches 
    the raw input local field measurements.
    """
    def __init__(self, matrix_size=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0)):
        super().__init__()
        self.matrix_size = matrix_size
        self.voxel_size = voxel_size
        self.mse_loss = nn.MSELoss()
        
        # Register dipole kernel as a buffer so it automatically transfers with model.to(device)
        kernel = compute_dipole_kernel(matrix_size, voxel_size, device="cpu")
        self.register_buffer("D", kernel)

    def forward(self, pred_qsm, input_local_field):
        """
        Args:
            pred_qsm (torch.Tensor): Reconstructed QSM patch from model [B, 1, 64, 64, 64]
            input_local_field (torch.Tensor): Raw measured field map volume patch [B, 1, 64, 64, 64]
        Returns:
            torch.Tensor: Scalar physical data consistency error node
        """
        # Remove single channel dimension for 3D FFT calculation path: [B, H, W, D]
        pred_chi = pred_qsm.squeeze(1)
        local_field = input_local_field.squeeze(1)
        
        # 1. Forward FFT to convert predicted susceptibility into frequency domain
        chi_fft = torch.fft.fftn(pred_chi, dim=(-3, -2, -1))
        
        # 2. Point-wise frequency domain multiplication with physics dipole operator
        # Broadcasts buffer 'D' across batch dimension smoothly
        sim_field_fft = self.D * chi_fft
        
        # 3. Inverse FFT to map simulated field back to spatial domain image coordinates
        sim_field = torch.fft.ifftn(sim_field_fft, dim=(-3, -2, -1)).real
        
        # 4. Compute pixel-wise residual mean squared error match metric
        loss_dc = self.mse_loss(sim_field, local_field)
        return loss_dc
#%%
import torch

def test_physics_forward_model():
    print("\n=== Verification: Physics Dipole Forward Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate Data Consistency Loss module for 64x64x64 patches
    dc_loss_fn = DataConsistencyLoss(matrix_size=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0)).to(device)
    
    # 2. Mock a batch size of N=2 containing prediction outputs (requires gradient tracking)
    # and scanned input local fields
    mock_pred_qsm = torch.randn(2, 1, 64, 64, 64, dtype=torch.float32, requires_grad=True, device=device)
    mock_local_field = torch.randn(2, 1, 64, 64, 64, dtype=torch.float32, device=device)
    
    print(f"[*] Input Predicted QSM shape: {list(mock_pred_qsm.shape)}")
    print(f"[*] Input Local Field shape:    {list(mock_local_field.shape)}")
    print(f"[*] Registered Buffer Operator Shape: {list(dc_loss_fn.D.shape)}")
    
    # 3. Compute loss tracking pass
    loss_dc = dc_loss_fn(mock_pred_qsm, mock_local_field)
    print(f" -> Computed L_DC Loss Scalar: {loss_dc.item():.6f}")
    
    # 4. Check backpropagation functionality
    loss_dc.backward()
    print(f" -> Is gradient populated for trainable QSM weights? {mock_pred_qsm.grad is not None}")
    print("==================================================")
    
    assert loss_dc.item() > 0, "Loss calculation should be a positive scalar."
    assert mock_pred_qsm.grad is not None, "Autograd connection broke during Fourier tracking loop."
    print("✅ Physics Dipole forward model verified successfully! Ready to secure your learning boundaries.")

if __name__ == "__main__":
    test_physics_forward_model()