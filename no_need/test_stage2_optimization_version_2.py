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
from monai.networks.nets import SwinUNETR

# Import model architecture directly from your verified repository code
from QSMnet import QSMnet

print('=' * 80)
print('LAUNCHING SCAN-SPECIFIC ASYMMETRIC DUAL-SPACE INFERENCE ENGINE')
print('=' * 80)

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
# 2. OPTIMIZATION OBJECTIVES & TEACHER INFRASTRUCTURE
# =====================================================================

class DataConsistencyLoss(nn.Module):
    """Evaluated on the FULL 176x176x160 space to prevent field truncation errors."""
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


class TestTimeContrastiveLoss(nn.Module):
    """Pulls latent embeddings toward the high-field sharp anchor domain."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_pred, z_7t_ref):
        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        z_7t_norm = F.normalize(z_7t_ref, p=2, dim=-1)
        sim = torch.matmul(z_pred_norm, z_7t_norm.T) / self.temperature
        return -torch.log(torch.exp(sim).mean() + 1e-8)


class FrozenSwinUNETRTeacher(nn.Module):
    """Extracts features from NATIVE 64x64x64 patches without sub-sampling blur."""
    def __init__(self, checkpoint_path):
        super().__init__()
        self.swin_unetr = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_checkpoint=False)
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        s_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else (checkpoint["model"] if "model" in checkpoint else checkpoint)
        encoder_dict = {k[11:] if k.startswith("swin_unetr.") else (k[7:] if k.startswith("module.") else k): v for k, v in s_dict.items()}
        self.swin_unetr.load_state_dict(encoder_dict, strict=False)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        hidden_states_out = self.swin_unetr.swinViT(x)
        pooled = self.gap(hidden_states_out[4])
        return pooled.view(pooled.size(0), -1)

# =====================================================================
# 3. GLOBAL CONFIGURATION & REFERENCE LOADING
# =====================================================================
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

# ⚙️ TEST-TIME OPTIMIZATION HYPERPARAMETERS[cite: 1, 2]
TEST_TIME_ITERS = 25  
TEST_TIME_LR    = 5e-5 

raw_data_path = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/raw_data_names_modified/'
data_path     = '/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/'
stats_path    = os.path.join(data_path, 'csv_files/tr-stats.mat')
ref_7t_path   = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches/"
teacher_path  = "pretrained/ssl_pretrained_weights.pth"
warm_start_checkpoint = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_Other_Experiments/QSMnet/savedModels/QSMnet/27Dec_0244pm_model/QSMnet_5_model.pth"

patients_list = [7, 8, 9, 10, 11, 12]
results = []

outdir = "savedModels_stage2/test_time_optimized_predictions/"
os.makedirs(outdir, exist_ok=True)

# Boot up frozen encoder teacher
teacher_net = FrozenSwinUNETRTeacher(checkpoint_path=teacher_path).to(device).eval()

# Load normalization stats
stats = sio.loadmat(stats_path)
phs_mean = torch.tensor(stats['inp_mean']).float().to(device)
phs_std  = torch.tensor(stats['inp_std']).float().to(device)
y_mean   = torch.tensor(stats['out_mean']).float().to(device)
y_std    = torch.tensor(stats['out_std']).float().to(device)

# Subspace Anchoring: Pre-extract target 7T domain vector[cite: 1, 2]
ref_7t_file = sorted(os.listdir(ref_7t_path))[0]
ref_7t_mat = sio.loadmat(os.path.join(ref_7t_path, ref_7t_file))
k_7t = 'qsm_7t' if 'qsm_7t' in ref_7t_mat else [k for k in ref_7t_mat.keys() if not k.startswith('__')][0]
ref_7t_tensor = torch.from_numpy(ref_7t_mat[k_7t].astype(float)).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    z_7t_target = teacher_net(ref_7t_tensor)

# Initialize full resolution loss operators
criterion_dc  = DataConsistencyLoss(matrix_size=(176, 176, 160)).to(device)
criterion_sem = TestTimeContrastiveLoss(temperature=0.07).to(device)

# =====================================================================
# 4. ACTIVE ASYMMETRIC TEST-TIME OPTIMIZATION LOOP[cite: 1, 2]
# =====================================================================
print(f"\n[*] Launching Active Adaptation Pipeline. Optimizing model weights per volume...")
print("-" * 85)

for i in patients_list:
    print(f"\n>>>> 🔄 Re-initializing Fresh Baseline Weight Map for Patient_{i}")
    
    # Reload baseline model weights to guarantee clean cross-patient separation[cite: 1, 2]
    dw = QSMnet().to(device)
    state_dict = torch.load(warm_start_checkpoint, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    dw.load_state_dict(state_dict)
    dw.train()  # Crucial: enables tracking local adaptation gradients[cite: 1, 2]
    
    test_time_optimizer = torch.optim.Adam(dw.parameters(), lr=TEST_TIME_LR)
    
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

        # --- Test-Time Optimization Iterations[cite: 1, 2] ---
        for step in range(TEST_TIME_ITERS):
            test_time_optimizer.zero_grad()
            
            # Step A: Predict on the FULL 176x176x160 volume (Preserves physics continuity)[cite: 1, 2]
            chi_norm_pred = dw(phs_norm)
            sus_cal = (chi_norm_pred * y_std) + y_mean
            sus_cal = sus_cal * msk
            
            # Step B: Compute DC Loss on full volume[cite: 1, 2]
            loss_dc = criterion_dc(sus_cal, phs, msk)
            
            # Step C: Extract a native 64x64x64 sub-patch from the brain volume center[cite: 1, 2]
            # Slicing bounds: H(56:120), W(56:120), D(48:112)
            sus_cal_patch = sus_cal[:, :, 56:120, 56:120, 48:112]
            
            # Step D: Pull patch semantics toward the 7T anchor[cite: 1, 2]
            z_pred = teacher_net(sus_cal_patch)
            loss_sem = criterion_sem(z_pred, z_7t_target)
            
            # Step E: Backward step updates parameters to align with 7T quality styles[cite: 1, 2]
            total_step_loss = loss_dc + (0.01 * loss_sem)
            total_step_loss.backward()
            test_time_optimizer.step()

        # Final evaluation inference step for this scan setup
        dw.eval()
        with torch.no_grad():
            chi_norm_final = dw(phs_norm)
            sus_final = (chi_norm_final * y_std) + y_mean
            sus_final = sus_final * msk
            x_k_cpu = sus_final.squeeze().cpu().numpy()

        # Save processed matrix cleanly
        sio.savemat(os.path.join(outdir, f"susc_cal_adapted_{i}_{j}.mat"), {"sus_cal": x_k_cpu})

        # Calculate exact metrics using custom verification modules
        psnr_val = compute_psnr(x_k_cpu, sus)
        rmse_val = compute_rmse(x_k_cpu, sus)
        hfen_val = compute_hfen(x_k_cpu, sus)
        ssim_val = compute_ssim_numpy(x_k_cpu, sus)

        print(f"   ↳ Orientation {j} Done | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB | RMSE: {rmse_val:.4f}%")

        results.append({
            'patient': i, 'orientation': j, 'ssim': ssim_val, 'psnr': psnr_val, 'rmse': rmse_val, 'hfen': hfen_val
        })
        dw.train()  # Re-enable gradient graphs for the next orientation run[cite: 1, 2]

# =====================================================================
# 5. DATASET CONSOLIDATION AND VISUALIZATION SUMMARY
# =====================================================================
df_results = pd.DataFrame(results)
csv_save_path = os.path.join(outdir, "test_time_optimization_results.csv")
df_results.to_csv(csv_save_path, index=False)

# Compute averages and variance distributions
means = df_results[['ssim', 'psnr', 'rmse', 'hfen']].mean()
stds  = df_results[['ssim', 'psnr', 'rmse', 'hfen']].std()

print("\n" + "="*65)
print("   FINAL ADAPTED TEST-TIME OPTIMIZATION PERFORMANCE SUMMARY")
print("="*65)
print(f"  • Overall 3D SSIM : {means['ssim']:.4f} ± {stds['ssim']:.4f}")
print(f"  • Overall PSNR    : {means['psnr']:.2f} ± {stds['psnr']:.2f} dB")
print(f"  • Overall RMSE (%) : {means['rmse']:.4f} ± {stds['rmse']:.4f}%")
print(f"  • Overall HFEN    : {means['hfen']:.4f} ± {stds['hfen']:.4f}")
print("="*65)

print("\n[Granular Breakdown] Mean Metrics Grouped Per Patient:")
patient_summary = df_results.groupby('patient')[['ssim', 'psnr', 'rmse', 'hfen']].mean()
print(patient_summary.to_string())
print("="*65)

# Save text summary log
summary_txt_path = os.path.join(outdir, "summary_metrics.txt")
with open(summary_txt_path, "w") as f:
    f.write("=== TEST-TIME OPTIMIZATION SUMMARY AGGREGATES ===\n")
    f.write(f"SSIM: {means['ssim']:.4f} ± {stds['ssim']:.4f}\n")
    f.write(f"PSNR: {means['psnr']:.2f} ± {stds['psnr']:.2f} dB\n")
    f.write(f"RMSE: {means['rmse']:.4f} ± {stds['rmse']:.4f}%\n")
    f.write(f"HFEN: {means['hfen']:.4f} ± {stds['hfen']:.4f}\n\n")
    f.write("=== PER PATIENT AVERAGES ===\n")
    f.write(patient_summary.to_string())

# Render visual quality slice cross check
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(x_k_cpu[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[0].set_title('Test-Time Optimized Prediction (Slice 80)')
axes[0].axis('off')

axes[1].imshow(sus[:, :, 80], cmap='gray', clim=(-0.1, 0.1))
axes[1].set_title('Ground Truth COSMOS')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "optimization_verification_slice.png"), dpi=150)
plt.close()
print(f"✅ Metric validation complete. Comparison slice generated inside: {outdir}")