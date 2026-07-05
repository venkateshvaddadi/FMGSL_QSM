import os
import torch
from torch.utils.data import DataLoader

# Import components across the repository architecture tree
from foundation_model import FrozenSwinUNETRTeacher
from contrastive_loss import DomainClusterContrastiveLoss
from qsm_dataset import QSMLPCNNUnpairedDataset
from dipole_kernel import DataConsistencyLoss
#%%
print("="*60)
print("LAUNCHING STAGE-II DOMAIN-CLUSTER UNPAIRED TRAINING")
print("="*60)

# -----------------------------------------------------------------
# 1. Paths & Environment Configurations
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4             # Standard batch alignment size
epochs = 50
lr = 1e-4

# Loss balance weights
lambda_dc = 1.0            # Physical guardrail anchor
lambda_rec = 1.0           # Supervised voxel-wise error mapping
lambda_sem = 0.05          # Domain contrastive macro-cluster weight

csv_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv"
csv_7t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/lpcnn_patches.csv"
dir_3t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches"
dir_7t = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches"

teacher_checkpoint = "pretrained/ssl_pretrained_weights.pth"
stage1_checkpoint="/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_Other_Experiments/QSMnet/savedModels/QSMnet/27Dec_0244pm_model/QSMnet_5_model.pth"

import scipy.io as sio
# Load your mean/std training statistics for correct data scaling
stats_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/tr-stats.mat"
stats = sio.loadmat(stats_path)

b_mean = torch.tensor(stats['inp_mean']).float().to(device)
b_std  = torch.tensor(stats['inp_std']).float().to(device)
y_mean = torch.tensor(stats['out_mean']).float().to(device)
y_std  = torch.tensor(stats['out_std']).float().to(device)
#%%
# -----------------------------------------------------------------
# 2. Pipeline Architecture Initialization
# -----------------------------------------------------------------
# Load your Trainable Reconstruction Network (Stage-1)

from QSMnet import QSMnet

recon_net = QSMnet().to(device)
if os.path.exists(stage1_checkpoint):
    recon_net.load_state_dict(torch.load(stage1_checkpoint, map_location=device))
    print(f"[*] Successfully loaded warm-up weights from: {stage1_checkpoint}")
recon_net.train()

#%%
# Initialize the Frozen Foundation Semantic Teacher
teacher_net = FrozenSwinUNETRTeacher(checkpoint_path=teacher_checkpoint).to(device)
teacher_net.eval() # Explicitly lock evaluation state (freezes BN/Dropout maps)

# Loss objective modules
criterion_dc = DataConsistencyLoss(matrix_size=(64, 64, 64)).to(device)
criterion_rec = torch.nn.L1Loss()
criterion_sem = DomainClusterContrastiveLoss(temperature=0.07).to(device)

# Configure the optimizer to update ONLY the reconstruction network layers
optimizer = torch.optim.Adam(recon_net.parameters(), lr=lr)

# -----------------------------------------------------------------
# 3. Data Loader Pipeline Instantiation
# -----------------------------------------------------------------
dataset = QSMLPCNNUnpairedDataset(csv_3t=csv_3t, csv_7t=csv_7t, data_dir_3t=dir_3t, data_dir_7t=dir_7t)
# Using multiple background workers for seamless batch delivery speed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

import tqdm  # <-- Added for live batch tracking
#%%
# -----------------------------------------------------------------
# 4. Master Optimization Loop
# -----------------------------------------------------------------
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    epoch_dc = 0.0
    epoch_rec = 0.0
    epoch_sem = 0.0
    
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", leave=True)    

    # for batch in dataloader:
    for batch in progress_bar:
        # Transfer input tensors to processing hardware
        b_3t = batch["input_field_3t"].to(device)
        chi_3t_gt = batch["target_qsm_3t"].to(device)
        chi_7t_ref = batch["reference_qsm_7t"].to(device)

        # Apply your exact data normalization transforms before forward passing
        b_norm = (b_3t - b_mean) / b_std
        y_norm = (chi_3t_gt - y_mean) / y_std

        
        # --- Forward Pass Execution ---
        optimizer.zero_grad()
        
        # 1. Process local field through trainable net to predict susceptibility
        chi_3t_pred = recon_net(b_3t)
        


        # 2. Compute representation vectors via Frozen Teacher
        with torch.no_grad():
            z_3t_pred = teacher_net(chi_3t_pred)
            z_7t_ref = teacher_net(chi_7t_ref)
            
        # --- Multi-Objective Loss Paths ---
        loss_dc = criterion_dc(chi_3t_pred, b_3t)
        loss_rec = criterion_rec(chi_3t_pred, chi_3t_gt)
        loss_sem = criterion_sem(z_3t_pred, z_7t_ref)
        
        # Total integrated training goal
        loss_total = (lambda_dc * loss_dc) + (lambda_rec * loss_rec) + (lambda_sem * loss_sem)
        
        # --- Optimization Control Step ---
        loss_total.backward()
        optimizer.step()
        
        # Metric accumulation logs
        epoch_loss += loss_total.item()
        epoch_dc += loss_dc.item()
        epoch_rec += loss_rec.item()
        epoch_sem += loss_sem.item()

        # Update progress bar metrics on the fly for immediate feedback
        progress_bar.set_postfix({
            "Loss": f"{loss_total.item():.4f}",
            "DC": f"{loss_dc.item():.4f}",
            "Rec": f"{loss_rec.item():.4f}",
            "Sem": f"{loss_sem.item():.4f}"
        })
        
    num_batches = len(dataloader)
    print(f"Epoch [{epoch}/{epochs}] -> "
          f"Total Loss: {epoch_loss/num_batches:.4f} | "
          f"L_DC: {epoch_dc/num_batches:.4f} | "
          f"L_Rec: {epoch_rec/num_batches:.4f} | "
          f"L_Sem (Clusters): {epoch_sem/num_batches:.4f}")
    
    # Periodic model parameter preservation
    if epoch % 10 == 0:
        os.makedirs("pretrained", exist_ok=True)
        torch.save(recon_net.state_dict(), f"pretrained/qsmnet_stage2_epoch_{epoch}.pt")
        print(f"[Checkpoint Save] Stored current model state snapshot for epoch {epoch}.")