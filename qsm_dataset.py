import os
import random
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

class QSMLPCNNUnpairedDataset(Dataset):
    """
    Unpaired 3D Patch Dataset for Stage-2 Domain-Aware Contrastive Learning.
    Clean, highly readable version using explicit string key names directly.
    """
    def __init__(self, csv_3t, csv_7t, data_dir_3t, data_dir_7t):
        self.data_dir_3t = data_dir_3t
        self.data_dir_7t = data_dir_7t
        
        # Load both tracking records
        self.df_3t = pd.read_csv(csv_3t)
        self.df_7t = pd.read_csv(csv_7t)
        
        self.files_3t = self.df_3t.iloc[:, 0].tolist()
        self.files_7t = self.df_7t.iloc[:, 0].tolist()

        print("[Dataset Init] Configured Direct Key Mapping Pipeline:")
        print(f" -> 3T Patch Target Count: {len(self.files_3t)}")
        print(f" -> 7T LPCNN Reference Count: {len(self.files_7t)}")

    def __len__(self):
        return len(self.files_3t)

    def __getitem__(self, idx):
        # 1. Load the sequential 3T subject mat file
        filename_3t = self.files_3t[idx]
        path_3t = os.path.join(self.data_dir_3t, filename_3t)
        mat_3t = sio.loadmat(path_3t)
        
        # 2. Sample an unpaired random 7T LPCNN reference mat file
        filename_7t = random.choice(self.files_7t)
        path_7t = os.path.join(self.data_dir_7t, filename_7t)
        mat_7t = sio.loadmat(path_7t)
        
        # 3. Explicit Key Mapping: Direct, clear, and bulletproof
        # Completely avoids MATLAB metadata index offsets by asking for the keys by name
        field_3t   = mat_3t['phs'].astype(float)    # 3T Raw Phase input
        qsm_3t_gt  = mat_3t['susc'].astype(float)   # 3T Ground Truth Susceptibility map
        qsm_7t_ref = mat_7t['susc'].astype(float)   # 7T Reference Susceptibility map (Pruned directly!)

        # 4. Format arrays directly into PyTorch tensors: [1, 64, 64, 64]
        field_3t_tensor = torch.from_numpy(field_3t).unsqueeze(0).float()
        qsm_3t_tensor   = torch.from_numpy(qsm_3t_gt).unsqueeze(0).float()
        qsm_7t_tensor   = torch.from_numpy(qsm_7t_ref).unsqueeze(0).float()

        return {
            "input_field_3t": field_3t_tensor,
            "target_qsm_3t": qsm_3t_tensor,
            "reference_qsm_7t": qsm_7t_tensor
        }
#%%
# =====================================================================
# SYSTEM EXECUTION AND DUMMY BATCH LOAD TEST
# =====================================================================
if __name__ == "__main__":
    csv_3t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv"
    csv_7t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/lpcnn_patches.csv"
    dir_3t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches"
    dir_7t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches"

    dataset = QSMLPCNNUnpairedDataset(
        csv_3t=csv_3t_path,
        csv_7t=csv_7t_path,
        data_dir_3t=dir_3t_path,
        data_dir_7t=dir_7t_path
    )
    
    # DataLoader initialized cleanly now
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Test batch fetch sanity check
    print("[*] Fetching first batch sample to check spatial dimensions...")
    batch = next(iter(dataloader))
    print(f" -> Batch input_field_3t shape: {batch['input_field_3t'].shape}")
    print(f" -> Batch target_qsm_3t shape:  {batch['target_qsm_3t'].shape}")
    print(f" -> Batch reference_qsm_7t shape: {batch['reference_qsm_7t'].shape}")
    print("✅ DataLoader verification successful!")

