# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader  # <-- FIX: Explicitly import DataLoader here
# import scipy.io as sio
# import random

# class QSMLPCNNUnpairedDataset(Dataset):
#     """
#     Unpaired 3D Patch Dataset for Stage-2 Domain-Aware Contrastive Learning.
#     Independently samples 3T items from train.csv and 7T reference patches from lpcnn_patches.csv.
#     """
#     def __init__(self, csv_3t, csv_7t, data_dir_3t, data_dir_7t):
#         """
#         Args:
#             csv_3t (str): Path to train.csv tracking the 3T patches.
#             csv_7t (str): Path to lpcnn_patches.csv tracking the 7T LPCNN patches.
#             data_dir_3t (str): Directory where 3T .mat patches are stored.
#             data_dir_7t (str): Directory where 7T LPCNN .mat patches are stored.
#         """
#         self.data_dir_3t = data_dir_3t
#         self.data_dir_7t = data_dir_7t
        
#         # Load both tracking records
#         self.df_3t = pd.read_csv(csv_3t)
#         self.df_7t = pd.read_csv(csv_7t)
        
#         self.files_3t = self.df_3t.iloc[:, 0].tolist()
#         self.files_7t = self.df_7t.iloc[:, 0].tolist()

#         print(f"[Dataset Init] Loaded Unpaired Configuration:")
#         print(f" -> 3T Patch Count: {len(self.files_3t)}")
#         print(f" -> 7T LPCNN Patch Count: {len(self.files_7t)}")

#     def __len__(self):
#         # Determine the epoch length by the size of your primary 3T training set
#         return len(self.files_3t)

#     def __getitem__(self, idx):
#         # 1. Fetch the structured 3T patch by index sequential entry
#         filename_3t = self.files_3t[idx]
#         path_3t = os.path.join(self.data_dir_3t, filename_3t)
#         mat_3t = sio.loadmat(path_3t)
        
#         # 2. Fetch a random 7T LPCNN patch to provide domain contrast diversity
#         filename_7t = random.choice(self.files_7t)
#         path_7t = os.path.join(self.data_dir_7t, filename_7t)
#         mat_7t = sio.loadmat(path_7t)
        
#         # --- Internal Key Selection Engine ---
#         # Automatically finds variables inside the Matlab files regardless of naming differences
#         key_field_3t = 'field_3t' if 'field_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][0]
#         key_qsm_3t = 'qsm_3t' if 'qsm_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][1]
#         key_qsm_7t = 'qsm_7t' if 'qsm_7t' in mat_7t else [k for k in mat_7t.keys() if not k.startswith('__')][0]
        
#         # Extract and convert matrices to float values
#         field_3t = mat_3t[key_field_3t].astype(float)
#         qsm_3t = mat_3t[key_qsm_3t].astype(float)
#         qsm_7t = mat_7t[key_qsm_7t].astype(float)

#         # 3. Shape arrays into PyTorch standard formats: [1, 64, 64, 64]
#         field_3t_tensor = torch.from_numpy(field_3t).unsqueeze(0).float()
#         qsm_3t_tensor = torch.from_numpy(qsm_3t).unsqueeze(0).float()
#         qsm_7t_tensor = torch.from_numpy(qsm_7t).unsqueeze(0).float()

#         return {
#             "input_field_3t": field_3t_tensor,
#             "target_qsm_3t": qsm_3t_tensor,
#             "reference_qsm_7t": qsm_7t_tensor
#         }
# #%%

# # Instantiate the new unpaired dataset handler
# dataset = QSMLPCNNUnpairedDataset(
#     csv_3t="/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv",
#     csv_7t="/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/lpcnn_patches.csv",
#     data_dir_3t="/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches",
#     data_dir_7t="/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches"
# )

# # Keep the standard loader configuration intact
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)


#%%
import os
import random
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader  # <-- FIX: Explicitly import DataLoader here

class QSMLPCNNUnpairedDataset(Dataset):
    """
    Unpaired 3D Patch Dataset for Stage-2 Domain-Aware Contrastive Learning.
    Independently samples 3T items from train.csv and 7T reference patches from lpcnn_patches.csv.
    """
    def __init__(self, csv_3t, csv_7t, data_dir_3t, data_dir_7t):
        """
        Args:
            csv_3t (str): Path to train.csv tracking the 3T patches.
            csv_7t (str): Path to lpcnn_patches.csv tracking the 7T LPCNN patches.
            data_dir_3t (str): Directory where 3T .mat patches are stored.
            data_dir_7t (str): Directory where 7T LPCNN .mat patches are stored.
        """
        self.data_dir_3t = data_dir_3t
        self.data_dir_7t = data_dir_7t
        
        # Load both tracking records
        self.df_3t = pd.read_csv(csv_3t)
        self.df_7t = pd.read_csv(csv_7t)
        
        self.files_3t = self.df_3t.iloc[:, 0].tolist()
        self.files_7t = self.df_7t.iloc[:, 0].tolist()

        print(f"[Dataset Init] Loaded Unpaired Configuration:")
        print(f" -> 3T Patch Count: {len(self.files_3t)}")
        print(f" -> 7T LPCNN Patch Count: {len(self.files_7t)}")

    def __len__(self):
        # Determine the epoch length by the size of your primary 3T training set
        return len(self.files_3t)

    def __getitem__(self, idx):
        # 1. Fetch the structured 3T patch by index sequential entry
        filename_3t = self.files_3t[idx]
        path_3t = os.path.join(self.data_dir_3t, filename_3t)
        mat_3t = sio.loadmat(path_3t)
        
        # 2. Fetch a random 7T LPCNN patch to provide domain contrast diversity
        filename_7t = random.choice(self.files_7t)
        path_7t = os.path.join(self.data_dir_7t, filename_7t)
        mat_7t = sio.loadmat(path_7t)
        
        # --- Internal Key Selection Engine ---
        # Automatically finds variables inside the Matlab files regardless of naming differences
        key_field_3t = 'field_3t' if 'field_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][0]
        key_qsm_3t = 'qsm_3t' if 'qsm_3t' in mat_3t else [k for k in mat_3t.keys() if not k.startswith('__')][1]
        key_qsm_7t = 'qsm_7t' if 'qsm_7t' in mat_7t else [k for k in mat_7t.keys() if not k.startswith('__')][0]
        
        # Extract and convert matrices to float values
        field_3t = mat_3t[key_field_3t].astype(float)
        qsm_3t = mat_3t[key_qsm_3t].astype(float)
        qsm_7t = mat_7t[key_qsm_7t].astype(float)

        # 3. Shape arrays into PyTorch standard formats: [1, 64, 64, 64]
        field_3t_tensor = torch.from_numpy(field_3t).unsqueeze(0).float()
        qsm_3t_tensor = torch.from_numpy(qsm_3t).unsqueeze(0).float()
        qsm_7t_tensor = torch.from_numpy(qsm_7t).unsqueeze(0).float()

        return {
            "input_field_3t": field_3t_tensor,
            "target_qsm_3t": qsm_3t_tensor,
            "reference_qsm_7t": qsm_7t_tensor
        }

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
    print(f" -> Batch reference_qsm_7t shape: {batch['reference_qsm_7t'].shape}")
    print("✅ DataLoader verification successful!")
#%%
import time
import torch
from torch.utils.data import DataLoader

def test_dataset_iterator():
    print("="*70)
    print("RUNNING 3D PATCH DATA LOADER ITERATOR TEST")
    print("="*70)
    
    # 1. Define your verified directory configuration paths
    csv_3t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_source_1/csv_files/train.csv"
    csv_7t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/lpcnn_patches.csv"
    dir_3t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/given_data/data_as_patches"
    dir_7t_path = "/media/venkatesh/DATA/venkatesh/IISc/MIG_LAB_WORK/Experiments/QSM_venkatesh/QSM_data/data_for_experiments/lpcnn_data_for_training/data_as_patches"

    # 2. Instantiate the Dataset handler module
    try:
        dataset = QSMLPCNNUnpairedDataset(
            csv_3t=csv_3t_path,
            csv_7t=csv_7t_path,
            data_dir_3t=dir_3t_path,
            data_dir_7t=dir_7t_path
        )
    except Exception as e:
        print(f"❌ Failed during Dataset initiation step: {e}")
        return

    # 3. Create the PyTorch DataLoader iterator
    # num_workers=2 handles asynchronous data loading in the background
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        drop_last=True,
        num_workers=2
    )

    print(f"[*] Starting iteration check loop...")
    print(f"[*] Configured Batch Size: 4")
    print(f"[*] Total micro-batches available in data loader: {len(dataloader)}")
    print("-" * 50)

    # 4. Iterate over a few sample steps to measure delivery latency speeds
    max_test_steps = 5
    start_time = time.time()
    
    for step, batch in enumerate(dataloader, 1):
        step_start = time.time()
        
        # Unpack tensors
        input_3t = batch["input_field_3t"]
        target_3t = batch["target_qsm_3t"]
        reference_7t = batch["reference_qsm_7t"]
        
        step_duration = time.time() - step_start
        
        print(f"Step [{step}/{max_test_steps}] | Load Time: {step_duration:.4f}s")
        print(f"   -> input_field_3t  shape: {list(input_3t.shape)}   | Type: {input_3t.dtype}")
        print(f"   -> target_qsm_3t   shape: {list(target_3t.shape)}   | Type: {target_3t.dtype}")
        print(f"   -> reference_qsm_7t shape: {list(reference_7t.shape)}   | Type: {reference_7t.dtype}")
        print("-" * 50)
        
        # Sizing integrity assertion test checks
        assert input_3t.shape == (4, 1, 64, 64, 64), "Input 3T Field dimensions are incorrect!"
        assert reference_7t.shape == (4, 1, 64, 64, 64), "LPCNN 7T Reference dimensions are incorrect!"
        
        if step >= max_test_steps:
            break

    total_duration = time.time() - start_time
    print(f"✅ Success! Iterator pipeline processed {max_test_steps} steps in {total_duration:.2f}s.")
    print(f" -> Average batch delivery rate: {total_duration / max_test_steps:.4f} seconds per batch.")
    print("="*70)

if __name__ == "__main__":
    test_dataset_iterator()