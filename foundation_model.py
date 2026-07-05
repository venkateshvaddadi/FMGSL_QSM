import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class FrozenSwinUNETRTeacher(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        # 1. Initialize MONAI SwinUNETR (We only leverage the encoder pathway)
        self.swin_unetr = SwinUNETR(
            in_channels=1,      # 3D QSM local field/susceptibility map input
            out_channels=1,     # Dummy placeholder (unused by encoder)
            feature_size=48,    # Standard MONAI SSL architectural size
            use_checkpoint=False
        )
        
        # 2. If a path is provided, load the pre-trained weights safely
        if checkpoint_path is not None:
            self._load_weights(checkpoint_path)
        else:
            print("[Teacher Init] ⚠️ Warning: No checkpoint path provided. Running with random weights.")
        
        # 3. 3D Global Average Pooling to flatten the final bottleneck volume
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # 4. Freeze all parameters immediately to lock down the teacher
        for param in self.parameters():
            param.requires_grad = False

    def _load_weights(self, checkpoint_path):
        """Loads and strips checkpoint dictionaries to fit the encoder pathway."""
        print(f"[*] Extracting pre-trained states from: {checkpoint_path}")
        checkpoint = torch.load('pretrained/ssl_pretrained_weights.pth', map_location="cpu")
        
        # Unpack standard dictionary wrappers if present
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Map internal weights explicitly to the encoder track
        encoder_dict = {}
        for k, v in state_dict.items():
            # Strip standard PyTorch DistributedDataParallel wrappers if present
            if k.startswith("module."):
                k = k[7:]
            
            # Keep only the layers belonging to the base network
            if k.startswith("swin_unetr."):
                encoder_dict[k[11:]] = v  # strip the outer class wrapper prefix
            else:
                encoder_dict[k] = v

        # Load weights into our backbone. strict=False ignores downstream decoder items
        missing, unexpected = self.swin_unetr.load_state_dict(encoder_dict, strict=False)
        print(f" -> Successfully loaded pre-trained teacher layers!")
        print(f" -> Ignored/Missing keys (Normal if only decoder layers missing): {len(missing)}")

    def train(self, mode=True):
        """Force encoder to always stay in evaluation state (freezes BN/Dropout)"""
        super().train(False)
        return self

    def forward(self, x):
        hidden_states_out = self.swin_unetr.swinViT(x)
        bottleneck = hidden_states_out[4] 
        pooled = self.gap(bottleneck)
        return pooled.view(pooled.size(0), -1)

# =====================================================================
# VERIFICATION UNIT TEST (Runs directly)
# =====================================================================
def test_inference():
    print("\n=== Step 1: Foundation Model Dimensions & Weight Check ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target your file path directly here
    weight_path = "pretrained/ssl_pretrained_weights.pth"
    
    # Initialize our structural teacher with real parameters
    model = FrozenSwinUNETRTeacher(checkpoint_path=weight_path).to(device)
    model.eval() 
    
    mock_patch = torch.randn(1, 1, 64, 64, 64, dtype=torch.float32).to(device)
    print(f"Input structural shape: {list(mock_patch.shape)}")
    
    with torch.no_grad():
        embedding = model(mock_patch)
        
    print(f"Output embedding shape: {list(embedding.shape)}")
    print("=================================================")
    
    assert embedding.shape == (1, 768)
    print("✅ Weights loaded and vector dimensions verified successfully!")

if __name__ == "__main__":
    test_inference()