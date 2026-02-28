import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stgcn import DummySTGCN

def export_onnx(checkpoint_path, onnx_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    num_classes = checkpoint.get('num_classes', 3)
    print(f"Detected {num_classes} classes.")
    
    model = DummySTGCN(in_channels=3, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Input shape: (Batch, Channels, Time, Vertices)
    # C=3, T=60, V=21
    dummy_input = torch.randn(1, 3, 60, 21)
    
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    cp_path = "models/stgcn_best.pth"
    out_path = "models/stgcn.onnx"
    
    if not os.path.exists(cp_path):
        print(f"Error: Checkpoint {cp_path} not found.")
        sys.exit(1)
        
    export_onnx(cp_path, out_path)
