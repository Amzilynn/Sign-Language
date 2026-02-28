import os
import sys
import torch
import numpy as np
import onnxruntime as ort

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stgcn import DummySTGCN

def verify_onnx(checkpoint_path, onnx_path):
    print("Loading PyTorch model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_classes = checkpoint.get('num_classes', 3)
    
    torch_model = DummySTGCN(in_channels=3, num_classes=num_classes)
    torch_model.load_state_dict(checkpoint['model_state_dict'])
    torch_model.eval()
    
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    dummy_input = torch.randn(1, 3, 60, 21)
    
    # PyTorch Inference
    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()
    
    # ONNX Inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    # Compare
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print("SUCCESS: PyTorch and ONNX outputs match!")
    
    # Test with real data if available
    X_path = "data/processed/X.npy"
    if os.path.exists(X_path):
        X = np.load(X_path)
        if len(X) > 0:
            sample = X[0:1].astype(np.float32)
            torch_out_real = torch_model(torch.from_numpy(sample)).detach().numpy()
            ort_out_real = ort_session.run(None, {'input': sample})[0]
            
            print(f"PyTorch Argmax: {np.argmax(torch_out_real)}")
            print(f"ONNX Argmax:    {np.argmax(ort_out_real)}")
            
            if np.argmax(torch_out_real) == np.argmax(ort_out_real):
                print("SUCCESS: Real data classification match!")
            else:
                print("WARNING: Real data classification mismatch.")

if __name__ == "__main__":
    cp_path = "models/stgcn_best.pth"
    on_path = "models/stgcn.onnx"
    
    if not os.path.exists(cp_path) or not os.path.exists(on_path):
        print("Missing files for verification.")
        sys.exit(1)
        
    verify_onnx(cp_path, on_path)
