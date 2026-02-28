import os
import argparse
import torch
import sys
# Make models accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stgcn import DummySTGCN
from train import DummySignLanguageDataset
from torch.utils.data import DataLoader

def evaluate(model_path, data_path, batch_size=16):
    print(f"Evaluating model loaded from: {model_path}")
    
    # Dataset
    dataset = DummySignLanguageDataset(data_path, split="test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Init and Load Model
    model = DummySTGCN(in_channels=3, num_classes=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print("Model file not found. Evaluating with random initialization.")
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / (total if total > 0 else 1)
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/stgcn_checkpoint.pth")
    parser.add_argument("--data_path", type=str, default="data/")
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path)
