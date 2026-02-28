import torch
import torch.nn as nn

class TransformerGestureModel(nn.Module):
    """
    A Transformer encoder-based sequential skeleton model scaffold.
    Often used as an alternative or in tandem with GCN models to handle 
    long-range temporal dependencies in continuous sign language.
    """
    def __init__(self, keypoints=21, coords=3, num_classes=50, d_model=128, nhead=8, num_layers=4):
        super(TransformerGestureModel, self).__init__()
        
        # Input features per frame: 21 points * 3 coord (x,y,z) = 63
        in_features = keypoints * coords
        
        # Project raw keypoints to d_model
        self.input_proj = nn.Linear(in_features, d_model)
        
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: shape [batch_size, sequence_length, keypoints * coords]
        """
        # Projection
        x = self.input_proj(x)
        
        # Transformer pass
        x = self.transformer_encoder(x)
        
        # Pool across sequence dimension (e.g., mean pooling)
        x = x.mean(dim=1)
        
        # Predict class
        logits = self.classifier(x)
        return logits
