"""
inference.py
============
Loads a trained ST-GCN checkpoint and the class map (classes.json),
then runs real-time inference on hand landmarks extracted by Mediapipe.

Returns the top-1 predicted word label.
"""

import os, sys, json, logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stgcn import DummySTGCN

log = logging.getLogger(__name__)


class GesturePredictor:
    """
    Wraps the trained ST-GCN for single-gesture inference.
    Accumulates landmark frames across the observation window and
    returns a prediction once the buffer is full.
    """

    def __init__(self,
                 model_path: str = 'models/stgcn_best.pth',
                 classes_path: str = 'data/processed/classes.json',
                 target_frames: int = 60,
                 device: str = None):

        self.target_frames = target_frames
        self.buffer = []  # list of (21, 3) landmark arrays

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Load class map
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                self.idx_to_class = {int(k): v for k, v in json.load(f).items()}
            num_classes = len(self.idx_to_class)
        else:
            log.warning("classes.json not found — using 2000 placeholder classes.")
            self.idx_to_class = {i: f"class_{i}" for i in range(2000)}
            num_classes = 2000

        # Load model
        self.model = DummySTGCN(in_channels=3, num_classes=num_classes)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            log.info(f"Loaded trained model from {model_path}")
        else:
            log.warning(f"Model weights not found at {model_path}. Using random init.")
        self.model.to(self.device).eval()

    def push_frame(self, landmarks: np.ndarray) -> tuple[str, float] | None:
        """
        Push one frame of landmarks (shape 21x3) into the sliding window.
        Returns (label, confidence) every 10 frames once the window is full.
        """
        if landmarks is not None and landmarks.shape == (21, 3):
            self.buffer.append(landmarks.astype(np.float32))
            log.debug("Frame pushed: [Hand Detected]")
        else:
            self.buffer.append(np.zeros((21, 3), dtype=np.float32))
            log.debug("Frame pushed: [No Hand]")

        # Keep buffer to sliding window size
        if len(self.buffer) > self.target_frames:
            self.buffer.pop(0)

        # Triger prediction only when window is full AND every N frames to save CPU
        if len(self.buffer) == self.target_frames:
            if not hasattr(self, '_pred_counter'): self._pred_counter = 0
            self._pred_counter += 1
            
            if self._pred_counter >= 10:
                self._pred_counter = 0
                return self._predict()

        return None

    def _predict(self) -> tuple[str, float]:
        """Run inference on a filled buffer. Returns (label, confidence)."""
        arr = np.array(self.buffer[:self.target_frames], dtype=np.float32)  # (T, 21, 3)
        arr = np.transpose(arr, (2, 0, 1))  # (3, T, 21)
        tensor = torch.tensor(arr).unsqueeze(0).to(self.device)  # (1, 3, T, 21)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf = conf.item()
            pred_idx = pred_idx.item()

        label = self.idx_to_class.get(pred_idx, f"unknown({pred_idx})")
        log.info(f"Prediction: {label} (Conf: {conf:.4f})")
        
        return label, conf
