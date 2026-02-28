# Real-Time Sign Language Translator (SOTA)

A robust real-time sign language recognition system using state-of-the-art models (Mediapipe for 21-keypoint extraction, ST-GCN/Transformer skeletons for gesture recognition). 

![Demo](demo.gif)

## Architecture Overview
1. **Keypoint Extraction:** Mediapipe Hands (robust real-time 21-point hand tracking).
2. **Action Recognition:** 
   - A PyTorch Scaffold for an ST-GCN (Spatial-Temporal Graph Convolutional Network) allowing highly accurate isolated gesture recognition.
   - For real-time prototype demonstration without full training, it employs a baseline dummy inference or classical ML model over extracted features. 
3. **Output Module:** GUI via PyQt5 coupled with pyttsx3 for continuous text and speech generation in a non-blocking background thread.

## Dataset
To train the SOTA models, it is recommended to use standard sign language datasets such as [WLASL](https://dxli94.github.io/WLASL/).

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Sign Language"
   ```

2. **Create a virtual environment & install requirements:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

## Adding New Gestures and Training
1. **Collect Data:** Record videos of the new gesture and extract landmarks using `utils/keypoints.py`.
2. **Update Dataset:** Place the processed `.npy` or tabular data into `data/`.
3. **Train Model:** Use the PyTorch scaffold located in `scripts/train.py`.
   ```bash
   python scripts/train.py --data_path data/
   ```
4. **Weights:** Update the saved model weights in `models/`.

## License
MIT
