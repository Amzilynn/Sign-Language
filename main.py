"""
main.py
=======
Entry point for the Real-Time Sign Language Translator.

Launches the PyQt5 GUI, which captures webcam frames, extracts hand landmarks
via Mediapipe, and predicts sign language words using the trained ST-GCN model
(or mock predictions if the model has not been trained yet).

Usage:
    python main.py
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication

from utils.keypoints import HandKeypointExtractor
from utils.audio import TTSEngine
from gui.app import SignLanguageApp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def build_inference_fn():
    """
    Attempts to load the trained GesturePredictor.  Falls back to a
    simple mock if the model or classes.json are not yet available.
    """
    MODEL_PATH   = 'models/stgcn_best.pth'
    CLASSES_PATH = 'data/processed/classes.json'

    import os
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        from models.inference import GesturePredictor
        predictor = GesturePredictor(model_path=MODEL_PATH, classes_path=CLASSES_PATH)
        log.info("Using trained ST-GCN model for inference.")

        def real_inference(landmarks):
            """landmarks: np.ndarray, shape (hands, 21, 3)."""
            # Feed only the first hand into the frame buffer
            hand = landmarks[0] if landmarks is not None and len(landmarks) > 0 else None
            result = predictor.push_frame(hand)
            return result  # None while still buffering, str when prediction ready

        return real_inference

    else:
        import random
        MOCK_CLASSES = ["Hello", "Thank you", "Yes", "No", "Please", "I love you",
                        "Sorry", "Help", "More", "Stop"]
        log.warning(
            "Trained model not found. Using MOCK inference.\n"
            "Run:  python scripts/preprocess_dataset.py\n"
            "Then: python scripts/train.py"
        )

        def mock_inference(landmarks):
            return random.choice(MOCK_CLASSES)

        return mock_inference


def main():
    # 1. Initialize utilities
    log.info("Initializing Keypoint Extractor...")
    extractor = HandKeypointExtractor(max_num_hands=2)

    log.info("Initializing TTS Engine...")
    tts = TTSEngine()
    tts.speak("System initialized.")

    # 2. Build inference function
    inference_fn = build_inference_fn()

    # 3. Start GUI
    app = QApplication(sys.argv)
    window = SignLanguageApp(extractor, tts, model_inference_fn=inference_fn)
    window.show()

    log.info("Application loop started.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
