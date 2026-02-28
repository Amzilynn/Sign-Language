"""
preprocess_dataset.py
=====================
Extracts Mediapipe hand landmarks from every MP4 video in the dataset and
saves them as .npy files ready for the ST-GCN / Transformer training pipeline.

Expected input structure
    data/videos/videos/
        a lot.mp4
        able.mp4
        ...
        video_letters/    (alphabet - optional)
            a.mp4
            b.mp4
            ...

Output structure
    data/processed/
        X.npy   – shape (N, 3, T, 21)   float32  (samples, channels, frames, keypoints)
        y.npy   – shape (N,)             int64    (class index per sample)
        classes.json                               (index -> word label mapping)

Usage
    python scripts/preprocess_dataset.py
"""

import os, sys, json, argparse, logging
import cv2
import numpy as np

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.keypoints import HandKeypointExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def extract_landmarks_from_video(video_path: str,
                                  extractor: HandKeypointExtractor,
                                  target_frames: int = 60) -> np.ndarray:
    """
    Run Mediapipe on every frame of a video and return a (3, T, 21) array.

    Returns
        np.ndarray of shape (3, target_frames, 21), dtype float32
        All-zeros if no hand is ever detected.
    """
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, landmarks = extractor.process_frame(frame)
        if landmarks is not None and len(landmarks) > 0:
            frames_data.append(landmarks[0])          # first hand  -> (21, 3)
        else:
            frames_data.append(np.zeros((21, 3), dtype=np.float32))

    cap.release()

    if len(frames_data) == 0:
        return np.zeros((3, target_frames, 21), dtype=np.float32)

    frames_data = np.array(frames_data, dtype=np.float32)   # (T_actual, 21, 3)

    # Pad or truncate to target_frames
    T_actual = frames_data.shape[0]
    if T_actual < target_frames:
        pad = np.zeros((target_frames - T_actual, 21, 3), dtype=np.float32)
        frames_data = np.concatenate([frames_data, pad], axis=0)
    else:
        frames_data = frames_data[:target_frames]

    # (target_frames, 21, 3) -> (3, target_frames, 21)   i.e. (C, T, V)
    frames_data = np.transpose(frames_data, (2, 0, 1))
    return frames_data


def preprocess(videos_dir: str,
               output_dir: str,
               target_frames: int = 60,
               include_letters: bool = False):
    """
    Iterate every video in videos_dir, extract landmarks, and save to output_dir.
    Word-level videos are single .mp4 files directly inside videos_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = HandKeypointExtractor(static_image_mode=False, max_num_hands=1)

    X_list, y_list = [], []
    class_to_idx = {}
    idx_to_class = {}

    log.info(f"Searching for videos in {videos_dir}...")
    
    mp4_files_with_labels = []
    
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                video_path = os.path.join(root, f)
                # If it's in a subfolder, use subfolder name as label. 
                # Otherwise, use filename.
                relative_path = os.path.relpath(video_path, videos_dir)
                parts = relative_path.split(os.sep)
                
                if len(parts) > 1:
                    label = parts[0].lower().strip() # Use folder name if nested
                else:
                    label = os.path.splitext(f)[0].lower().strip() # Use filename if top-level
                
                mp4_files_with_labels.append((video_path, label))

    log.info(f"Found {len(mp4_files_with_labels)} video files. Starting extraction...")

    for i, (video_path, label) in enumerate(mp4_files_with_labels):

        if label not in class_to_idx:
            idx = len(class_to_idx)
            class_to_idx[label] = idx
            idx_to_class[idx] = label

        features = extract_landmarks_from_video(video_path, extractor, target_frames)
        X_list.append(features)
        y_list.append(class_to_idx[label])

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(mp4_files)}] processed...")

    extractor.release()

    X = np.stack(X_list, axis=0)   # (N, 3, T, 21)
    y = np.array(y_list, dtype=np.int64)  # (N,)

    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(idx_to_class, f, indent=2)

    log.info(f"Done! Saved {X.shape[0]} samples.")
    log.info(f"  X.npy shape : {X.shape}  (N, C, T, V)")
    log.info(f"  y.npy shape : {y.shape}")
    log.info(f"  Num classes : {len(class_to_idx)}")
    log.info(f"  Output dir  : {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ASL skeletal videos into landmark arrays.')
    parser.add_argument('--videos_dir',    type=str, default='data/videos/videos',
                        help='Path to the folder containing the MP4 word videos.')
    parser.add_argument('--output_dir',   type=str, default='data/processed',
                        help='Where to save X.npy, y.npy and classes.json.')
    parser.add_argument('--target_frames', type=int, default=60,
                        help='Fixed temporal length to pad/truncate every video to.')
    parser.add_argument('--include_letters', action='store_true',
                        help='Also include the video_letters alphabet subfolder.')
    args = parser.parse_args()
    preprocess(args.videos_dir, args.output_dir, args.target_frames, args.include_letters)
