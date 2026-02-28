import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class HandKeypointExtractor:
    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Processes a BGR image frame and returns the annotated frame and extracted landmarks.
        Returns:
            annotated_image: The image with landmarks drawn.
            landmarks_array: A numpy array of shape (num_hands, 21, 3) where the last dimension is (x, y, z).
                             Returns None if no hands detected.
        """
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        
        # Draw the hand annotations on the original frame
        annotated_image = frame.copy()
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Extract coordinates
                hand_points = []
                for lm in hand_landmarks.landmark:
                    hand_points.append([lm.x, lm.y, lm.z])
                landmarks_list.append(hand_points)

            return annotated_image, np.array(landmarks_list)
        
        return annotated_image, None

    def release(self):
        self.hands.close()
