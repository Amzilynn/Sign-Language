import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class SignLanguageApp(QMainWindow):
    def __init__(self, keypoint_extractor, tts_engine, model_inference_fn=None):
        super().__init__()
        self.extractor = keypoint_extractor
        self.tts = tts_engine
        self.model_inference_fn = model_inference_fn
        
        # UI Setup
        self.setWindowTitle("Real-Time Sign Language Translator")
        self.setGeometry(100, 100, 800, 700)
        
        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Title Label
        self.title_label = QLabel("Sign Language Translator", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(self.title_label)
        
        # Video Feed Label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #ccc; background-color: #000;")
        self.video_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.video_label)
        
        # Recognized Text Display
        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMaximumHeight(80)
        self.text_display.setStyleSheet("font-size: 18px; padding: 5px;")
        self.layout.addWidget(self.text_display)

        # Status Label
        self.status_label = QLabel("Status: Ready", self)
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.layout.addWidget(self.status_label)

        # Live Guess Label
        self.live_guess_label = QLabel("Live Guess: ---", self)
        self.live_guess_label.setStyleSheet("font-size: 16px; color: #555;")
        self.layout.addWidget(self.live_guess_label)
        
        # Button Layout
        self.btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        
        self.btn_layout.addWidget(self.start_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.layout.addLayout(self.btn_layout)
        
        # Video Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        
        # Prediction stability tracking
        self.frame_counter = 0
        self.last_prediction = None
        self.prediction_count = 0
        self.stable_threshold = 5  # Word must be the same for 5 consecutive predictions (~1.5s)
        self.last_spoken_word = None

    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30) # 30 ms ~ 33 fps

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1) # Mirror
        
        # Extract features and draw annotations
        annotated_frame, landmarks = self.extractor.process_frame(frame)
        
        # Update Status
        if landmarks is not None and len(landmarks) > 0:
            self.status_label.setText(f"Status: DETECTING ({len(landmarks)} hand(s))")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("Status: NO HANDS DETECTED")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

        # Run real/mock inference
        result = None
        if self.model_inference_fn:
            # Push every frame to the inference engine
            result = self.model_inference_fn(landmarks)
        else:
            # Mock logic: still every 30 frames
            if landmarks is not None and len(landmarks) > 0:
                self.frame_counter += 1
                if self.frame_counter % 30 == 0:
                    result = ("Hello", 0.99)

        if result:
            label, confidence = result
            self.live_guess_label.setText(f"Live Guess: {label} ({confidence*100:.1f}%)")
            
            # Stability Logic: Only "commit" the word if it's stable and high confidence
            if confidence > 0.4:
                if label == self.last_prediction:
                    self.prediction_count += 1
                else:
                    self.last_prediction = label
                    self.prediction_count = 1
                
                # If word is stable and hasn't just been spoken
                if self.prediction_count >= self.stable_threshold:
                    if label != self.last_spoken_word:
                        self.text_display.append(f"<b>Detected: {label.upper()}</b>")
                        self.tts.speak(label)
                        self.last_spoken_word = label
                        # Reset stability to avoid repeating until it changes
                        self.status_label.setText(f"Status: TRANSLATED -> {label.upper()}")
                        self.status_label.setStyleSheet("color: darkgreen; font-weight: bold; font-size: 20px;")
            else:
                # Low confidence or different word
                if confidence < 0.2:
                    self.last_prediction = None
                    self.prediction_count = 0
                    if self.last_spoken_word:
                        # Allow re-triggering the same word if we "lost" it
                        self.last_spoken_word = None

        # Convert to QImage for PyQt
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.stop_camera()
        self.extractor.release()
        self.tts.stop()
        event.accept()
