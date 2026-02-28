import cv2
import os
import time
import argparse

def record_word(word, num_samples=5, save_dir="data/videos/custom"):
    cap = cv2.VideoCapture(0)
    word_dir = os.path.join(save_dir, word.lower())
    os.makedirs(word_dir, exist_ok=True)

    print(f"\nRecording {num_samples} samples for word: '{word}'")
    print("Prepare your signs! We will record 2 seconds (60 frames) per sample.")
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        # Countdown
        for count in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Ready? {count}", (100, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.imshow("Recorder", frame)
            cv2.waitKey(1000)

        # Recording
        filename = os.path.join(word_dir, f"sample_{i+1}_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))

        print("Recording...")
        for _ in range(60): # 2 seconds
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Show "RECORDING" on screen
            disp = frame.copy()
            cv2.putText(disp, "RECORDING", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Recorder", disp)
            
            out.write(frame)
            cv2.waitKey(1)
        
        out.release()
        print(f"Saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nRecording complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record your own sign language videos.")
    parser.add_argument("--word", type=str, required=True, help="The word you want to sign.")
    parser.add_argument("--samples", type=int, default=5, help="Number of times to record the sign.")
    args = parser.parse_args()
    
    record_word(args.word, args.samples)
