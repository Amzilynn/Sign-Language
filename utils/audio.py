import os
import queue
import threading
import logging
import time
from gtts import gTTS
import pygame

class TTSEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logging.info("TTS Engine (gTTS + Pygame) initialized.")

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                # Block for up to 1 second waiting for new text
                text = self.speech_queue.get(timeout=1.0)
                if text:
                    logging.info(f"TTS Thread: Processing '{text}'")
                    
                    # Create temporary mp3 file
                    temp_file = f"temp_speech_{int(time.time())}.mp3"
                    try:
                        tts = gTTS(text=text, lang='en')
                        tts.save(temp_file)
                        
                        # Play via pygame
                        pygame.mixer.music.load(temp_file)
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        pygame.mixer.music.unload()
                        logging.info(f"TTS Thread: Finished speaking '{text}'")
                    except Exception as e:
                        logging.error(f"TTS Playback Error: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS Loop Error: {e}")

    def speak(self, text):
        """Add text to the speech queue."""
        if text and text.strip():
            self.speech_queue.put(text)

    def stop(self):
        """Stop the TTS engine."""
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        pygame.mixer.quit()
