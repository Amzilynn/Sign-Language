import kagglehub
import os

def download():
    print("Starting download...")
    # Using the handle provided by the user (corrected if necessary)
    # The user provided: "hossammbalaha/asl-20-words-dataset-v1"
    # But search results said: "hossambalaha/arabic-sign-language-arsl-20-words-dataset-v1"
    # I'll try both or the specific one from search first.
    try:
        path = kagglehub.dataset_download("hossambalaha/arabic-sign-language-arsl-20-words-dataset-v1")
        print(f"SUCCESS_PATH:{path}")
    except Exception as e:
        print(f"Error downloading hossambalaha: {e}")
        try:
            # Fallback to the one the user typed exactly
            path = kagglehub.dataset_download("hossammbalaha/asl-20-words-dataset-v1")
            print(f"SUCCESS_PATH:{path}")
        except Exception as e2:
            print(f"Error downloading hossammbalaha: {e2}")

if __name__ == "__main__":
    download()
