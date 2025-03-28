import tarfile
import os

# Correct the path to your dataset file
ZIP_FILE_PATH = r"D:\AI_voice_deepfake\cv-corpus-21.0-delta-2025-03-14-hi.tar"
EXTRACT_FOLDER = r"D:\AI_voice_deepfake\dataset"

# Ensure the extract folder exists
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# Extract TAR file
try:
    with tarfile.open(ZIP_FILE_PATH, 'r') as tar_ref:  # Use 'r:gz' if it's a .tar.gz file
        tar_ref.extractall(EXTRACT_FOLDER)
    print("✅ Dataset extracted successfully!")
except FileNotFoundError:
    print(f"❌ Error: The file '{ZIP_FILE_PATH}' was not found.")
except tarfile.ReadError:
    print(f"❌ Error: '{ZIP_FILE_PATH}' is not a valid TAR file.")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
