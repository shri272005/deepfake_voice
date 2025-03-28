import librosa
import numpy as np
import os
import pandas as pd

DATASET_PATH = "dataset"
OUTPUT_FILE = "audio_features.csv"

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

data = []
for label in ["AI", "Human"]:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        features = extract_features(file_path)
        if features is not None:
            data.append([file] + list(features) + [label])

# Save to CSV
columns = ["Filename"] + [f"MFCC_{i}" for i in range(40)] + ["Label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Feature extraction complete. Saved to {OUTPUT_FILE}")
