import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/ai_detector.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# Test with an audio file
audio_path = "test_audio.wav"
features = extract_features(audio_path)
prediction = model.predict(features)[0][0]

if prediction > 0.5:
    print(f"Prediction: AI-Generated Voice ({prediction:.2f})")
else:
    print(f"Prediction: Human Voice ({1 - prediction:.2f})")
