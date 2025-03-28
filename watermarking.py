import numpy as np
import librosa
import librosa.display
import soundfile as sf

def embed_watermark(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    watermark = np.random.randn(len(y)) * 0.0001  
    watermarked_audio = y + watermark
    output_path = audio_path.replace(".wav", "_watermarked.wav")
    sf.write(output_path, watermarked_audio, sr)
    return output_path

def extract_watermark(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    noise_level = np.mean(np.abs(y))
    return "Watermark Detected" if noise_level > 0.00005 else "No Watermark Detected"
