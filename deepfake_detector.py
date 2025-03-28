import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

model_name = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def detect_deepfake(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    score = torch.softmax(logits, dim=-1)[0, 1].item()
    return "AI-generated Voice Detected" if score > 0.5 else "Human Voice Detected"
