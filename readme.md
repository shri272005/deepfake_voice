AI Voice Deepfake Detection & Watermarking System
📌 Project Overview
This project aims to detect deepfake voices and apply tamper-resistant watermarking to audio files. It includes AI-based detection, watermark embedding, and noise difference visualization.

AI_voice_deepfake/
│── app.py                   # Main Flask application
│── deepfake_detector.py      # AI model for detecting deepfake audio
│── extract_dataset.py        # Dataset preprocessing script
│── feature_extraction.py     # Extracts features from audio for AI detection
│── test_model.py             # Tests trained deepfake detection model
│── train_model.py            # Script for training AI model
│── watermarking.py           # Implements watermark embedding in audio
│── processed/                # Stores processed (watermarked) audio files
|── uploads/                  # Stores uploaded audio files

🚀 Features
✔ Deepfake Detection - Uses AI to distinguish between real and AI-generated voices.
✔ Audio Watermarking - Embeds an imperceptible watermark for tracking.
✔ Noise Visualization - Displays differences between original and watermarked audio.
✔ User-Friendly UI - Web-based interface for easy upload and analysis.

Clone or Download the Repository
git clone https://github.com/your-repo/AI_voice_deepfake.git
cd AI_voice_deepfake

Install Required Dependencies
pip install -r requirements.txt

Run the Web Application
python app.py

Access the Web Interface
Open http://127.0.0.1:5000/ in your browser.

🛠 Technologies Used
Python (Flask, PyTorch, librosa)
Chart.js (Visualization)
Web Audio API (Playback & Processing)

