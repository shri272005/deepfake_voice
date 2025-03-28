import os
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"wav", "mp3"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Validate file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Embed an imperceptible watermark and compute noise difference
def embed_watermark(audio_path, output_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        watermark_signal = np.random.uniform(-0.0005, 0.0005, len(y))  # Low-intensity watermark
        y_watermarked = y + watermark_signal
        sf.write(output_path, y_watermarked, sr)

        # Compute noise difference
        noise_difference = y_watermarked - y
        return output_path, noise_difference.tolist()
    except Exception as e:
        print(f"Error embedding watermark: {e}")
        return None, None

# AI-generated voice detection (Dummy function - Replace with ML model)
def detect_ai_generated(audio_path):
    return "Fake" if "ai_generated" in audio_path else "Real"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["audio"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process watermarking
        output_filename = "watermarked_" + filename
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        output_path, noise_difference = embed_watermark(filepath, output_path)

        if output_path is None:
            return jsonify({"error": "Watermarking failed"}), 500

        # AI Voice Detection
        detection_result = detect_ai_generated(filepath)

        # Get file sizes
        original_size = os.path.getsize(filepath)
        processed_size = os.path.getsize(output_path)

        return jsonify({
            "original_audio": filename,
            "processed_audio": output_filename,
            "original_size": original_size,
            "processed_size": processed_size,
            "detection_result": detection_result,
            "noise_difference": noise_difference[:500]  # Limit to avoid large response
        })

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

    

if __name__ == "__main__":
    app.run(debug=True)
