import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("audio_features.csv")

# Prepare features and labels
X = df.iloc[:, 1:-1].values  # MFCC features
y = df["Label"].values        # AI or Human

# Encode labels (AI = 1, Human = 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(40,)), 
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary Classification (AI or Human)
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Save Model
model.save("models/ai_detector.h5")
print("Model training complete. Saved as 'models/ai_detector.h5'")
