import os

import joblib
import librosa
import numpy as np

MODEL_FILENAME = "random_forest_model.pkl"

loaded_model = joblib.load(MODEL_FILENAME)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rmse = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfccs_mean, spectral_centroid, chroma, zero_crossing_rate, rmse])


def predict_fake_or_real(wav_file, model=None):
    model = model or loaded_model
    features = extract_features(wav_file).reshape(1, -1)
    return "Fake" if int(model.predict(features)[0]) == 1 else "Real"


def voice_detector(input_voice):
    if not os.path.exists(input_voice):
        raise FileNotFoundError(f"Audio file not found: {input_voice}")
    return predict_fake_or_real(input_voice)
