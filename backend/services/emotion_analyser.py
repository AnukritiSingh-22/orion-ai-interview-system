"""
emotion_analyser.py
-------------------
Audio-based emotion inference for ORION (replaces DeepFace facial emotion).

Used in two places:
  1. ml/train_emotion_model.py imports extract_features() so training and
     inference always compute features the same way.
  2. The pipeline calls analyse_emotion(wav_path) on the 16 kHz mono WAV that
     audio_extractor.py already produces.

If the trained model file is missing, analyse_emotion() returns None instead
of crashing, so the rest of the pipeline keeps working.
"""
import logging
import os
from pathlib import Path

import joblib
import librosa
import numpy as np

logger = logging.getLogger(__name__)

SR = 16000                     # matches audio_extractor.py (16 kHz mono)
CHUNK_SECONDS = 3.0            # RAVDESS clips are ~3-4 s, so use similar windows
MIN_CHUNK_SECONDS = 1.0

# Emotions counted as "positive" for the non-verbal score. Adjust if you like.
POSITIVE_EMOTIONS = {"happy", "calm", "neutral"}

BACKEND_DIR = Path(__file__).resolve().parents[1]
_bundle = None
_load_failed = False


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Fixed-length feature vector (96 values) from a mono waveform."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),        # 80
        chroma.mean(axis=1),                        # 12
        zcr.mean(), rms.mean(), rms.std(),          # 3
        centroid.mean(),                            # 1
    ])


def _model_path() -> Path:
    p = Path(os.getenv("EMOTION_MODEL_PATH", "ml/emotion_model.joblib"))
    return p if p.is_absolute() else BACKEND_DIR / p


def _load():
    global _bundle, _load_failed
    if _bundle is not None or _load_failed:
        return _bundle
    path = _model_path()
    if not path.exists():
        logger.warning("Emotion model not found at %s - emotion analysis disabled", path)
        _load_failed = True
        return None
    try:
        _bundle = joblib.load(path)
    except Exception as exc:  # corrupted file, version mismatch, etc.
        logger.error("Could not load emotion model: %s", exc)
        _load_failed = True
    return _bundle


def analyse_emotion(wav_path: str):
    """
    Returns:
        {
          "dominant_emotion": "calm",
          "positive_emotion_pct": 72.4,          # 0-100
          "emotion_probabilities": {"calm": 0.51, ...}
        }
        or None if the model is unavailable or the audio is too short/silent.
    """
    bundle = _load()
    if bundle is None:
        return None

    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception as exc:
        logger.error("Emotion analysis could not read %s: %s", wav_path, exc)
        return None

    chunk = int(CHUNK_SECONDS * SR)
    min_len = int(MIN_CHUNK_SECONDS * SR)
    pieces = [y[i:i + chunk] for i in range(0, len(y), chunk)]
    pieces = [p for p in pieces if len(p) >= min_len]
    if not pieces:
        return None

    feats = np.vstack([extract_features(p) for p in pieces])
    model = bundle["model"]
    classes = list(model.classes_)
    mean_proba = model.predict_proba(feats).mean(axis=0)

    probs = {c: float(p) for c, p in zip(classes, mean_proba)}
    positive = sum(p for c, p in probs.items() if c in POSITIVE_EMOTIONS)

    return {
        "dominant_emotion": max(probs, key=probs.get),
        "positive_emotion_pct": round(100.0 * positive, 1),
        "emotion_probabilities": probs,
    }
