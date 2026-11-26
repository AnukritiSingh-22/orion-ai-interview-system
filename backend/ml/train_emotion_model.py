"""
train_emotion_model.py
----------------------
Trains the speech-emotion model on the RAVDESS *speech* dataset.

Run from the backend folder:
    python ml/train_emotion_model.py

Optional:
    python ml/train_emotion_model.py --actors 12      # use only first 12 actors (faster)
    python ml/train_emotion_model.py --data ml/ravdess

RAVDESS filename format:  03-01-06-01-02-01-12.wav
    field 3 = emotion (01 neutral, 02 calm, 03 happy, 04 sad,
                       05 angry, 06 fearful, 07 disgust, 08 surprised)
    field 7 = actor number (01-24)

The train/test split is done BY ACTOR so the same speaker never appears in
both sets (otherwise accuracy looks better than it really is).
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Make "services" importable when running `python ml/train_emotion_model.py`
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from services.emotion_analyser import SR, extract_features  # noqa: E402

EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


def load_dataset(data_dir: Path, max_actors: int):
    files = sorted(data_dir.rglob("*.wav"))
    if not files:
        sys.exit(f"No .wav files found under {data_dir}. Check the extraction step.")

    X, y, groups = [], [], []
    skipped = 0
    start = time.time()
    for i, f in enumerate(files, 1):
        parts = f.stem.split("-")
        if len(parts) != 7 or parts[2] not in EMOTIONS:
            skipped += 1
            continue
        actor = int(parts[6])
        if actor > max_actors:
            continue
        try:
            wav, _ = librosa.load(f, sr=SR, mono=True)
            wav, _ = librosa.effects.trim(wav, top_db=30)
            if len(wav) < SR // 2:
                skipped += 1
                continue
            X.append(extract_features(wav))
            y.append(EMOTIONS[parts[2]])
            groups.append(actor)
        except Exception as exc:
            print(f"  skipped {f.name}: {exc}")
            skipped += 1
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} files ({time.time() - start:.0f}s)")

    print(f"Loaded {len(X)} clips from {len(set(groups))} actors (skipped {skipped}).")
    return np.array(X), np.array(y), np.array(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BACKEND_DIR / "ml" / "ravdess"),
                    help="Folder containing the extracted Actor_XX folders")
    ap.add_argument("--actors", type=int, default=24, help="Use actors 1..N (default 24)")
    ap.add_argument("--out", default=str(BACKEND_DIR / "ml" / "emotion_model.joblib"))
    args = ap.parse_args()

    print("Extracting features...")
    X, y, groups = load_dataset(Path(args.data), args.actors)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    print(f"Train actors: {sorted(set(groups[train_idx]))}")
    print(f"Test actors:  {sorted(set(groups[test_idx]))}")

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    )
    print("Training...")
    model.fit(X[train_idx], y[train_idx])

    pred = model.predict(X[test_idx])
    print(f"\nHeld-out accuracy: {accuracy_score(y[test_idx], pred):.3f}")
    print(classification_report(y[test_idx], pred, zero_division=0))

    # Refit on all data for the saved model (the report above is the honest number)
    model.fit(X, y)
    joblib.dump({"model": model, "sr": SR}, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
